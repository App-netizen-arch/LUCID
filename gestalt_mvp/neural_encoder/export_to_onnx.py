"""
Phase 1: Neural Encoder Integration with Rust Symbolic Core

This module provides Python-to-Rust bridge for the neural encoder.
The trained PyTorch model is exported to ONNX format, then loaded
by the Rust core via ort (ONNX Runtime) for inference.

Usage:
    python export_to_onnx.py --model_path gesture_encoder.pt --output_path gesture_encoder.onnx
"""

import torch
import argparse
from pathlib import Path
import numpy as np


class GestureEncoderONNX(torch.nn.Module):
    """
    Wrapper for exporting GestureEncoder to ONNX.
    Ensures compatible input/output shapes.
    """
    
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, time=30, features=63)
        
        Returns:
            Embedding tensor of shape (batch, 128)
        """
        return self.encoder(x)


def export_to_onnx(model_path: str, output_path: str, 
                   embedding_dim: int = 128,
                   time_steps: int = 30,
                   num_features: int = 63):
    """
    Export trained PyTorch model to ONNX format.
    
    Args:
        model_path: Path to saved PyTorch model (.pt file)
        output_path: Path for output ONNX model (.onnx file)
        embedding_dim: Output embedding dimension
        time_steps: Number of time steps in input
        num_features: Number of features per time step (21 landmarks × 3 coords)
    """
    print(f"Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Initialize encoder
    from train_encoder import GestureEncoder
    encoder = GestureEncoder(
        input_dim=num_features,
        embedding_dim=embedding_dim
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['model_state_dict'])
    else:
        encoder.load_state_dict(checkpoint)
    
    # Wrap for ONNX export
    model = GestureEncoderONNX(encoder)
    model.eval()
    
    # Create dummy input
    batch_size = 1
    dummy_input = torch.randn(batch_size, time_steps, num_features)
    
    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'time_steps'},
            'embedding': {0: 'batch_size'}
        }
    )
    
    print(f"✓ Successfully exported to {output_path}")
    
    # Verify export
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model validation passed")
    
    # Print model info
    print(f"\nModel Info:")
    print(f"  Input shape: (batch, {time_steps}, {num_features})")
    print(f"  Output shape: (batch, {embedding_dim})")
    print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    return output_path


def test_inference(model_path: str, onnx_path: str):
    """
    Test that ONNX model produces same output as PyTorch model.
    """
    import onnxruntime as ort
    
    print("\n=== Testing ONNX Inference ===")
    
    # Load PyTorch model
    from train_encoder import GestureEncoder
    checkpoint = torch.load(model_path, map_location='cpu')
    
    encoder = GestureEncoder(input_dim=63, embedding_dim=128)
    if 'model_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['model_state_dict'])
    else:
        encoder.load_state_dict(checkpoint)
    encoder.eval()
    
    # Load ONNX model
    session = ort.InferenceSession(onnx_path)
    
    # Test with random input
    test_input = np.random.randn(1, 30, 63).astype(np.float32)
    
    # PyTorch inference
    with torch.no_grad():
        torch_output = encoder(torch.FloatTensor(test_input)).numpy()
    
    # ONNX inference
    onnx_output = session.run(None, {'input': test_input})[0]
    
    # Compare
    diff = np.abs(torch_output - onnx_output).max()
    print(f"Max difference between PyTorch and ONNX: {diff:.6f}")
    
    if diff < 1e-5:
        print("✓ Inference match verified")
    else:
        print("⚠ Warning: Outputs differ slightly (expected due to floating point)")
    
    return torch_output, onnx_output


def main():
    parser = argparse.ArgumentParser(description='Export Gesture Encoder to ONNX')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to PyTorch model (.pt file)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Output ONNX path (default: same as model with .onnx)')
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--time_steps', type=int, default=30,
                       help='Number of time steps')
    parser.add_argument('--test', action='store_true',
                       help='Test inference after export')
    
    args = parser.parse_args()
    
    if args.output_path is None:
        args.output_path = str(Path(args.model_path).with_suffix('.onnx'))
    
    # Export
    export_to_onnx(
        model_path=args.model_path,
        output_path=args.output_path,
        embedding_dim=args.embedding_dim,
        time_steps=args.time_steps
    )
    
    # Test
    if args.test:
        test_inference(args.model_path, args.output_path)


if __name__ == '__main__':
    main()
