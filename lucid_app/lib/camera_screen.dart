import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:provider/provider.dart';
import 'gesture_overlay.dart';
import 'status_bar.dart';
import 'bridge.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({Key? key}) : super(key: key);

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  Future<void>? _initializeControllerFuture;
  bool _isProcessing = false;
  List<List<double>>? _currentLandmarks;
  
  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }
  
  Future<void> _initializeCamera() async {
    final cameras = await availableCameras();
    // Use front-facing camera (mounted, facing user)
    final frontCamera = cameras.firstWhere(
      (camera) => camera.lensDirection == CameraLensDirection.front,
      orElse: () => cameras.first,
    );
    
    _controller = CameraController(
      frontCamera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    
    _initializeControllerFuture = _controller!.initialize().then((_) {
      if (!mounted) return;
      setState(() {});
      _startFrameProcessing();
    });
    
    if (_controller!.value.hasError) {
      throw CameraException(_controller!.value.errorDescription!, 'Camera initialization failed');
    }
  }
  
  void _startFrameProcessing() {
    if (_controller == null || !_controller!.value.isInitialized) return;
    
    _controller!.startImageStream((CameraImage image) async {
      if (_isProcessing) return;
      _isProcessing = true;
      
      try {
        // Process frame through MediaPipe (simulated here, actual C++ bridge in production)
        final landmarks = await _processFrameWithMediaPipe(image);
        
        if (landmarks != null && mounted) {
          setState(() {
            _currentLandmarks = landmarks;
          });
          
          // Send to Rust symbolic core
          final result = await GestaltApi.processFrame(landmarks);
          
          if (result != null && mounted) {
            Provider.of<GestaltState>(context, listen: false).updateResult(result);
          }
        }
      } catch (e) {
        debugPrint('Frame processing error: \$e');
      } finally {
        _isProcessing = false;
      }
    });
  }
  
  Future<List<List<double>>?> _processFrameWithMediaPipe(CameraImage image) async {
    // In production: call C++ MediaPipe via FFI
    // For now: return placeholder (actual implementation in hand_tracking.cpp)
    // This would call the mediapipe_cpp/hand_tracking.h functions
    
    // Placeholder: return 21 landmarks with dummy data
    // Actual implementation extracts from YUV image and runs MediaPipe Hands
    return List.generate(21, (i) => [0.5, 0.5, 0.5]);
  }
  
  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: FutureBuilder<void>(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.done) {
            return Stack(
              fit: StackFit.expand,
              children: [
                // Camera preview
                Transform.scale(
                  scale: -1, // Mirror for front camera
                  child: CameraPreview(_controller!),
                ),
                
                // Gesture overlay showing landmarks
                if (_currentLandmarks != null)
                  GestureOverlay(landmarks: _currentLandmarks!),
                
                // Status bar showing FSM state
                Positioned(
                  top: 0,
                  left: 0,
                  right: 0,
                  child: StatusBar(),
                ),
                
                // Output panel at bottom
                Positioned(
                  bottom: 0,
                  left: 0,
                  right: 0,
                  child: _buildOutputPanel(),
                ),
              ],
            );
          } else {
            return const Center(child: CircularProgressIndicator());
          }
        },
      ),
    );
  }
  
  Widget _buildOutputPanel() {
    return Consumer<GestaltState>(
      builder: (context, state, child) {
        return Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.black.withOpacity(0.8),
            borderRadius: const BorderRadius.vertical(top: Radius.circular(16)),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Current phrase
              Text(
                state.currentPhrase,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,
              ),
              
              const SizedBox(height: 12),
              
              // Confidence indicator
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Text(
                    'Confidence: ',
                    style: TextStyle(color: Colors.white70),
                  ),
                  Container(
                    width: 100,
                    height: 8,
                    decoration: BoxDecoration(
                      color: Colors.grey[800],
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: FractionallySizedBox(
                      alignment: Alignment.centerLeft,
                      widthFactor: state.confidence.clamp(0.0, 1.0),
                      child: Container(
                        decoration: BoxDecoration(
                          color: _getConfidenceColor(state.confidence),
                          borderRadius: BorderRadius.circular(4),
                        ),
                      ),
                    ),
                  ),
                  Text(
                    '\${(state.confidence * 100).toInt()}%',
                    style: TextStyle(
                      color: _getConfidenceColor(state.confidence),
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
              
              const SizedBox(height: 12),
              
              // Correction and silence buttons
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  ElevatedButton.icon(
                    onPressed: () => _handleCorrection(),
                    icon: const Icon(Icons.edit, color: Colors.black),
                    label: const Text('That was wrong', style: TextStyle(color: Colors.black)),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.amber,
                    ),
                  ),
                  
                  ElevatedButton.icon(
                    onPressed: () => _toggleSilence(state.isSilent),
                    icon: Icon(state.isSilent ? Icons.volume_off : Icons.volume_up, color: Colors.white),
                    label: Text(state.isSilent ? 'Unmute' : 'Mute', style: const TextStyle(color: Colors.white)),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: state.isSilent ? Colors.red : Colors.blue,
                    ),
                  ),
                ],
              ),
            ],
          ),
        );
      },
    );
  }
  
  Color _getConfidenceColor(double confidence) {
    if (confidence > 0.8) return Colors.green;
    if (confidence > 0.5) return Colors.orange;
    return Colors.red;
  }
  
  void _handleCorrection() {
    // Show dialog to select intended gesture
    showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Select Correct Gesture'),
        content: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              for (var i = 1; i <= 10; i++)
                ListTile(
                  title: Text('G\$i - \${_getGestureName(i)}'),
                  onTap: () {
                    Navigator.pop(context, 'G\$i');
                  },
                ),
            ],
          ),
        ),
      ),
    ).then((intendedGestureId) {
      if (intendedGestureId != null) {
        final currentState = Provider.of<GestaltState>(context, listen: false);
        if (currentState.lastGestureId != null) {
          GestaltApi.submitCorrection(currentState.lastGestureId!, intendedGestureId);
        }
      }
    });
  }
  
  void _toggleSilence(bool currentSilent) {
    GestaltApi.setSilentMode(!currentSilent);
    Provider.of<GestaltState>(context, listen: false).toggleSilent();
  }
  
  String _getGestureName(int id) {
    const names = [
      'Water', 'Help', 'Yes', 'No', 'Food',
      'Pain', 'Stop', 'More', 'Wait', 'Bathroom'
    ];
    return names[id - 1];
  }
}
