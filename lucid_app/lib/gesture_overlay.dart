import 'package:flutter/material.dart';

class GestureOverlay extends StatelessWidget {
  final List<List<double>> landmarks; // 21 x [x, y, z]
  
  const GestureOverlay({Key? key, required this.landmarks}) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: HandLandmarkPainter(landmarks: landmarks),
      size: Size.infinite,
    );
  }
}

class HandLandmarkPainter extends CustomPainter {
  final List<List<double>> landmarks;
  
  HandLandmarkPainter({required this.landmarks});
  
  // MediaPipe hand landmark connections
  static const List<List<int>> _connections = [
    [0, 1], [1, 2], [2, 3], [3, 4],     // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],     // Index finger
    [0, 9], [9, 10], [10, 11], [11, 12], // Middle finger
    [0, 13], [13, 14], [14, 15], [15, 16], // Ring finger
    [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
    [5, 9], [9, 13], [13, 17],           // Palm base connections
  ];
  
  @override
  void paint(Canvas canvas, Size size) {
    if (landmarks.isEmpty || landmarks[0].length < 2) return;
    
    // Convert normalized landmarks to screen coordinates
    final points = landmarks.map((landmark) {
      return Offset(
        landmark[0] * size.width,
        landmark[1] * size.height,
      );
    }).toList();
    
    // Draw connections
    final linePaint = Paint()
      ..color = Colors.green.withOpacity(0.7)
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round;
    
    for (final connection in _connections) {
      if (connection.length >= 2 && 
          connection[0] < points.length && 
          connection[1] < points.length) {
        canvas.drawLine(
          points[connection[0]],
          points[connection[1]],
          linePaint,
        );
      }
    }
    
    // Draw landmarks
    final pointPaint = Paint()
      ..color = Colors.red
      ..style = PaintingStyle.fill;
    
    for (int i = 0; i < points.length; i++) {
      // Fingertips (4, 8, 12, 16, 20) are larger
      final isFingertip = [4, 8, 12, 16, 20].contains(i);
      final radius = isFingertip ? 8.0 : 5.0;
      
      canvas.drawCircle(points[i], radius, pointPaint);
      
      // Label fingertips
      if (isFingertip) {
        final textPainter = TextPainter(
          text: TextSpan(
            text: '\$i',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 10,
              fontWeight: FontWeight.bold,
            ),
          ),
          textDirection: TextDirection.ltr,
        );
        textPainter.layout();
        textPainter.paint(
          canvas,
          Offset(points[i].dx - 5, points[i].dy - 15),
        );
      }
    }
    
    // Draw wrist prominently
    final wristPaint = Paint()
      ..color = Colors.blue
      ..style = PaintingStyle.fill;
    canvas.drawCircle(points[0], 10, wristPaint);
  }
  
  @override
  bool shouldRepaint(covariant HandLandmarkPainter oldDelegate) {
    return oldDelegate.landmarks != landmarks;
  }
}
