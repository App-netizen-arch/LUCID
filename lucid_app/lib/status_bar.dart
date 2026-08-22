import 'package:flutter/material.dart';

class StatusBar extends StatelessWidget {
  const StatusBar({Key? key}) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
    return Consumer<GestaltState>(
      builder: (context, state, child) {
        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          decoration: BoxDecoration(
            color: Colors.black.withOpacity(0.7),
            borderRadius: const BorderRadius.vertical(bottom: Radius.circular(12)),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              // FSM State indicator
              _buildStateIndicator(state.fsmState),
              
              // Silent mode indicator
              if (state.isSilent)
                _buildBadge('SILENT', Colors.red),
              
              // Connection status
              _buildBadge('ONLINE', Colors.green),
            ],
          ),
        );
      },
    );
  }
  
  Widget _buildStateIndicator(String fsmState) {
    Color stateColor;
    IconData stateIcon;
    
    switch (fsmState.toUpperCase()) {
      case 'IDLE':
        stateColor = Colors.grey;
        stateIcon = Icons.pause_circle_outline;
        break;
      case 'LISTENING':
        stateColor = Colors.blue;
        stateIcon = Icons.hearing;
        break;
      case 'GESTURE_DETECTED':
        stateColor = Colors.orange;
        stateIcon = Icons.gesture;
        break;
      case 'HIGH_CONF':
        stateColor = Colors.green;
        stateIcon = Icons.check_circle;
        break;
      case 'LOW_CONF':
        stateColor = Colors.amber;
        stateIcon = Icons.help_outline;
        break;
      case 'CORRECTION':
        stateColor = Colors.purple;
        stateIcon = Icons.edit;
        break;
      case 'SILENT':
        stateColor = Colors.red;
        stateIcon = Icons.volume_off;
        break;
      case 'ERROR':
        stateColor = Colors.red;
        stateIcon = Icons.error_outline;
        break;
      default:
        stateColor = Colors.grey;
        stateIcon = Icons.help_outline;
    }
    
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(stateIcon, color: stateColor, size: 20),
        const SizedBox(width: 8),
        Text(
          fsmState,
          style: TextStyle(
            color: stateColor,
            fontWeight: FontWeight.bold,
            fontSize: 14,
          ),
        ),
      ],
    );
  }
  
  Widget _buildBadge(String text, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.3),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color, width: 1),
      ),
      child: Text(
        text,
        style: TextStyle(
          color: color,
          fontSize: 12,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }
}

// Application state provider (shared with camera_screen)
class GestaltState extends ChangeNotifier {
  String _currentPhrase = '';
  double _confidence = 0.0;
  String _fsmState = 'IDLE';
  String? _lastGestureId;
  bool _isSilent = false;
  
  String get currentPhrase => _currentPhrase;
  double get confidence => _confidence;
  String get fsmState => _fsmState;
  String? get lastGestureId => _lastGestureId;
  bool get isSilent => _isSilent;
  
  void updateResult(dynamic result) {
    if (result != null) {
      _currentPhrase = result.phrase ?? '';
      _confidence = result.confidence;
      _fsmState = result.fsmState;
      _lastGestureId = result.gestureId;
      notifyListeners();
    }
  }
  
  void toggleSilent() {
    _isSilent = !_isSilent;
    notifyListeners();
  }
}
