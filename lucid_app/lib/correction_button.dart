// Correction Button Widget for Gestalt-MVP
// Allows user/caregiver to signal "That was wrong"

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'bridge.dart';

/// Button that triggers the correction state in the FSM
class CorrectionButton extends StatelessWidget {
  const CorrectionButton({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Consumer<GestaltProvider>(
      builder: (context, gestaltProvider, child) {
        final fsmState = gestaltProvider.fsmState;
        final lastGestureId = gestaltProvider.lastGestureId;
        
        // Only show correction button if a gesture was recently recognized
        final canCorrect = lastGestureId != null && 
                          fsmState != FsmState.idle &&
                          fsmState != FsmState.error;

        return AnimatedOpacity(
          opacity: canCorrect ? 1.0 : 0.3,
          duration: const Duration(milliseconds: 300),
          child: IgnorePointer(
            ignoring: !canCorrect,
            child: ElevatedButton.icon(
              onPressed: canCorrect ? () => _showCorrectionDialog(context, gestaltProvider) : null,
              icon: const Icon(Icons.edit),
              label: const Text('That Was Wrong'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.orange,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(
                  horizontal: 20.0,
                  vertical: 12.0,
                ),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8.0),
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  void _showCorrectionDialog(BuildContext context, GestaltProvider gestaltProvider) {
    showDialog<String>(
      context: context,
      builder: (BuildContext context) {
        String? selectedGesture;
        
        return AlertDialog(
          title: const Text('What did you mean?'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text('Select the correct gesture:'),
              const SizedBox(height: 16.0),
              ...List.generate(10, (index) {
                final gestureId = 'G${index + 1}';
                final gestureName = _getGestureName(gestureId);
                return RadioListTile<String>(
                  title: Text('$gestureName ($gestureId)'),
                  value: gestureId,
                  groupValue: selectedGesture,
                  onChanged: (value) {
                    Navigator.of(context).pop(value);
                  },
                );
              }),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Cancel'),
            ),
          ],
        );
      },
    ).then((selectedGesture) {
      if (selectedGesture != null) {
        gestaltProvider.applyCorrection(selectedGesture);
        
        // Show confirmation snackbar
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Correction applied: $selectedGesture'),
            backgroundColor: Colors.green,
            duration: const Duration(seconds: 2),
          ),
        );
      }
    });
  }

  String _getGestureName(String gestureId) {
    switch (gestureId) {
      case 'G1': return 'Water';
      case 'G2': return 'Help';
      case 'G3': return 'Yes';
      case 'G4': return 'No';
      case 'G5': return 'Food';
      case 'G6': return 'Pain';
      case 'G7': return 'Stop';
      case 'G8': return 'More';
      case 'G9': return 'Wait';
      case 'G10': return 'Bathroom';
      default: return 'Unknown';
    }
  }
}
