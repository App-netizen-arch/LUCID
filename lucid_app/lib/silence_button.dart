// Silence Button Widget for Gestalt-MVP
// Hard mute button that bypasses all recognition and stops TTS

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'bridge.dart';

/// Hard mute button - triggers SILENT state and stops TTS immediately
class SilenceButton extends StatefulWidget {
  const SilenceButton({Key? key}) : super(key: key);

  @override
  State<SilenceButton> createState() => _SilenceButtonState();
}

class _SilenceButtonState extends State<SilenceButton> {
  bool _isSilenced = false;
  bool _isLongPressing = false;

  @override
  Widget build(BuildContext context) {
    return Consumer<GestaltProvider>(
      builder: (context, gestaltProvider, child) {
        final fsmState = gestaltProvider.fsmState;
        _isSilenced = fsmState == FsmState.silent;

        return GestureDetector(
          onLongPress: () {
            setState(() => _isLongPressing = true);
            // Long press triggers hard mute
            gestaltProvider.triggerSilence();
            
            // Show confirmation
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text('Hard Mute Activated'),
                backgroundColor: Colors.red,
                duration: Duration(seconds: 2),
              ),
            );
          },
          onLongPressUp: () {
            setState(() => _isLongPressing = false);
          },
          onTap: _isSilenced ? () {
            // Tap to release from silent state
            gestaltProvider.releaseSilence();
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text('Mute Released'),
                backgroundColor: Colors.green,
                duration: Duration(seconds: 1),
              ),
            );
          } : null,
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 300),
            padding: const EdgeInsets.symmetric(
              horizontal: 24.0,
              vertical: 16.0,
            ),
            decoration: BoxDecoration(
              color: _isSilenced 
                  ? Colors.red 
                  : (_isLongPressing ? Colors.red.shade300 : Colors.grey.shade300),
              borderRadius: BorderRadius.circular(50.0),
              border: Border.all(
                color: _isSilenced ? Colors.red.shade900 : Colors.grey,
                width: 2.0,
              ),
              boxShadow: [
                if (_isSilenced || _isLongPressing)
                  BoxShadow(
                    color: Colors.red.withOpacity(0.5),
                    blurRadius: 12.0,
                    spreadRadius: 4.0,
                  ),
              ],
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  _isSilenced ? Icons.volume_off : Icons.volume_mute,
                  color: _isSilenced ? Colors.white : Colors.black54,
                  size: 24.0,
                ),
                const SizedBox(width: 8.0),
                Text(
                  _isSilenced ? 'MUTED' : 'Hold to Mute',
                  style: TextStyle(
                    fontSize: 14.0,
                    fontWeight: FontWeight.bold,
                    color: _isSilenced ? Colors.white : Colors.black54,
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}
