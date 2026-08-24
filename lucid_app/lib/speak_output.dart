// Speak Output Widget for Gestalt-MVP
// Displays recognized phrase and triggers TTS output

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'bridge.dart';

/// Widget that displays the recognized gesture phrase and provides TTS output
class SpeakOutput extends StatelessWidget {
  const SpeakOutput({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Consumer<GestaltProvider>(
      builder: (context, gestaltProvider, child) {
        final currentPhrase = gestaltProvider.currentPhrase;
        final fsmState = gestaltProvider.fsmState;
        final isSpeaking = gestaltProvider.isSpeaking;

        // Determine color based on FSM state
        Color stateColor;
        switch (fsmState) {
          case FsmState.highConf:
            stateColor = Colors.green;
            break;
          case FsmState.lowConf:
            stateColor = Colors.orange;
            break;
          case FsmState.silent:
            stateColor = Colors.red;
            break;
          case FsmState.error:
            stateColor = Colors.red.shade900;
            break;
          default:
            stateColor = Colors.grey;
        }

        return Container(
          padding: const EdgeInsets.all(16.0),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(12.0),
            border: Border.all(color: stateColor, width: 2.0),
            boxShadow: [
              BoxShadow(
                color: stateColor.withOpacity(0.3),
                blurRadius: 8.0,
                spreadRadius: 2.0,
              ),
            ],
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header with state indicator
              Row(
                children: [
                  Container(
                    width: 12.0,
                    height: 12.0,
                    decoration: BoxDecoration(
                      color: stateColor,
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 8.0),
                  Text(
                    'Output',
                    style: TextStyle(
                      fontSize: 16.0,
                      fontWeight: FontWeight.bold,
                      color: stateColor,
                    ),
                  ),
                  const Spacer(),
                  if (isSpeaking)
                    const Icon(
                      Icons.volume_up,
                      color: Colors.green,
                      size: 20.0,
                    ),
                ],
              ),
              const SizedBox(height: 12.0),
              
              // Phrase text
              if (currentPhrase != null && currentPhrase.isNotEmpty)
                Text(
                  currentPhrase,
                  style: const TextStyle(
                    fontSize: 20.0,
                    fontWeight: FontWeight.w500,
                    height: 1.4,
                  ),
                  textAlign: TextAlign.left,
                )
              else
                const Text(
                  'No gesture recognized',
                  style: TextStyle(
                    fontSize: 18.0,
                    fontStyle: FontStyle.italic,
                    color: Colors.grey,
                  ),
                ),
              
              const SizedBox(height: 16.0),
              
              // Action buttons
              Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  // Retry button
                  if (currentPhrase != null && currentPhrase.isNotEmpty)
                    IconButton(
                      icon: const Icon(Icons.repeat),
                      tooltip: 'Repeat',
                      onPressed: () {
                        gestaltProvider.speak(currentPhrase);
                      },
                      color: Colors.blue,
                    ),
                  
                  // Clear button
                  IconButton(
                    icon: const Icon(Icons.clear),
                    tooltip: 'Clear',
                    onPressed: () {
                      gestaltProvider.clearOutput();
                    },
                    color: Colors.grey,
                  ),
                ],
              ),
            ],
          ),
        );
      },
    );
  }
}
