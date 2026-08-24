// FFI Bridge - Auto-generated bindings wrapper
// This file provides Dart bindings to the Rust symbolic core via flutter_rust_bridge

import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';

part 'bridge.g.dart';

// Initialize the Rust library
Future<void> initRustEngine() async {
  await api.initEngine();
}

// Gesture recognition result from Rust
class GestureResult {
  final String? gestureId;
  final double confidence;
  final String phrase;
  final String fsmState;
  
  GestureResult({
    this.gestureId,
    required this.confidence,
    required this.phrase,
    required this.fsmState,
  });
}

// Hand landmark data structure
class LandmarkFrame {
  final double t;
  final List<List<double>> landmarks; // 21 x [x, y, z]
  
  LandmarkFrame({required this.t, required this.landmarks});
}

// Message types from Rust message bus
enum MessageType {
  perceptualCandidate,
  matchResult,
  gateDecision,
  correctionMessage,
  controlMessage,
}

class BusMessage {
  final MessageType type;
  final dynamic payload;
  final DateTime timestamp;
  
  BusMessage({required this.type, required this.payload, required this.timestamp});
}

// FSM State enumeration
enum FsmState {
  idle,
  listening,
  gestureDetected,
  highConf,
  lowConf,
  correction,
  silent,
  error,
}

// Correction request to modify mapping
class CorrectionRequest {
  final String detectedGestureId;
  final String intendedGestureId;
  
  CorrectionRequest({
    required this.detectedGestureId,
    required this.intendedGestureId,
  });
}

// Main API class for Rust FFI
class GestaltApi {
  static late RustLib _api;
  
  static Future<void> initialize() async {
    _api = await RustLib.init();
  }
  
  static Future<void> initEngine() async {
    return _api.initEngine();
  }
  
  static Future<GestureResult?> processFrame(List<List<double>> landmarks) async {
    return _api.processFrame(landmarks);
  }
  
  static Future<void> submitCorrection(String detected, String intended) async {
    return _api.submitCorrection(detected, intended);
  }
  
  static Future<void> setSilentMode(bool silent) async {
    return _api.setSilentMode(silent);
  }
  
  static Future<String> getCurrentPhrase() async {
    return _api.getCurrentPhrase();
  }
  
  static Future<FsmState> getCurrentState() async {
    return _api.getCurrentState();
  }
  
  static Future<void> loadExemplars(String sessionPath) async {
    return _api.loadExemplars(sessionPath);
  }
  
  static Future<void> calibrateThresholds(String calibrationPath) async {
    return _api.calibrateThresholds(calibrationPath);
  }
  
  static Stream<BusMessage> getMessageStream() {
    return _api.messageStream();
  }
  
  // TTS controls (Phase 2)
  static Future<void> speak(String text) async {
    return _api.speak(text);
  }
  
  static Future<void> stopSpeech() async {
    return _api.stopSpeech();
  }
  
  static Future<void> setTtsEngine(String engine) async {
    return _api.setTtsEngine(engine);
  }
  
  static Future<void> setTtsProsody(double? speed, double? pitch, double? volume) async {
    return _api.setTtsProsody(speed, pitch, volume);
  }
}

/// Provider for managing Gestalt state across the Flutter app
class GestaltProvider extends ChangeNotifier {
  FsmState _fsmState = FsmState.idle;
  String? _currentPhrase;
  String? _lastGestureId;
  bool _isSpeaking = false;
  double _confidence = 0.0;
  
  FsmState get fsmState => _fsmState;
  String? get currentPhrase => _currentPhrase;
  String? get lastGestureId => _lastGestureId;
  bool get isSpeaking => _isSpeaking;
  double get confidence => _confidence;
  
  /// Update state from Rust engine
  void updateState(FsmState newState, String? phrase, String? gestureId, double conf) {
    _fsmState = newState;
    _currentPhrase = phrase;
    _lastGestureId = gestureId;
    _confidence = conf;
    notifyListeners();
  }
  
  /// Trigger speech output
  Future<void> speak(String text) async {
    _isSpeaking = true;
    notifyListeners();
    
    try {
      await GestaltApi.speak(text);
    } finally {
      _isSpeaking = false;
      notifyListeners();
    }
  }
  
  /// Stop speech immediately
  Future<void> stopSpeech() async {
    await GestaltApi.stopSpeech();
    _isSpeaking = false;
    notifyListeners();
  }
  
  /// Clear current output
  void clearOutput() {
    _currentPhrase = null;
    _lastGestureId = null;
    notifyListeners();
  }
  
  /// Apply user correction
  Future<void> applyCorrection(String intendedGesture) async {
    if (_lastGestureId != null) {
      await GestaltApi.submitCorrection(_lastGestureId!, intendedGesture);
      notifyListeners();
    }
  }
  
  /// Trigger silent mode (hard mute)
  Future<void> triggerSilence() async {
    await GestaltApi.setSilentMode(true);
    _fsmState = FsmState.silent;
    await stopSpeech();
    notifyListeners();
  }
  
  /// Release from silent mode
  Future<void> releaseSilence() async {
    await GestaltApi.setSilentMode(false);
    _fsmState = FsmState.idle;
    notifyListeners();
  }
  
  /// Set TTS engine
  Future<void> setTtsEngine(String engine) async {
    await GestaltApi.setTtsEngine(engine);
    notifyListeners();
  }
  
  /// Set TTS prosody
  Future<void> setTtsProsody({double? speed, double? pitch, double? volume}) async {
    await GestaltApi.setTtsProsody(speed, pitch, volume);
    notifyListeners();
  }
}
