import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'camera_screen.dart';
import 'bridge.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Initialize Rust FFI bridge
  try {
    await GestaltApi.initialize();
    debugPrint('Rust engine initialized successfully');
  } catch (e) {
    debugPrint('Failed to initialize Rust engine: \$e');
  }
  
  runApp(const GestaltApp());
}

class GestaltApp extends StatelessWidget {
  const GestaltApp({Key? key}) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => GestaltState(),
      child: MaterialApp(
        title: 'Gestalt MVP',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          brightness: Brightness.dark,
          primarySwatch: Colors.blue,
          scaffoldBackgroundColor: Colors.black,
        ),
        home: const CameraScreen(),
      ),
    );
  }
}
