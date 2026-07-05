import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';

import '../services/backend_service.dart';
import '../services/history_service.dart';
import '../theme/app_theme.dart';
import '../widgets/detection_card.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with WidgetsBindingObserver {
  CameraController? _cameraController;

  final BackendService _backendService = BackendService();
  final FlutterTts _flutterTts = FlutterTts();

  bool _isCameraReady = false;
  bool _isSendingFrame = false;
  bool _isCollecting = true;

  String _liveWord = 'WAITING';
  double _confidence = 0.0;

  String _sentence = '';
  String _lastAddedWord = '';
  String _candidateWord = '';
  int _candidateCount = 0;

  static const int _stableCountNeeded = 3;

  DateTime _lastFrameSentTime = DateTime.now();

  int _framesCollected = 0;
  int _totalFrames = 30;
  int _handCount = 0;

  String _statusMessage = 'Starting camera...';

  // This fixed your upside-down issue.
  // Try 90 / 180 / 270 only if orientation becomes wrong again.
  static const int _frameRotateDegrees = 270;

  // Change to true only if model acts like left/right is mirrored wrongly.
  static const bool _mirrorFrame = false;

  // Send one frame every 180 ms.
  // Lower = faster but heavier on phone/backend.
  static const int _frameIntervalMs = 180;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _setupTts();
    _initCamera();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _flutterTts.stop();
    _cameraController?.dispose();
    super.dispose();
  }

  Future<void> _setupTts() async {
    await _flutterTts.setLanguage('en-IN');
    await _flutterTts.setSpeechRate(0.45);
    await _flutterTts.setVolume(1.0);
    await _flutterTts.setPitch(1.0);
  }

  Future<void> _speakText(String text) async {
    final cleanText = text.trim();

    if (cleanText.isEmpty) return;

    await _flutterTts.stop();
    await _flutterTts.speak(cleanText);
  }

  Future<void> _speakSentence() async {
    if (_sentence.trim().isNotEmpty) {
      await _speakText(_sentence);
      return;
    }

    if (_liveWord != 'WAITING' &&
        _liveWord != 'UNKNOWN' &&
        _liveWord != 'IDLE' &&
        _liveWord != 'ERROR') {
      await _speakText(_liveWord);
    }
  }

  Future<void> _initCamera() async {
    final permission = await Permission.camera.request();

    if (!permission.isGranted) {
      if (!mounted) return;

      setState(() {
        _statusMessage = 'Camera permission denied';
        _liveWord = 'NO CAMERA';
      });
      return;
    }

    try {
      final cameras = await availableCameras();

      final frontCamera = cameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );

      final controller = CameraController(
        frontCamera,
        ResolutionPreset.low,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      await controller.initialize();

      _cameraController = controller;

      await controller.startImageStream(_processCameraImage);

      if (!mounted) return;

      setState(() {
        _isCameraReady = true;
        _statusMessage = 'Camera ready. Connecting to backend...';
      });

      try {
        await _backendService.reset();

        if (!mounted) return;

        setState(() {
          _statusMessage = 'Backend connected';
        });
      } catch (e) {
        if (!mounted) return;

        setState(() {
          _statusMessage = 'Camera ready, but backend not reachable';
        });
      }
    } catch (e) {
      if (!mounted) return;

      setState(() {
        _statusMessage = 'Camera error: $e';
        _liveWord = 'ERROR';
      });
    }
  }

  Future<void> _processCameraImage(CameraImage image) async {
    if (_isSendingFrame) return;

    final now = DateTime.now();
    final diff = now.difference(_lastFrameSentTime).inMilliseconds;

    if (diff < _frameIntervalMs) return;

    _lastFrameSentTime = now;
    _isSendingFrame = true;

    try {
      final base64Image = _cameraImageToJpegBase64(image);
      final prediction = await _backendService.predict(base64Image);

      if (!mounted) return;

      setState(() {
        _liveWord = prediction.label;
        _confidence = prediction.confidence;
        _framesCollected = prediction.framesCollected;
        _totalFrames = prediction.totalFrames;
        _handCount = prediction.handCount;
        _isCollecting = prediction.status == 'collecting';

        _statusMessage = prediction.message.isEmpty
            ? 'Hands detected: $_handCount'
            : '${prediction.message} | Hands: $_handCount';
      });

      if (!_isCollecting) {
        _handleSentencePrediction(prediction.label, prediction.confidence);
      }
    } catch (e) {
      if (!mounted) return;

      setState(() {
        _statusMessage = 'Backend not reachable';
      });
    } finally {
      _isSendingFrame = false;
    }
  }

  String _cameraImageToJpegBase64(CameraImage image) {
    img.Image convertedImage = _convertYUV420ToImage(image);

    if (_frameRotateDegrees != 0) {
      convertedImage = img.copyRotate(
        convertedImage,
        angle: _frameRotateDegrees,
      );
    }

    if (_mirrorFrame) {
      convertedImage = img.flipHorizontal(convertedImage);
    }

    final jpgBytes = img.encodeJpg(convertedImage, quality: 65);
    return base64Encode(Uint8List.fromList(jpgBytes));
  }

  img.Image _convertYUV420ToImage(CameraImage image) {
    final int width = image.width;
    final int height = image.height;

    final img.Image imgBuffer = img.Image(width: width, height: height);

    final Plane planeY = image.planes[0];
    final Plane planeU = image.planes[1];
    final Plane planeV = image.planes[2];

    final int uvRowStride = planeU.bytesPerRow;
    final int uvPixelStride = planeU.bytesPerPixel ?? 1;

    for (int y = 0; y < height; y++) {
      final int uvRow = uvRowStride * (y >> 1);
      final int yRow = planeY.bytesPerRow * y;

      for (int x = 0; x < width; x++) {
        final int uvIndex = uvRow + (x >> 1) * uvPixelStride;
        final int yIndex = yRow + x;

        final int yp = planeY.bytes[yIndex];
        final int up = planeU.bytes[uvIndex];
        final int vp = planeV.bytes[uvIndex];

        int r = (yp + 1.402 * (vp - 128)).round();
        int g = (yp - 0.344136 * (up - 128) - 0.714136 * (vp - 128)).round();
        int b = (yp + 1.772 * (up - 128)).round();

        r = r.clamp(0, 255);
        g = g.clamp(0, 255);
        b = b.clamp(0, 255);

        imgBuffer.setPixelRgb(x, y, r, g, b);
      }
    }

    return imgBuffer;
  }

  void _handleSentencePrediction(String word, double confidence) {
    final String label = word.toUpperCase().trim();

    if (label == 'IDLE' ||
        label == 'WAITING' ||
        label == 'UNKNOWN' ||
        label == 'ERROR' ||
        confidence < 0.70) {
      _candidateWord = '';
      _candidateCount = 0;
      _lastAddedWord = '';
      return;
    }

    if (_candidateWord == label) {
      _candidateCount++;
    } else {
      _candidateWord = label;
      _candidateCount = 1;
    }

    if (_candidateCount < _stableCountNeeded) {
      return;
    }

    // Prevent same word from being added again while user is still showing it.
    if (_lastAddedWord == label) {
      return;
    }

    setState(() {
      if (_sentence.trim().isEmpty) {
        _sentence = label;
      } else {
        _sentence = '$_sentence $label';
      }

      _lastAddedWord = label;
    });
  }

  Future<void> _reset() async {
    try {
      await _backendService.reset();
    } catch (_) {
      // Even if backend reset fails, reset frontend UI.
    }

    if (!mounted) return;

    setState(() {
      _liveWord = 'WAITING';
      _confidence = 0.0;
      _sentence = '';
      _lastAddedWord = '';
      _candidateWord = '';
      _candidateCount = 0;
      _framesCollected = 0;
      _isCollecting = true;
      _statusMessage = 'Reset done';
    });
  }

  Future<void> _saveSentence() async {
    if (_sentence.trim().isEmpty) return;

    await HistoryService.addSentence(_sentence);

    if (!mounted) return;

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Sentence saved to history'),
        backgroundColor: AppTheme.surface,
      ),
    );
  }

  void _deleteLastWord() {
    if (_sentence.trim().isEmpty) return;

    final words = _sentence.trim().split(' ');

    if (words.isEmpty) return;

    words.removeLast();

    setState(() {
      _sentence = words.join(' ');
      _lastAddedWord = '';
      _candidateWord = '';
      _candidateCount = 0;
    });
  }

  @override
  Widget build(BuildContext context) {
    final controller = _cameraController;

    return Scaffold(
      backgroundColor: AppTheme.background,
      body: SafeArea(
        child: Stack(
          children: [
            if (!_isCameraReady || controller == null)
              Center(
                child: Padding(
                  padding: const EdgeInsets.all(24),
                  child: Text(
                    _statusMessage,
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      color: AppTheme.textSecondary,
                      fontSize: 16,
                    ),
                  ),
                ),
              )
            else
              Positioned.fill(
                child: CameraPreview(controller),
              ),

            Positioned(
              top: 12,
              left: 12,
              right: 12,
              child: Row(
                children: [
                  CircleAvatar(
                    backgroundColor: AppTheme.surface.withOpacity(0.85),
                    child: IconButton(
                      icon: const Icon(
                        Icons.arrow_back,
                        color: AppTheme.accent,
                      ),
                      onPressed: () => Navigator.pop(context),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 14,
                        vertical: 10,
                      ),
                      decoration: BoxDecoration(
                        color: AppTheme.surface.withOpacity(0.85),
                        borderRadius: BorderRadius.circular(24),
                      ),
                      child: const Text(
                        'Real Model Backend',
                        style: TextStyle(
                          color: AppTheme.textPrimary,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),

            Positioned(
              left: 16,
              right: 16,
              bottom: 225,
              child: Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: AppTheme.surface.withOpacity(0.92),
                  borderRadius: BorderRadius.circular(18),
                  border: Border.all(color: AppTheme.surfaceLight),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'SENTENCE',
                      style: TextStyle(
                        color: AppTheme.textSecondary,
                        fontSize: 11,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 1.4,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      _sentence.isEmpty
                          ? 'Detected words will appear here...'
                          : _sentence,
                      style: TextStyle(
                        color: _sentence.isEmpty
                            ? AppTheme.textSecondary
                            : AppTheme.textPrimary,
                        fontSize: 18,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(height: 10),
                    Text(
                      _statusMessage,
                      style: const TextStyle(
                        color: AppTheme.textSecondary,
                        fontSize: 12,
                      ),
                    ),
                    const SizedBox(height: 12),

                    Row(
                      children: [
                        Expanded(
                          child: OutlinedButton.icon(
                            onPressed: _deleteLastWord,
                            icon: const Icon(
                              Icons.backspace_outlined,
                              size: 18,
                            ),
                            label: const Text('Back'),
                            style: OutlinedButton.styleFrom(
                              foregroundColor: AppTheme.textPrimary,
                              side: const BorderSide(
                                color: AppTheme.surfaceLight,
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: OutlinedButton.icon(
                            onPressed: _speakSentence,
                            icon: const Icon(
                              Icons.volume_up_outlined,
                              size: 18,
                            ),
                            label: const Text('Speak'),
                            style: OutlinedButton.styleFrom(
                              foregroundColor: AppTheme.accent,
                              side: const BorderSide(
                                color: AppTheme.accent,
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: ElevatedButton.icon(
                            onPressed: _saveSentence,
                            icon: const Icon(
                              Icons.save_outlined,
                              size: 18,
                            ),
                            label: const Text('Save'),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),

            Positioned(
              left: 0,
              right: 0,
              bottom: 0,
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  DetectionCard(
                    label: _liveWord,
                    confidence: _confidence,
                    isCollecting: _isCollecting,
                    framesCollected: _framesCollected,
                    totalFrames: _totalFrames,
                  ),
                  Container(
                    color: AppTheme.surface,
                    padding: const EdgeInsets.fromLTRB(20, 0, 20, 16),
                    child: Row(
                      children: [
                        Expanded(
                          child: OutlinedButton.icon(
                            onPressed: _reset,
                            icon: const Icon(Icons.refresh),
                            label: const Text('Reset'),
                            style: OutlinedButton.styleFrom(
                              foregroundColor: AppTheme.textPrimary,
                              side: const BorderSide(
                                color: AppTheme.surfaceLight,
                              ),
                              padding: const EdgeInsets.symmetric(vertical: 13),
                            ),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: ElevatedButton.icon(
                            onPressed: _speakSentence,
                            icon: const Icon(Icons.volume_up),
                            label: const Text('Speak'),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}