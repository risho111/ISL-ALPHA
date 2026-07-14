import 'dart:convert';
import 'package:http/http.dart' as http;

class BackendPrediction {
  final String status;
  final String label;
  final String rawLabel;
  final double confidence;
  final int framesCollected;
  final int totalFrames;
  final int handCount;
  final String message;

  BackendPrediction({
    required this.status,
    required this.label,
    required this.rawLabel,
    required this.confidence,
    required this.framesCollected,
    required this.totalFrames,
    required this.handCount,
    required this.message,
  });

  factory BackendPrediction.fromJson(Map<String, dynamic> json) {
    return BackendPrediction(
      status: json['status']?.toString() ?? 'unknown',
      label: json['label']?.toString() ?? 'UNKNOWN',
      rawLabel: json['raw_label']?.toString() ?? 'unknown',
      confidence: (json['confidence'] ?? 0.0).toDouble(),
      framesCollected: json['frames_collected'] ?? 0,
      totalFrames: json['total_frames'] ?? 30,
      handCount: json['hand_count'] ?? 0,
      message: json['message']?.toString() ?? '',
    );
  }
}

class BackendService {
  static const String baseUrl = 'http://10.139.23.219:8000';

  Future<BackendPrediction> predict(String base64Image) async {
    final url = Uri.parse('$baseUrl/predict');

    final response = await http
        .post(
          url,
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'image': base64Image}),
        )
        .timeout(const Duration(seconds: 4));

    if (response.statusCode != 200) {
      throw Exception('Backend error: ${response.statusCode}');
    }

    final data = jsonDecode(response.body);
    return BackendPrediction.fromJson(data);
  }

  Future<void> reset() async {
    final url = Uri.parse('$baseUrl/reset');

    await http.post(url).timeout(const Duration(seconds: 4));
  }
}