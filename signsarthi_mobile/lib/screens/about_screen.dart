import 'package:flutter/material.dart';
import '../theme/app_theme.dart';

class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  Widget _infoCard({
    required IconData icon,
    required String title,
    required String body,
  }) {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppTheme.surfaceLight),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: AppTheme.accent, size: 26),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    color: AppTheme.textPrimary,
                    fontSize: 16,
                    fontWeight: FontWeight.w800,
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  body,
                  style: const TextStyle(
                    color: AppTheme.textSecondary,
                    fontSize: 14,
                    height: 1.45,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.background,
      appBar: AppBar(
        title: const Text('About'),
      ),
      body: ListView(
        padding: const EdgeInsets.all(22),
        children: [
          const Text(
            'SignSarthi AI',
            style: TextStyle(
              color: AppTheme.textPrimary,
              fontSize: 32,
              fontWeight: FontWeight.w900,
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'An AI-powered Indian Sign Language recognition app designed to help deaf and speech-impaired people communicate more easily.',
            style: TextStyle(
              color: AppTheme.textSecondary,
              fontSize: 15,
              height: 1.5,
            ),
          ),
          const SizedBox(height: 26),

          _infoCard(
            icon: Icons.camera_alt_outlined,
            title: 'Live Camera Recognition',
            body:
                'The app uses the phone camera to observe body and hand movement in real time.',
          ),
          const SizedBox(height: 14),

          _infoCard(
            icon: Icons.psychology_outlined,
            title: 'On-device AI',
            body:
                'This version uses Google ML Kit Pose Detection on-device for fast offline landmark detection. Full trained TFLite model integration can be added next.',
          ),
          const SizedBox(height: 14),

          _infoCard(
            icon: Icons.text_fields,
            title: 'Sentence Builder',
            body:
                'Detected words are added together to form a readable sentence that can be saved in history.',
          ),
          const SizedBox(height: 14),

          _infoCard(
            icon: Icons.school_outlined,
            title: 'Current Signs',
            body:
                'Current prototype signs: YES, WATER, I, and IDLE. More ISL words can be added as the model improves.',
          ),
        ],
      ),
    );
  }
}