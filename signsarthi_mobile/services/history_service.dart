import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

class HistoryService {
  static const String _historyKey = 'sentence_history';

  static Future<List<String>> getHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_historyKey);

    if (raw == null || raw.isEmpty) {
      return [];
    }

    final List<dynamic> decoded = jsonDecode(raw);
    return decoded.map((e) => e.toString()).toList();
  }

  static Future<void> addSentence(String sentence) async {
    final clean = sentence.trim();

    if (clean.isEmpty) return;

    final prefs = await SharedPreferences.getInstance();
    final history = await getHistory();

    history.insert(0, clean);

    // Keep only latest 30 sentences
    final limited = history.take(30).toList();

    await prefs.setString(_historyKey, jsonEncode(limited));
  }

  static Future<void> clearHistory() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_historyKey);
  }
}