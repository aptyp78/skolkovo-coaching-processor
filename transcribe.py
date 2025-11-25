#!/usr/bin/env python3
"""
Быстрая транскрипция аудио семинара.

Использование:
    python transcribe.py запись_семинара.mp3
    python transcribe.py запись.m4a --title "Позитивная психология" --speaker "Улановский"
    python transcribe.py video.mp4 --provider local_whisper
"""

import sys
import os
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

from audio_transcriber import process_audio_file, AudioTranscriber, SeminarProcessor


def quick_transcribe():
    """Быстрая транскрипция с минимальными параметрами."""
    
    if len(sys.argv) < 2:
        print("""
🎙️ SKOLKOVO Seminar Transcriber

Использование:
    python transcribe.py <аудио_файл> [опции]

Примеры:
    python transcribe.py семинар.mp3
    python transcribe.py лекция.m4a --title "Коучинг" --speaker "Улановский"
    python transcribe.py видео.mp4 --transcript-only
    
Поддерживаемые форматы: mp3, m4a, wav, mp4, webm, ogg, flac

Опции:
    --title "Название"     Название семинара
    --speaker "Имя"        Спикер
    --date "2025-11-25"    Дата
    --module "2"           Номер модуля
    --transcript-only      Только транскрипция без анализа
    --provider X           openai (по умолчанию), local_whisper, assemblyai

API ключи (через переменные окружения):
    export OPENAI_API_KEY="sk-..."
    export ANTHROPIC_API_KEY="sk-ant-..."
""")
        return
    
    audio_path = sys.argv[1]
    
    if not Path(audio_path).exists():
        print(f"❌ Файл не найден: {audio_path}")
        return
    
    # Парсим простые аргументы
    metadata = {}
    transcript_only = False
    provider = "openai"
    
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--title" and i + 1 < len(args):
            metadata["title"] = args[i + 1]
            i += 2
        elif args[i] == "--speaker" and i + 1 < len(args):
            metadata["speaker"] = args[i + 1]
            i += 2
        elif args[i] == "--date" and i + 1 < len(args):
            metadata["date"] = args[i + 1]
            i += 2
        elif args[i] == "--module" and i + 1 < len(args):
            metadata["module"] = args[i + 1]
            i += 2
        elif args[i] == "--provider" and i + 1 < len(args):
            provider = args[i + 1]
            i += 2
        elif args[i] == "--transcript-only":
            transcript_only = True
            i += 1
        else:
            i += 1
    
    # Проверяем API ключи
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("❌ Установите OPENAI_API_KEY: export OPENAI_API_KEY='sk-...'")
        return
    
    if not transcript_only and not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ Установите ANTHROPIC_API_KEY: export ANTHROPIC_API_KEY='sk-ant-...'")
        return
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║     🎙️ SKOLKOVO Seminar Transcriber                          ║
╚══════════════════════════════════════════════════════════════╝

📁 Файл: {audio_path}
🔧 Провайдер: {provider}
📝 Режим: {"Только транскрипция" if transcript_only else "Транскрипция + Анализ"}
""")
    
    if metadata:
        print("📋 Метаданные:")
        for k, v in metadata.items():
            print(f"   {k}: {v}")
        print()
    
    try:
        if transcript_only:
            # Только транскрипция
            transcriber = AudioTranscriber(provider=provider)
            transcript = transcriber.transcribe(audio_path)
            
            output_path = Path(audio_path).with_suffix(".txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcript["text"])
            
            print(f"\n✅ Готово!")
            print(f"   📝 Транскрипт: {output_path}")
            print(f"   ⏱️ Длительность: {transcript['duration'] / 60:.1f} мин")
            print(f"   📊 Символов: {len(transcript['text']):,}")
            
        else:
            # Полная обработка
            process_audio_file(
                audio_path,
                transcription_provider=provider,
                metadata=metadata
            )
    
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    quick_transcribe()
