#!/usr/bin/env python3
"""
SKOLKOVO Materials Processor - Универсальный обработчик
=======================================================
Автоматически определяет тип файла и обрабатывает его.

Поддерживает:
- PDF документы
- Аудио записи (mp3, m4a, wav, ogg, flac)
- Видео (mp4, webm, mkv)

Использование:
    python process.py материал.pdf
    python process.py семинар.mp3 --title "Позитивная психология"
    python process.py папка/ --batch
"""

import sys
import os
from pathlib import Path
import argparse

# Расширения по типам
PDF_EXTENSIONS = {'.pdf'}
AUDIO_EXTENSIONS = {'.mp3', '.m4a', '.wav', '.ogg', '.flac', '.aac', '.wma'}
VIDEO_EXTENSIONS = {'.mp4', '.webm', '.mkv', '.avi', '.mov', '.wmv'}


def detect_file_type(file_path: Path) -> str:
    """Определяет тип файла."""
    suffix = file_path.suffix.lower()
    
    if suffix in PDF_EXTENSIONS:
        return "pdf"
    elif suffix in AUDIO_EXTENSIONS:
        return "audio"
    elif suffix in VIDEO_EXTENSIONS:
        return "video"
    else:
        return "unknown"


def process_file(file_path: str, **kwargs):
    """Обрабатывает файл в зависимости от типа."""
    
    path = Path(file_path)
    file_type = detect_file_type(path)
    
    print(f"\n📁 Файл: {path.name}")
    print(f"📋 Тип: {file_type}")
    
    if file_type == "pdf":
        from pdf_ai_processor import PDFAIProcessor
        
        processor = PDFAIProcessor()
        mode = kwargs.get("mode", "full_analysis")
        processor.process_pdf(str(path), mode=mode)
        
    elif file_type in ("audio", "video"):
        from audio_transcriber import process_audio_file
        
        metadata = {}
        if kwargs.get("title"):
            metadata["title"] = kwargs["title"]
        if kwargs.get("speaker"):
            metadata["speaker"] = kwargs["speaker"]
        if kwargs.get("date"):
            metadata["date"] = kwargs["date"]
        if kwargs.get("module"):
            metadata["module"] = kwargs["module"]
        
        provider = kwargs.get("provider", "openai")
        
        process_audio_file(
            str(path),
            transcription_provider=provider,
            metadata=metadata if metadata else None
        )
    
    else:
        print(f"❌ Неподдерживаемый тип файла: {path.suffix}")
        print("   Поддерживаемые форматы:")
        print(f"   PDF: {', '.join(PDF_EXTENSIONS)}")
        print(f"   Аудио: {', '.join(AUDIO_EXTENSIONS)}")
        print(f"   Видео: {', '.join(VIDEO_EXTENSIONS)}")


def process_folder(folder_path: str, **kwargs):
    """Обрабатывает все файлы в папке."""
    
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"❌ Не директория: {folder}")
        return
    
    all_extensions = PDF_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
    files = [f for f in folder.iterdir() if f.suffix.lower() in all_extensions]
    
    if not files:
        print(f"❌ Поддерживаемые файлы не найдены в: {folder}")
        return
    
    print(f"\n📂 Найдено {len(files)} файлов:")
    
    # Группируем по типу
    pdf_files = [f for f in files if detect_file_type(f) == "pdf"]
    audio_files = [f for f in files if detect_file_type(f) in ("audio", "video")]
    
    print(f"   📄 PDF: {len(pdf_files)}")
    print(f"   🎙️ Аудио/Видео: {len(audio_files)}")
    
    for i, file in enumerate(files, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(files)}]")
        process_file(str(file), **kwargs)
    
    print(f"\n{'='*60}")
    print(f"✅ Обработано {len(files)} файлов")


def main():
    parser = argparse.ArgumentParser(
        description="SKOLKOVO Materials Processor - обработка PDF и аудио"
    )
    
    parser.add_argument("path", help="Путь к файлу или папке")
    
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Обработать все файлы в папке"
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["summary", "key_concepts", "coaching_tools", "questions", "full_analysis"],
        default="full_analysis",
        help="Режим обработки PDF (default: full_analysis)"
    )
    
    parser.add_argument(
        "--provider", "-p",
        choices=["openai", "local_whisper", "assemblyai"],
        default="openai",
        help="Провайдер транскрипции (default: openai)"
    )
    
    # Метаданные для аудио
    parser.add_argument("--title", "-t", help="Название семинара")
    parser.add_argument("--speaker", "-s", help="Спикер")
    parser.add_argument("--date", "-d", help="Дата")
    parser.add_argument("--module", help="Номер модуля")
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║     📚 SKOLKOVO Materials Processor                          ║
║     PDF + Audio + Video → Structured Knowledge               ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    path = Path(args.path)
    
    if not path.exists():
        print(f"❌ Путь не найден: {path}")
        sys.exit(1)
    
    kwargs = {
        "mode": args.mode,
        "provider": args.provider,
        "title": args.title,
        "speaker": args.speaker,
        "date": args.date,
        "module": args.module
    }
    
    try:
        if path.is_dir() or args.batch:
            process_folder(str(path), **kwargs)
        else:
            process_file(str(path), **kwargs)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Прервано пользователем")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
