#!/usr/bin/env python3
"""
SKOLKOVO Materials Processor - GUI
===================================
Графический интерфейс для обработки учебных материалов программы Executive Coaching.

Запуск:
    python gui.py

Откроется веб-интерфейс по адресу http://localhost:7860
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
import tempfile

import gradio as gr

# Импортируем общие утилиты
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_env_file, mask_api_key, check_api_keys as utils_check_api_keys,
    validate_page_range, get_logger, ensure_directories,
    OUTPUT_DIR, TRANSCRIPTS_DIR, KNOWLEDGE_DIR, BASE_DIR
)

# Инициализация: загружаем .env и создаём директории
load_env_file()
ensure_directories()

# Логгер
logger = get_logger(__name__)

# Импортируем наши модули обработки
from pdf_vision_processor import PDFVisionProcessor
from audio_transcriber import AudioTranscriber, SeminarProcessor, process_audio_file


# ============================================================================
# ФУНКЦИИ ДЛЯ ОБРАБОТКИ PDF
# ============================================================================

def get_pdf_modes():
    """Возвращает доступные режимы обработки PDF."""
    return [
        ("Извлечение текста", "extract_text"),
        ("Краткое саммари", "summary"),
        ("Ключевые концепции", "key_concepts"),
        ("Инструменты коучинга", "coaching_tools"),
        ("Вопросы для рефлексии", "questions"),
        ("Полный анализ", "full_analysis")
    ]


def process_pdf_file(
    pdf_file,
    mode: str,
    page_range: str,
    dpi: int,
    progress=gr.Progress()
) -> Tuple[str, str, str]:
    """
    Обрабатывает PDF файл через Vision API.

    Returns:
        Tuple[status, markdown_result, json_path]
    """
    if pdf_file is None:
        return "Загрузите PDF файл", "", ""

    try:
        progress(0.05, desc="Инициализация...")

        # Получаем путь к файлу (Gradio 5.x передаёт строку)
        if isinstance(pdf_file, str):
            pdf_path = pdf_file
        elif hasattr(pdf_file, 'name'):
            pdf_path = pdf_file.name
        else:
            pdf_path = str(pdf_file)

        print(f"[GUI DEBUG] PDF path: {pdf_path}")

        # Проверяем существование файла
        if not Path(pdf_path).exists():
            return f"Ошибка: файл не найден: {pdf_path}", "", ""

        # Проверяем API ключ
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return "Ошибка: ANTHROPIC_API_KEY не найден в .env", "", ""

        progress(0.1, desc="Создание процессора...")
        processor = PDFVisionProcessor(api_key=api_key)

        # Парсим и валидируем диапазон страниц
        page_tuple = None
        if page_range and page_range.strip():
            try:
                start, end = validate_page_range(page_range)
                page_tuple = (start, end)
            except ValueError as e:
                return f"Ошибка в диапазоне страниц: {e}", "", ""

        # Устанавливаем DPI (ниже = быстрее)
        processor.config["dpi"] = dpi
        print(f"[GUI DEBUG] DPI: {dpi}, mode: {mode}, page_range: {page_tuple}")

        progress(0.15, desc="Конвертация PDF в изображения (может занять время)...")

        # Конвертируем PDF в изображения отдельно для показа прогресса
        images = processor.pdf_to_images(pdf_path)

        # Применяем фильтр по диапазону страниц если указан
        if page_tuple:
            start_page, end_page = page_tuple
            # Индексация с 0, page_tuple с 1
            images = images[start_page-1:end_page]
            print(f"[GUI DEBUG] Applied page range {page_tuple}: {len(images)} pages selected")
        total_pages = len(images)
        print(f"[GUI DEBUG] Converted {total_pages} pages to images")

        progress(0.3, desc=f"Конвертировано {total_pages} страниц. Отправка в Claude...")

        # Определяем номер первой страницы для правильной нумерации
        page_start = page_tuple[0] if page_tuple else 1

        # Обрабатываем страницы через process_pages (он сам делает батчинг)
        print(f"[GUI DEBUG] Processing {total_pages} pages starting from {page_start}")
        progress(0.5, desc="Анализ страниц через Claude Vision...")

        results = processor.process_pages(images, mode=mode, page_start=page_start)

        # Подсчитываем токены
        total_tokens = {"input": 0, "output": 0}
        successful_results = []
        for r in results:
            if r.get("status") == "success":
                successful_results.append(r)
                total_tokens["input"] += r.get("usage", {}).get("input_tokens", 0)
                total_tokens["output"] += r.get("usage", {}).get("output_tokens", 0)

        progress(0.9, desc="Сохранение результатов...")

        # Формируем итоговый результат
        from datetime import datetime
        session_id = f"{Path(pdf_path).stem}_vision_{datetime.now().strftime('%H%M%S')}"

        # Сохраняем результаты
        final_result = {
            "session_id": session_id,
            "source_pdf": pdf_path,
            "total_pages": total_pages,
            "mode": mode,
            "processed_at": datetime.now().isoformat(),
            "total_tokens": total_tokens,
            "results": results
        }

        # Сохраняем JSON
        json_path = OUTPUT_DIR / f"{session_id}_processed.json"
        with open(json_path, "w", encoding="utf-8") as f:
            import json
            json.dump(final_result, f, ensure_ascii=False, indent=2)

        # Создаём Markdown (используем _save_as_markdown)
        md_path = OUTPUT_DIR / f"{session_id}_processed.md"
        processor._save_as_markdown(final_result, md_path)

        # Читаем содержимое для отображения
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        status = f"""✅ Обработка завершена!

📊 Статистика:
- Страниц: {total_pages}
- Успешных батчей: {len(successful_results)}/{len(results)}
- Токенов: {total_tokens['input']:,} → {total_tokens['output']:,}

📁 Файлы:
- {md_path.name}
- {json_path.name}
"""

        progress(1.0, desc="Готово!")
        return status, md_content, str(json_path)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[GUI ERROR] {error_details}")
        return f"❌ Ошибка: {str(e)}\n\nПодробности в консоли", "", ""


def process_pdf_folder(
    folder_path: str,
    mode: str,
    dpi: int,
    progress=gr.Progress()
) -> str:
    """Обрабатывает папку с PDF файлами."""
    if not folder_path or not folder_path.strip():
        return "Укажите путь к папке"

    folder = Path(folder_path)
    if not folder.exists():
        return f"Папка не найдена: {folder_path}"

    # Находим все PDF
    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        return f"PDF файлы не найдены в {folder_path}"

    results = []
    total = len(pdf_files)

    for i, pdf_path in enumerate(pdf_files):
        progress((i + 1) / total, desc=f"Обрабатываю {pdf_path.name}...")

        try:
            processor = PDFVisionProcessor()
            processor.config["dpi"] = dpi
            result = processor.process_pdf(str(pdf_path), mode=mode)

            results.append(f"✅ {pdf_path.name}: {result['total_pages']} стр, {result['total_tokens']['output']:,} токенов")
        except Exception as e:
            results.append(f"❌ {pdf_path.name}: {str(e)}")

    return f"Обработано {total} файлов:\n\n" + "\n".join(results)


# ============================================================================
# ФУНКЦИИ ДЛЯ ТРАНСКРИПЦИИ АУДИО
# ============================================================================

def _convert_to_mp3_if_needed(audio_path: str, progress_callback=None) -> str:
    """
    Конвертирует большие аудиофайлы в MP3 для ускорения обработки.
    Возвращает путь к MP3 файлу (оригинальный или конвертированный).
    """
    from pydub import AudioSegment

    path = Path(audio_path)
    file_size_mb = path.stat().st_size / (1024 * 1024)

    # Если файл маленький или уже MP3 - не конвертируем
    if file_size_mb < 25 or path.suffix.lower() == '.mp3':
        return audio_path

    # Для больших файлов в экзотических форматах - конвертируем в MP3
    if progress_callback:
        progress_callback(f"Конвертирую {file_size_mb:.0f}MB в MP3 (может занять несколько минут)...")

    print(f"   🔄 Конвертирую {path.name} ({file_size_mb:.0f}MB) в MP3...")

    # Создаём временный MP3 файл
    mp3_path = OUTPUT_DIR / f"{path.stem}_converted.mp3"

    # Конвертируем через ffmpeg напрямую (быстрее чем pydub для больших файлов)
    try:
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', str(path), '-vn', '-acodec', 'libmp3lame', '-q:a', '4', str(mp3_path)],
            capture_output=True,
            text=True,
            timeout=600  # 10 минут таймаут
        )
        if result.returncode == 0 and mp3_path.exists():
            new_size = mp3_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ Конвертировано: {new_size:.0f}MB")
            return str(mp3_path)
    except subprocess.TimeoutExpired:
        print("   ⚠️ Таймаут конвертации")
    except Exception as e:
        print(f"   ⚠️ Ошибка ffmpeg: {e}")

    # Если ffmpeg не сработал, пробуем pydub
    try:
        audio = AudioSegment.from_file(str(path))
        audio.export(str(mp3_path), format="mp3")
        return str(mp3_path)
    except Exception as e:
        print(f"   ⚠️ Ошибка pydub: {e}")
        return audio_path  # Возвращаем оригинал


def transcribe_audio_file(
    audio_file,
    title: str,
    speaker: str,
    date: str,
    module: str,
    provider: str,
    transcript_only: bool,
    progress=gr.Progress()
) -> Tuple[str, str, str]:
    """
    Транскрибирует аудио файл.

    Returns:
        Tuple[status, markdown_result, transcript_text]
    """
    if audio_file is None:
        return "Загрузите аудио файл", "", ""

    try:
        progress(0.05, desc="Проверка файла...")

        # Проверяем API ключи
        if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            return "Ошибка: OPENAI_API_KEY не найден", "", ""

        if not transcript_only and not os.environ.get("ANTHROPIC_API_KEY"):
            return "Ошибка: ANTHROPIC_API_KEY не найден", "", ""

        audio_path = audio_file.name if hasattr(audio_file, 'name') else audio_file

        # Получаем информацию о файле
        file_size_mb = Path(audio_path).stat().st_size / (1024 * 1024)
        file_ext = Path(audio_path).suffix.lower()

        progress(0.1, desc=f"Файл: {file_size_mb:.0f}MB ({file_ext})")

        # Для больших файлов в экзотических форматах - конвертируем в MP3
        if file_size_mb > 25 and file_ext not in ['.mp3', '.wav', '.m4a']:
            progress(0.15, desc=f"Конвертирую {file_size_mb:.0f}MB в MP3...")
            audio_path = _convert_to_mp3_if_needed(
                audio_path,
                lambda msg: progress(0.2, desc=msg)
            )
            progress(0.25, desc="Конвертация завершена")

        # Метаданные
        metadata = {}
        if title:
            metadata["title"] = title
        if speaker:
            metadata["speaker"] = speaker
        if date:
            metadata["date"] = date
        if module:
            metadata["module"] = module

        if transcript_only:
            # Только транскрипция
            progress(0.3, desc="Транскрибирую аудио...")
            transcriber = AudioTranscriber(provider=provider)
            transcript = transcriber.transcribe(audio_path)

            # Сохраняем транскрипт
            output_name = Path(audio_path).stem
            transcript_path = TRANSCRIPTS_DIR / f"{output_name}_transcript.txt"
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript["text"])

            status = f"""✅ Транскрипция завершена!

⏱️ Длительность: {transcript['duration'] / 60:.1f} мин
📝 Символов: {len(transcript['text']):,}
🔧 Провайдер: {provider}

📁 Файл: {transcript_path.name}
"""
            return status, "", transcript["text"]

        else:
            # Полная обработка
            progress(0.3, desc="Транскрипция (большие файлы: 3-10 мин)...")

            transcriber = AudioTranscriber(provider=provider)
            transcript = transcriber.transcribe(audio_path)

            progress(0.5, desc="Анализ через Claude...")

            processor = SeminarProcessor()
            results = processor.process_seminar(transcript, metadata)

            progress(0.9, desc="Сохранение...")

            json_path, md_path = processor.save_results(results)

            # Читаем Markdown
            md_content = ""
            if md_path.exists():
                with open(md_path, encoding="utf-8") as f:
                    md_content = f.read()

            status = f"""✅ Обработка завершена!

⏱️ Длительность: {transcript['duration'] / 60:.1f} мин
📝 Символов в транскрипции: {len(transcript['text']):,}
🔧 Провайдер: {provider}

📁 Файлы:
- {md_path.name}
- {json_path.name}
"""

            progress(1.0, desc="Готово!")
            return status, md_content, transcript["text"]

    except Exception as e:
        import traceback
        return f"❌ Ошибка: {str(e)}\n\n{traceback.format_exc()}", "", ""


# ============================================================================
# ФУНКЦИИ ДЛЯ ПРОСМОТРА РЕЗУЛЬТАТОВ
# ============================================================================

def get_processed_files() -> List[str]:
    """Возвращает список обработанных файлов."""
    files = []

    # JSON файлы из output
    for f in OUTPUT_DIR.glob("*_processed.json"):
        files.append(f.name)

    # Транскрипты
    for f in TRANSCRIPTS_DIR.glob("*.txt"):
        files.append(f"[transcripts] {f.name}")

    return sorted(files, reverse=True)


def load_processed_file(filename: str) -> Tuple[str, str]:
    """Загружает обработанный файл."""
    if not filename:
        return "", ""

    try:
        if filename.startswith("[transcripts]"):
            # Транскрипт
            actual_name = filename.replace("[transcripts] ", "")
            path = TRANSCRIPTS_DIR / actual_name
            with open(path, encoding="utf-8") as f:
                content = f.read()
            return content, ""

        else:
            # JSON + MD
            json_path = OUTPUT_DIR / filename
            md_path = json_path.with_suffix("").with_suffix(".md")

            md_content = ""
            json_content = ""

            if md_path.exists():
                with open(md_path, encoding="utf-8") as f:
                    md_content = f.read()

            if json_path.exists():
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
                    json_content = json.dumps(data, ensure_ascii=False, indent=2)

            return md_content, json_content

    except Exception as e:
        return f"Ошибка: {e}", ""


def refresh_file_list():
    """Обновляет список файлов."""
    return gr.Dropdown(choices=get_processed_files())


def search_in_results(query: str) -> str:
    """Поиск по обработанным материалам."""
    if not query or len(query) < 2:
        return "Введите поисковый запрос (минимум 2 символа)"

    results = []
    query_lower = query.lower()

    # Поиск в markdown файлах
    for md_file in OUTPUT_DIR.glob("*_processed.md"):
        try:
            with open(md_file, encoding="utf-8") as f:
                content = f.read()

            if query_lower in content.lower():
                # Находим контекст
                idx = content.lower().find(query_lower)
                start = max(0, idx - 100)
                end = min(len(content), idx + 200)
                snippet = content[start:end].replace("\n", " ")

                results.append(f"### {md_file.name}\n\n...{snippet}...\n")
        except Exception:
            continue

    # Поиск в транскриптах
    for txt_file in TRANSCRIPTS_DIR.glob("*.txt"):
        try:
            with open(txt_file, encoding="utf-8") as f:
                content = f.read()

            if query_lower in content.lower():
                idx = content.lower().find(query_lower)
                start = max(0, idx - 100)
                end = min(len(content), idx + 200)
                snippet = content[start:end].replace("\n", " ")

                results.append(f"### [transcript] {txt_file.name}\n\n...{snippet}...\n")
        except Exception:
            continue

    if results:
        return f"Найдено {len(results)} результатов:\n\n" + "\n---\n".join(results)
    else:
        return f"По запросу '{query}' ничего не найдено"


# ============================================================================
# ФУНКЦИИ ДЛЯ НАСТРОЕК
# ============================================================================

def check_api_keys() -> str:
    """Проверяет наличие API ключей (безопасно, без раскрытия значений)."""
    keys_info = utils_check_api_keys()
    status = []

    for key_name, info in keys_info.items():
        if info["present"]:
            status.append(f"✅ {key_name}: {info['masked']}")
        else:
            status.append(f"❌ {key_name}: не установлен")

    return "\n".join(status)


def get_stats() -> str:
    """Возвращает статистику."""
    pdf_count = len(list(OUTPUT_DIR.glob("*_processed.json")))
    transcript_count = len(list(TRANSCRIPTS_DIR.glob("*.txt")))

    # Подсчитываем токены
    total_input = 0
    total_output = 0

    for json_file in OUTPUT_DIR.glob("*_processed.json"):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            total_input += data.get("total_tokens", {}).get("input", 0)
            total_output += data.get("total_tokens", {}).get("output", 0)
        except Exception:
            continue

    return f"""📊 Статистика базы знаний:

📄 PDF обработано: {pdf_count}
🎙️ Аудио транскрибировано: {transcript_count}

💰 Использовано токенов:
- Входящих: {total_input:,}
- Исходящих: {total_output:,}
- Всего: {total_input + total_output:,}

📁 Директории:
- Output: {OUTPUT_DIR}
- Transcripts: {TRANSCRIPTS_DIR}
"""


# ============================================================================
# СОЗДАНИЕ ИНТЕРФЕЙСА
# ============================================================================

def create_interface():
    """Создает Gradio интерфейс."""

    with gr.Blocks(title="SKOLKOVO Materials Processor") as app:

        # Заголовок
        gr.Markdown("""
        # 📚 SKOLKOVO Materials Processor

        **Система обработки учебных материалов программы Executive Coaching & Mentoring**

        Обрабатывайте PDF документы, транскрибируйте аудио записи, просматривайте результаты.
        """, elem_classes="main-header")

        with gr.Tabs():

            # ========================
            # TAB 1: PDF ОБРАБОТКА
            # ========================
            with gr.Tab("PDF Обработка"):
                gr.Markdown("### Обработка PDF через Claude Vision")

                with gr.Row():
                    with gr.Column(scale=1):
                        pdf_file = gr.File(
                            label="PDF файл",
                            file_types=[".pdf"],
                            type="filepath"
                        )

                        pdf_mode = gr.Dropdown(
                            label="Режим обработки",
                            choices=[m[0] for m in get_pdf_modes()],
                            value="Полный анализ",
                            info="Выберите тип анализа"
                        )

                        with gr.Row():
                            pdf_pages = gr.Textbox(
                                label="Страницы",
                                placeholder="1-10 или пусто для всех",
                                scale=1
                            )
                            pdf_dpi = gr.Slider(
                                label="DPI",
                                minimum=72,
                                maximum=300,
                                value=150,
                                step=10,
                                scale=1
                            )

                        pdf_btn = gr.Button("🚀 Обработать PDF", variant="primary")

                    with gr.Column(scale=2):
                        pdf_status = gr.Textbox(
                            label="Статус",
                            lines=8,
                            interactive=False
                        )
                        pdf_json_path = gr.Textbox(
                            label="Путь к JSON",
                            visible=False
                        )

                pdf_result = gr.Markdown(
                    label="Результат",
                    value="*Результат появится здесь после обработки*"
                )

                # Обработка папки
                gr.Markdown("---\n### Пакетная обработка папки")

                with gr.Row():
                    folder_path = gr.Textbox(
                        label="Путь к папке",
                        placeholder="/Users/.../Documents/PDFs",
                        scale=3
                    )
                    folder_btn = gr.Button("📁 Обработать папку", scale=1)

                folder_result = gr.Textbox(
                    label="Результат пакетной обработки",
                    lines=10,
                    interactive=False
                )

                # Привязываем функции
                def get_mode_key(mode_name):
                    modes = dict(get_pdf_modes())
                    for name, key in get_pdf_modes():
                        if name == mode_name:
                            return key
                    return "full_analysis"

                pdf_btn.click(
                    fn=lambda f, m, p, d: process_pdf_file(f, get_mode_key(m), p, d),
                    inputs=[pdf_file, pdf_mode, pdf_pages, pdf_dpi],
                    outputs=[pdf_status, pdf_result, pdf_json_path]
                )

                folder_btn.click(
                    fn=lambda p, m, d: process_pdf_folder(p, get_mode_key(m), d),
                    inputs=[folder_path, pdf_mode, pdf_dpi],
                    outputs=[folder_result]
                )

            # ========================
            # TAB 2: АУДИО ТРАНСКРИПЦИЯ
            # ========================
            with gr.Tab("🎙️ Аудио Транскрипция"):
                gr.Markdown("### Транскрипция аудио через Whisper + анализ через Claude")

                with gr.Row():
                    with gr.Column(scale=1):
                        audio_file = gr.File(
                            label="Аудио файл",
                            file_types=[".mp3", ".m4a", ".wav", ".mp4", ".webm", ".ogg", ".flac", ".qta"],
                            type="filepath"
                        )

                        audio_title = gr.Textbox(
                            label="Название",
                            placeholder="Семинар по коучингу"
                        )

                        audio_speaker = gr.Textbox(
                            label="Спикер",
                            placeholder="Елена Клековкина"
                        )

                        with gr.Row():
                            audio_date = gr.Textbox(
                                label="Дата",
                                placeholder="2025-11-25",
                                scale=1
                            )
                            audio_module = gr.Textbox(
                                label="Модуль",
                                placeholder="1",
                                scale=1
                            )

                        audio_provider = gr.Radio(
                            label="Провайдер транскрипции",
                            choices=["openai", "local_whisper"],
                            value="openai",
                            info="OpenAI Whisper API или локальный Whisper"
                        )

                        audio_only = gr.Checkbox(
                            label="Только транскрипция (без анализа Claude)",
                            value=False
                        )

                        audio_btn = gr.Button("🎤 Транскрибировать", variant="primary")

                    with gr.Column(scale=2):
                        audio_status = gr.Textbox(
                            label="Статус",
                            lines=10,
                            interactive=False
                        )

                with gr.Tabs():
                    with gr.Tab("Анализ"):
                        audio_result = gr.Markdown(
                            label="Результат анализа",
                            value="*Анализ появится здесь*"
                        )

                    with gr.Tab("Транскрипт"):
                        audio_transcript = gr.Textbox(
                            label="Транскрипт",
                            lines=20,
                            interactive=False
                        )

                audio_btn.click(
                    fn=transcribe_audio_file,
                    inputs=[audio_file, audio_title, audio_speaker, audio_date, audio_module, audio_provider, audio_only],
                    outputs=[audio_status, audio_result, audio_transcript]
                )

            # ========================
            # TAB 3: ПРОСМОТР РЕЗУЛЬТАТОВ
            # ========================
            with gr.Tab("📖 Просмотр результатов"):
                gr.Markdown("### Просмотр обработанных материалов")

                with gr.Row():
                    file_dropdown = gr.Dropdown(
                        label="Выберите файл",
                        choices=get_processed_files(),
                        scale=3
                    )
                    refresh_btn = gr.Button("🔄 Обновить", scale=1)

                with gr.Tabs():
                    with gr.Tab("Markdown"):
                        result_md = gr.Markdown(
                            label="Содержимое",
                            value="*Выберите файл для просмотра*"
                        )

                    with gr.Tab("JSON"):
                        result_json = gr.Code(
                            label="JSON данные",
                            language="json"
                        )

                gr.Markdown("---\n### Поиск по материалам")

                with gr.Row():
                    search_query = gr.Textbox(
                        label="Поисковый запрос",
                        placeholder="коучинг, рефлексия, лидерство...",
                        scale=3
                    )
                    search_btn = gr.Button("🔍 Искать", scale=1)

                search_result = gr.Markdown(
                    label="Результаты поиска"
                )

                file_dropdown.change(
                    fn=load_processed_file,
                    inputs=[file_dropdown],
                    outputs=[result_md, result_json]
                )

                refresh_btn.click(
                    fn=lambda: gr.Dropdown(choices=get_processed_files()),
                    outputs=[file_dropdown]
                )

                search_btn.click(
                    fn=search_in_results,
                    inputs=[search_query],
                    outputs=[search_result]
                )

            # ========================
            # TAB 4: НАСТРОЙКИ
            # ========================
            with gr.Tab("⚙️ Настройки"):
                gr.Markdown("### Настройки и статистика")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### API Ключи")
                        api_status = gr.Textbox(
                            label="Статус API",
                            value=check_api_keys(),
                            lines=4,
                            interactive=False
                        )

                        check_api_btn = gr.Button("🔑 Проверить ключи")
                        check_api_btn.click(
                            fn=check_api_keys,
                            outputs=[api_status]
                        )

                        gr.Markdown("""
                        **Как добавить ключи:**

                        Отредактируйте файл `.env` в папке проекта:
                        ```
                        ANTHROPIC_API_KEY=sk-ant-...
                        OPENAI_API_KEY=sk-...
                        ```
                        """)

                    with gr.Column():
                        gr.Markdown("#### Статистика")
                        stats = gr.Textbox(
                            label="Статистика",
                            value=get_stats(),
                            lines=12,
                            interactive=False
                        )

                        refresh_stats_btn = gr.Button("📊 Обновить статистику")
                        refresh_stats_btn.click(
                            fn=get_stats,
                            outputs=[stats]
                        )

        # Футер
        gr.Markdown("""
        ---

        **SKOLKOVO Materials Processor** | Executive Coaching & Mentoring Program

        Поддерживаемые форматы:
        - PDF: обычные, сканы, презентации (через Claude Vision)
        - Аудио: MP3, M4A, WAV, MP4, WEBM, OGG, FLAC, QTA
        """)

    return app


# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║     📚 SKOLKOVO Materials Processor - GUI                    ║
║                                                              ║
║     Открывается в браузере: http://localhost:7860            ║
╚══════════════════════════════════════════════════════════════╝
""")

    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
