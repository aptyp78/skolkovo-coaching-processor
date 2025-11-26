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

# Локальные процессоры
try:
    from local_processor import (
        LocalPDFProcessor, LocalVisionProcessor, LocalWhisperTranscriber,
        check_local_requirements, OllamaClient
    )
    LOCAL_AVAILABLE = True
except ImportError:
    LOCAL_AVAILABLE = False
    logger.warning("Локальные процессоры недоступны")

# Глобальная переменная для хранения текущего процесса конвертации
_current_conversion_process = None


def get_system_memory_gb() -> int:
    """Получает объём RAM системы в GB."""
    try:
        import subprocess
        result = subprocess.run(
            ['sysctl', '-n', 'hw.memsize'],
            capture_output=True, text=True, timeout=5
        )
        bytes_ram = int(result.stdout.strip())
        return bytes_ram // (1024 ** 3)
    except Exception:
        return 32  # Консервативная оценка по умолчанию


def estimate_model_size(model_name: str) -> tuple:
    """
    Оценивает размер модели по названию.
    Возвращает (размер_в_GB, можно_запустить, рекомендация).
    """
    name_lower = model_name.lower()

    # Извлекаем размер из названия (7b, 13b, 34b, 70b, 235b, 671b)
    import re
    size_match = re.search(r'(\d+)b', name_lower)

    if size_match:
        params_b = int(size_match.group(1))
    else:
        # Оценка по умолчанию для известных моделей
        if 'llama3.3' in name_lower or 'llama3:latest' in name_lower:
            params_b = 70
        elif 'llama3:8b' in name_lower or 'qwen3:8b' in name_lower:
            params_b = 8
        elif 'llava' in name_lower and '34b' not in name_lower:
            params_b = 7  # llava:latest обычно 7b
        else:
            params_b = 7  # По умолчанию предполагаем маленькую модель

    # Приблизительный размер в памяти (параметры * 2 байта для fp16 + накладные расходы)
    estimated_ram_gb = (params_b * 2) // 1 + 2  # ~2GB на параметр + 2GB overhead

    # Получаем RAM системы
    system_ram = get_system_memory_gb()

    # Определяем, можно ли запустить (оставляем 16GB для системы)
    available_ram = system_ram - 16
    can_run = estimated_ram_gb <= available_ram

    # Рекомендации
    if params_b >= 200:
        recommendation = "🔴 Слишком большая"
    elif params_b >= 70:
        if system_ram >= 128:
            recommendation = "🟡 Медленная, но работает"
        else:
            recommendation = "🔴 Требует много RAM"
    elif params_b >= 30:
        recommendation = "🟢 Хороший баланс"
    elif params_b >= 7:
        recommendation = "🟢 Быстрая"
    else:
        recommendation = "🟢 Очень быстрая"

    return estimated_ram_gb, can_run, recommendation, params_b


def get_ollama_models() -> dict:
    """
    Получает список доступных моделей Ollama с фильтрацией и рекомендациями.
    Возвращает словарь с текстовыми и vision моделями.
    """
    result = {
        "text": [],           # Модели для текстового анализа
        "vision": [],         # Модели с поддержкой изображений
        "all": [],            # Все модели
        "recommended": [],    # Рекомендованные модели
        "text_choices": [],   # Форматированные варианты для Dropdown (текст)
        "vision_choices": [], # Форматированные варианты для Dropdown (vision)
        "system_ram": 0
    }

    if not LOCAL_AVAILABLE:
        return result

    try:
        client = OllamaClient()
        models = client.list_models()
        result["all"] = models
        result["system_ram"] = get_system_memory_gb()

        # Vision/OCR ключевые слова (модели, способные работать с изображениями)
        vision_keywords = [
            'llava', 'vision', 'bakllava', 'moondream', 'cogvlm',
            'qwen-vl', 'qwen3-vl', 'qwen2.5vl', 'qwen2-vl',  # Qwen Vision-Language
            'granite3.2-vision', 'granite-vision',            # IBM Granite OCR
            'deepseek-ocr', 'minicpm-v',                      # Специализированные OCR
            'llama3.2-vision', 'llama-vision'                 # Meta Vision
        ]

        # Модели с лучшим OCR (приоритет для документов с текстом)
        ocr_optimized = ['granite3.2-vision', 'deepseek-ocr', 'qwen2.5vl', 'minicpm-v']

        # Фильтруем модели с "-cloud" суффиксом (они требуют API и не работают локально)
        cloud_keywords = ['-cloud', ':cloud', 'cloud-']

        for model in models:
            model_lower = model.lower()

            # Пропускаем "облачные" модели (это заглушки, требующие внешнего API)
            if any(kw in model_lower for kw in cloud_keywords):
                continue

            # Получаем информацию о модели
            ram_gb, can_run, recommendation, params = estimate_model_size(model)

            # Форматируем для отображения
            display_name = f"{model} {recommendation}"

            # Разделяем на vision и текстовые
            is_vision = any(kw in model_lower for kw in vision_keywords)

            # Проверяем, оптимизирована ли модель для OCR
            is_ocr_optimized = any(kw in model_lower for kw in ocr_optimized)

            if is_vision:
                result["vision"].append(model)
                if can_run:
                    # Добавляем метку OCR для оптимизированных моделей
                    if is_ocr_optimized:
                        display_name = f"⭐ {model} {recommendation} [OCR]"
                    # Gradio 5.x: используем строки напрямую (display_name включает model)
                    result["vision_choices"].append(display_name)
            else:
                result["text"].append(model)
                if can_run:
                    result["text_choices"].append(display_name)

            # Добавляем в рекомендованные если хороший баланс
            if can_run and params <= 70 and "🟢" in recommendation:
                result["recommended"].append(model)

        # Сортируем: сначала OCR-оптимизированные (⭐), потом 🟢, потом 🟡
        def sort_key(display):
            if "⭐" in display:      # OCR-оптимизированные первыми
                return (0, display)
            elif "🟢" in display:
                return (1, display)
            elif "🟡" in display:
                return (2, display)
            else:
                return (3, display)

        result["text_choices"].sort(key=sort_key)
        result["vision_choices"].sort(key=sort_key)

        return result
    except Exception as e:
        logger.warning(f"Не удалось получить список моделей Ollama: {e}")
        return result


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
    processing_mode: str = "cloud",  # "cloud" или "local"
    ollama_model: str = None,        # Модель Ollama для локальной обработки
    progress=gr.Progress()
) -> Tuple[str, str, str]:
    """
    Обрабатывает PDF файл через Vision API или локально.

    Args:
        processing_mode: "cloud" (Claude API) или "local" (Ollama)
        ollama_model: Модель Ollama (llava:34b, llama3.3 и т.д.)

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

        print(f"[GUI DEBUG] PDF path: {pdf_path}, mode: {processing_mode}")

        # Проверяем существование файла
        if not Path(pdf_path).exists():
            return f"Ошибка: файл не найден: {pdf_path}", "", ""

        # ЛОКАЛЬНАЯ ОБРАБОТКА
        if processing_mode == "local":
            if not LOCAL_AVAILABLE:
                return "Ошибка: локальные процессоры не установлены", "", ""

            # Определяем модель (извлекаем имя из display_name)
            model_display = ollama_model or "llava:34b"
            model = extract_model_name(model_display)
            progress(0.1, desc=f"Создание локального процессора ({model})...")

            # Проверяем, это vision модель или текстовая
            vision_keywords = [
                'llava', 'vision', 'bakllava', 'moondream', 'cogvlm',
                'qwen-vl', 'qwen3-vl', 'qwen2.5vl', 'qwen2-vl',
                'granite3.2-vision', 'granite-vision',
                'deepseek-ocr', 'minicpm-v',
                'llama3.2-vision', 'llama-vision'
            ]
            is_vision = any(kw in model.lower() for kw in vision_keywords)

            try:
                if is_vision:
                    processor = LocalVisionProcessor({"dpi": dpi, "vision_model": model})
                else:
                    # Для текстовых моделей используем LocalPDFProcessor
                    processor = LocalPDFProcessor({"text_model": model})
            except ConnectionError as e:
                return f"Ошибка: {e}\n\nЗапустите Ollama: ollama serve", "", ""

            # Парсим диапазон страниц (только для Vision процессора)
            page_tuple = None
            if page_range and page_range.strip():
                try:
                    start, end = validate_page_range(page_range)
                    page_tuple = (start, end)
                except ValueError as e:
                    return f"Ошибка в диапазоне страниц: {e}", "", ""

            progress(0.2, desc="Обработка PDF...")

            # Обрабатываем локально (разная логика для Vision и текстовых моделей)
            if is_vision:
                result = processor.process_pdf(pdf_path, mode=mode, page_range=page_tuple)
            else:
                # LocalPDFProcessor не поддерживает page_range
                result = processor.process_pdf(pdf_path, mode=mode)

            # Читаем markdown результат
            session_id = result["session_id"]
            md_path = OUTPUT_DIR / f"{session_id}_processed.md"
            json_path = OUTPUT_DIR / f"{session_id}_processed.json"

            md_content = ""
            if md_path.exists():
                with open(md_path, "r", encoding="utf-8") as f:
                    md_content = f.read()

            status = f"""✅ Локальная обработка завершена!

🏠 Процессор: Ollama ({model})
📊 Статистика:
- Страниц: {result['total_pages']}
- Успешно: {result['successful_pages']}/{result['total_pages']}

📁 Файлы:
- {md_path.name}
- {json_path.name}
"""
            progress(1.0, desc="Готово!")
            return status, md_content, str(json_path)

        # ОБЛАЧНАЯ ОБРАБОТКА (Claude API)
        # Проверяем API ключ
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return "Ошибка: ANTHROPIC_API_KEY не найден в .env", "", ""

        progress(0.1, desc="Создание процессора (Claude API)...")
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
    import re

    path = Path(audio_path)
    file_size_mb = path.stat().st_size / (1024 * 1024)

    # Если файл маленький или уже MP3 - не конвертируем
    if file_size_mb < 25 or path.suffix.lower() == '.mp3':
        return audio_path

    print(f"   🔄 Конвертирую {path.name} ({file_size_mb:.0f}MB) в MP3...")

    # Создаём временный MP3 файл
    mp3_path = OUTPUT_DIR / f"{path.stem}_converted.mp3"

    # Получаем длительность для расчёта прогресса
    try:
        duration_result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(path)],
            capture_output=True,
            text=True
        )
        total_duration = float(duration_result.stdout.strip())
    except:
        total_duration = None

    # Конвертируем через ffmpeg с отслеживанием прогресса
    global _current_conversion_process
    try:
        print(f"   🎵 Запускаю ffmpeg для {path.name}...")

        process = subprocess.Popen(
            ['ffmpeg', '-y', '-i', str(path), '-vn', '-acodec', 'libmp3lame', '-q:a', '4', str(mp3_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        _current_conversion_process = process  # Сохраняем для возможности отмены

        # Читаем stderr для отслеживания прогресса
        while True:
            line = process.stderr.readline()
            if not line and process.poll() is not None:
                break

            if progress_callback and total_duration and 'time=' in line:
                # Парсим время: time=00:01:23.45
                time_match = re.search(r'time=(\d{2}):(\d{2}):(\d{2})', line)
                if time_match:
                    h, m, s = map(int, time_match.groups())
                    current_time = h * 3600 + m * 60 + s
                    progress_pct = min(0.95, current_time / total_duration)
                    progress_callback(f"Конвертация: {int(progress_pct * 100)}%")

        process.wait()
        _current_conversion_process = None  # Очищаем после завершения

        if process.returncode == 0 and mp3_path.exists():
            new_size = mp3_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ Конвертировано: {new_size:.0f}MB")
            if progress_callback:
                progress_callback(f"Конвертация завершена: {new_size:.0f}MB")
            return str(mp3_path)
        else:
            stderr = process.stderr.read() if process.stderr else ""
            print(f"   ❌ ffmpeg ошибка (код {process.returncode}):\n{stderr[-500:]}")

    except Exception as e:
        print(f"   ⚠️ Ошибка ffmpeg: {e}")
        _current_conversion_process = None

    # Если ffmpeg не сработал, пробуем pydub
    try:
        if progress_callback:
            progress_callback("Пробую альтернативный метод конвертации...")
        audio = AudioSegment.from_file(str(path))
        audio.export(str(mp3_path), format="mp3")
        return str(mp3_path)
    except Exception as e:
        print(f"   ⚠️ Ошибка pydub: {e}")
        return audio_path  # Возвращаем оригинал


def cancel_audio_conversion():
    """Отменяет текущую конвертацию аудио."""
    global _current_conversion_process
    if _current_conversion_process and _current_conversion_process.poll() is None:
        _current_conversion_process.terminate()
        _current_conversion_process = None
        return "❌ Конвертация отменена пользователем"
    return "⚠️ Нет активной конвертации для отмены"


def transcribe_audio_file(
    audio_file,
    title: str,
    speaker: str,
    date: str,
    module: str,
    provider: str,
    transcript_only: bool,
    analysis_mode: str = "cloud",  # "cloud" или "local" для анализа
    analysis_model: str = None,    # Модель Ollama для локального анализа
    progress=gr.Progress()
) -> Tuple[str, str, str]:
    """
    Транскрибирует аудио файл.

    Args:
        provider: "openai", "local_whisper", "mlx_whisper"
        analysis_mode: "cloud" (Claude) или "local" (Ollama) для анализа транскрипта
        analysis_model: Модель Ollama для анализа (llama3.3 и т.д.)

    Returns:
        Tuple[status, markdown_result, transcript_text]
    """
    if audio_file is None:
        return "Загрузите аудио файл", "", ""

    try:
        # Извлекаем чистые значения из Radio (Gradio 5.x передаёт display-строки)
        provider = extract_radio_value(provider)
        analysis_mode = extract_radio_value(analysis_mode)

        progress(0.05, desc="Проверка файла...")

        # Проверяем API ключи в зависимости от провайдера
        if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            return "Ошибка: OPENAI_API_KEY не найден", "", ""

        if provider == "mlx_whisper" and not LOCAL_AVAILABLE:
            return "Ошибка: MLX-Whisper не установлен. pip install mlx-whisper", "", ""

        if not transcript_only and analysis_mode == "cloud" and not os.environ.get("ANTHROPIC_API_KEY"):
            return "Ошибка: ANTHROPIC_API_KEY не найден для облачного анализа", "", ""

        audio_path = audio_file.name if hasattr(audio_file, 'name') else audio_file

        # Получаем информацию о файле
        file_size_mb = Path(audio_path).stat().st_size / (1024 * 1024)
        file_ext = Path(audio_path).suffix.lower()

        progress(0.1, desc=f"Файл: {file_size_mb:.0f}MB ({file_ext})")

        # Конвертируем экзотические форматы (qta, avi, mov и т.д.) или большие файлы
        STANDARD_FORMATS = ['.mp3', '.wav', '.m4a', '.ogg', '.flac']
        needs_conversion = (
            file_ext not in STANDARD_FORMATS or  # Экзотический формат
            file_size_mb > 25  # Или слишком большой для API
        )

        if needs_conversion:
            progress(0.15, desc=f"Конвертирую {file_ext} ({file_size_mb:.0f}MB) в MP3...")
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

            # MLX-Whisper (локальный)
            if provider == "mlx_whisper":
                transcriber = LocalWhisperTranscriber(model="large-v3")
                transcript = transcriber.transcribe(audio_path)
            else:
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

            # MLX-Whisper (локальный)
            if provider == "mlx_whisper":
                transcriber = LocalWhisperTranscriber(model="large-v3")
                transcript = transcriber.transcribe(audio_path)
            else:
                transcriber = AudioTranscriber(provider=provider)
                transcript = transcriber.transcribe(audio_path)

            # Анализ транскрипта
            if analysis_mode == "local":
                model_display = analysis_model or "llama3.3"
                model = extract_model_name(model_display)
                progress(0.5, desc=f"Анализ через Ollama ({model})...")

                try:
                    client = OllamaClient()

                    # Подготавливаем текст для анализа (ограничиваем размер)
                    text_to_analyze = transcript['text'][:10000]

                    system_prompt = """Ты эксперт по анализу семинаров и лекций.
Проанализируй транскрипцию и создай структурированное резюме."""

                    analysis_prompt = f"""Проанализируй следующую транскрипцию семинара:

Метаданные:
- Название: {metadata.get('title', 'Не указано')}
- Спикер: {metadata.get('speaker', 'Не указан')}
- Дата: {metadata.get('date', 'Не указана')}

Транскрипция:
{text_to_analyze}

Создай структурированный анализ:
1. Краткое содержание (3-5 предложений)
2. Ключевые темы и концепции
3. Основные идеи спикера
4. Практические рекомендации (если есть)
5. Вопросы для рефлексии

Отвечай на русском языке."""

                    analysis_result = client.generate(
                        model=model,
                        prompt=analysis_prompt,
                        system=system_prompt
                    )

                    results = {
                        "metadata": metadata,
                        "transcript": transcript,
                        "chunk_results": [],
                        "final_summary": f"# Анализ семинара\n\n**Модель:** {model}\n\n{analysis_result}",
                        "processed_at": datetime.now().isoformat(),
                        "model": model
                    }
                except Exception as e:
                    logger.error(f"Ошибка локального анализа: {e}")
                    results = {
                        "metadata": metadata,
                        "transcript": transcript,
                        "chunk_results": [],
                        "final_summary": f"# Транскрипция\n\n⚠️ Ошибка анализа: {e}\n\n{transcript['text'][:5000]}...",
                        "processed_at": datetime.now().isoformat()
                    }
            else:
                progress(0.5, desc="Анализ через Claude...")
                processor = SeminarProcessor()
                results = processor.process_seminar(transcript, metadata)

            progress(0.9, desc="Сохранение...")

            # Сохранение результатов
            if analysis_mode == "local":
                # Сохраняем локально без SeminarProcessor
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                title_clean = "".join(c for c in metadata.get('title', 'audio') if c.isalnum() or c in " _-")[:30]
                output_name = f"{title_clean}_{timestamp}"

                json_path = OUTPUT_DIR / f"{output_name}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

                md_path = OUTPUT_DIR / f"{output_name}.md"
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(results.get("final_summary", ""))

                # Сохраняем транскрипт
                transcript_path = TRANSCRIPTS_DIR / f"{output_name}_transcript.txt"
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(transcript.get("text", ""))
            else:
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

def extract_model_name(display_name: str) -> str:
    """Извлекает имя модели из display_name (например, '⭐ qwen2.5vl:7b 🟢 ...' -> 'qwen2.5vl:7b')."""
    # Убираем ⭐ и [OCR] если есть
    name = display_name.replace("⭐ ", "").replace(" [OCR]", "")
    # Берём первую часть до 🟢/🟡/🔴
    for marker in ["🟢", "🟡", "🔴"]:
        if marker in name:
            name = name.split(marker)[0].strip()
            break
    return name.strip()


def extract_radio_value(display_value: str) -> str:
    """Извлекает значение из Radio choice (например, '☁️ OpenAI (openai)' -> 'openai')."""
    if "(" in display_value and ")" in display_value:
        # Извлекаем значение в скобках
        value = display_value.split("(")[1].split(")")[0].strip()
        return value
    # Если скобок нет, возвращаем как есть
    return display_value.strip()


def create_interface():
    """Создает Gradio интерфейс."""

    # Получаем модели Ollama один раз при создании интерфейса
    ollama_models = get_ollama_models()

    # Используем форматированные choices с рекомендациями (строки для Gradio 5.x)
    vision_choices = ollama_models["vision_choices"] if ollama_models["vision_choices"] else ["llava:34b 🟢 Хороший баланс"]
    text_choices = ollama_models["text_choices"] if ollama_models["text_choices"] else ["llama3.3 🟡 Медленная, но работает"]

    # Объединяем все модели (vision первые, потом текстовые)
    all_model_choices = vision_choices + text_choices

    # Для PDF и Audio используем одинаковый список
    pdf_model_choices = all_model_choices
    audio_model_choices = all_model_choices  # Одинаковый с PDF для консистентности

    # Значения по умолчанию (первый элемент списка)
    default_vision = vision_choices[0] if vision_choices else "llava:34b 🟢 Хороший баланс"
    default_text = text_choices[0] if text_choices else "llama3.3 🟡 Медленная, но работает"

    # RAM системы для отображения
    system_ram = ollama_models.get("system_ram", 0)

    # CSS для улучшения отображения Dropdown с прокруткой
    custom_css = """
    /* Фикс прокрутки для dropdown списков Ollama моделей */
    .dropdown-container ul {
        max-height: 300px !important;
        overflow-y: auto !important;
    }
    /* Улучшенный scrollbar */
    .dropdown-container ul::-webkit-scrollbar {
        width: 8px;
    }
    .dropdown-container ul::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    .dropdown-container ul::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    .dropdown-container ul::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    """

    with gr.Blocks(title="SKOLKOVO Materials Processor", css=custom_css) as app:

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
            with gr.Tab("📄 PDF Обработка"):
                gr.Markdown("### Обработка PDF через Claude Vision или локально (Ollama)")

                with gr.Row():
                    with gr.Column(scale=1):
                        pdf_file = gr.File(
                            label="PDF файл",
                            file_types=[".pdf"],
                            type="filepath"
                        )

                        # Переключатель облако/локально
                        pdf_processing_mode = gr.Radio(
                            label="🔧 Режим обработки",
                            choices=[
                                ("☁️ Облако (Claude API)", "cloud"),
                                ("🏠 Локально (Ollama)", "local")
                            ],
                            value="cloud",
                            info="Облако: лучшее качество, платно | Локально: бесплатно, требует Ollama"
                        )

                        # Выбор модели Ollama для PDF (vision)
                        with gr.Row():
                            pdf_ollama_model = gr.Dropdown(
                                label="🤖 Модель Ollama",
                                choices=pdf_model_choices,
                                value=default_vision,
                                visible=False,  # Скрыт по умолчанию (показывается при выборе local)
                                info=f"RAM: {system_ram}GB | ⭐ OCR | 🟢 Рекомендовано | 🟡 Медленная",
                                scale=4
                            )
                            pdf_refresh_models = gr.Button(
                                "🔄",
                                visible=False,
                                scale=1,
                                min_width=50
                            )

                        # Функция обновления списка моделей PDF (vision + text)
                        def refresh_pdf_models():
                            """Обновляет список моделей для PDF."""
                            models = get_ollama_models()
                            vision = models["vision_choices"] if models["vision_choices"] else ["llava:34b 🟢"]
                            text = models["text_choices"] if models["text_choices"] else ["llama3.3 🟡"]
                            pdf_choices = vision + text
                            default_v = vision[0] if vision else "llava:34b"
                            return gr.update(choices=pdf_choices, value=default_v)

                        # Показываем/скрываем выбор модели при смене режима
                        def toggle_model_visibility(mode):
                            visible = (mode == "local")
                            return gr.update(visible=visible), gr.update(visible=visible)

                        pdf_processing_mode.change(
                            fn=toggle_model_visibility,
                            inputs=[pdf_processing_mode],
                            outputs=[pdf_ollama_model, pdf_refresh_models]
                        )

                        pdf_mode = gr.Dropdown(
                            label="Тип анализа",
                            choices=[m[0] for m in get_pdf_modes()],
                            value="Полный анализ",
                            info="Выберите глубину анализа"
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
                    fn=lambda f, m, p, d, pm, om: process_pdf_file(f, get_mode_key(m), p, d, pm, om),
                    inputs=[pdf_file, pdf_mode, pdf_pages, pdf_dpi, pdf_processing_mode, pdf_ollama_model],
                    outputs=[pdf_status, pdf_result, pdf_json_path]
                )

                folder_btn.click(
                    fn=lambda p, m, d: process_pdf_folder(p, get_mode_key(m), d),
                    inputs=[folder_path, pdf_mode, pdf_dpi],
                    outputs=[folder_result]
                )

                # Обновление списка моделей PDF
                pdf_refresh_models.click(
                    fn=refresh_pdf_models,
                    outputs=[pdf_ollama_model]
                )

            # ========================
            # TAB 2: АУДИО ТРАНСКРИПЦИЯ
            # ========================
            with gr.Tab("🎙️ Аудио Транскрипция"):
                gr.Markdown("### Транскрипция аудио + анализ (облако или локально)")

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

                        # Провайдер транскрипции с MLX
                        audio_provider = gr.Radio(
                            label="🎤 Транскрипция",
                            choices=["☁️ OpenAI Whisper (openai)", "🏠 MLX-Whisper (mlx_whisper)"],
                            value="☁️ OpenAI Whisper (openai)",
                            info="OpenAI: платно, быстро | MLX-Whisper: бесплатно, локально на Mac"
                        )

                        # Режим анализа
                        audio_analysis_mode = gr.Radio(
                            label="📊 Анализ транскрипта",
                            choices=["☁️ Claude API (cloud)", "🏠 Ollama (local)"],
                            value="☁️ Claude API (cloud)",
                            info="Claude: глубокий анализ | Ollama: быстрый локальный анализ"
                        )

                        # Выбор модели Ollama для анализа аудио
                        with gr.Row():
                            audio_ollama_model = gr.Dropdown(
                                label="🤖 Модель Ollama (анализ)",
                                choices=audio_model_choices,
                                value=default_text,
                                visible=False,
                                info=f"RAM: {system_ram}GB | 🟢 Рекомендовано | 🟡 Медленная",
                                scale=4
                            )
                            audio_refresh_models = gr.Button(
                                "🔄",
                                visible=False,
                                scale=1,
                                min_width=50
                            )

                        # Показываем/скрываем выбор модели при смене режима анализа
                        def toggle_audio_model_visibility(mode):
                            mode_value = extract_radio_value(mode)
                            visible = (mode_value == "local")
                            return gr.update(visible=visible), gr.update(visible=visible)

                        audio_analysis_mode.change(
                            fn=toggle_audio_model_visibility,
                            inputs=[audio_analysis_mode],
                            outputs=[audio_ollama_model, audio_refresh_models]
                        )

                        audio_only = gr.Checkbox(
                            label="Только транскрипция (без анализа)",
                            value=False
                        )

                        with gr.Row():
                            audio_btn = gr.Button("🎤 Транскрибировать", variant="primary", scale=3)
                            audio_cancel_btn = gr.Button("❌ Отменить", variant="stop", scale=1)

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
                    inputs=[audio_file, audio_title, audio_speaker, audio_date, audio_module, audio_provider, audio_only, audio_analysis_mode, audio_ollama_model],
                    outputs=[audio_status, audio_result, audio_transcript]
                )

                # Кнопка отмены конвертации
                audio_cancel_btn.click(
                    fn=cancel_audio_conversion,
                    outputs=[audio_status]
                )

                # Функция обновления списка моделей Audio (все модели, как для PDF)
                def refresh_audio_models():
                    """Обновляет список моделей для анализа аудио (все модели, как для PDF)."""
                    models = get_ollama_models()
                    vision = models["vision_choices"] if models["vision_choices"] else []
                    text = models["text_choices"] if models["text_choices"] else ["llama3.3 🟡"]
                    all_models = vision + text  # Одинаковый список с PDF
                    default_m = all_models[0] if all_models else "llama3.3"
                    return gr.update(choices=all_models, value=default_m)

                # Обновление списка моделей Audio
                audio_refresh_models.click(
                    fn=refresh_audio_models,
                    outputs=[audio_ollama_model]
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
                        gr.Markdown("#### ☁️ Облачные API")
                        api_status = gr.Textbox(
                            label="Статус API ключей",
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

                        gr.Markdown("#### 🏠 Локальные компоненты")

                        def check_local_status():
                            if not LOCAL_AVAILABLE:
                                return "❌ Локальные процессоры не установлены"
                            try:
                                status = check_local_requirements()
                                lines = []
                                lines.append(f"Ollama: {'✅ Запущен' if status['ollama'] else '❌ Не запущен (ollama serve)'}")
                                if status['ollama'] and status['ollama_models']:
                                    all_models = status['ollama_models']

                                    # Разделяем модели
                                    vision_kw = ['llava', 'vision', 'bakllava', 'moondream', 'cogvlm', 'qwen-vl', 'qwen3-vl']
                                    vision_m = [m for m in all_models if any(kw in m.lower() for kw in vision_kw)]
                                    text_m = [m for m in all_models if m not in vision_m]

                                    lines.append(f"\n📝 Текстовые модели ({len(text_m)}):")
                                    for m in text_m[:8]:
                                        lines.append(f"   • {m}")
                                    if len(text_m) > 8:
                                        lines.append(f"   ... и ещё {len(text_m) - 8}")

                                    lines.append(f"\n🖼️ Vision модели ({len(vision_m)}):")
                                    if vision_m:
                                        for m in vision_m:
                                            lines.append(f"   • {m}")
                                    else:
                                        lines.append("   ⚠️ Нет (ollama pull llava:34b)")

                                lines.append(f"\nMLX-Whisper: {'✅ Установлен' if status['mlx_whisper'] else '❌ Не установлен'}")
                                lines.append(f"pdf2image: {'✅' if status['pdf2image'] else '❌'}")
                                return "\n".join(lines)
                            except Exception as e:
                                return f"Ошибка проверки: {e}"

                        local_status = gr.Textbox(
                            label="Статус локальных компонентов",
                            value=check_local_status() if LOCAL_AVAILABLE else "Локальные процессоры не загружены",
                            lines=6,
                            interactive=False
                        )

                        check_local_btn = gr.Button("🔄 Проверить локальные")
                        check_local_btn.click(
                            fn=check_local_status,
                            outputs=[local_status]
                        )

                        gr.Markdown("""
                        **Установка локальных компонентов:**
                        ```bash
                        # Ollama
                        brew install ollama
                        ollama serve  # в отдельном терминале
                        ollama pull llama3.3      # текстовая модель
                        ollama pull llava:34b     # vision модель

                        # MLX-Whisper (транскрипция)
                        pip install mlx-whisper
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
