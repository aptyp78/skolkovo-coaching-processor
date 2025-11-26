#!/usr/bin/env python3
"""
Local Processor - Локальная обработка через Ollama и MLX-Whisper
================================================================
Полностью локальная обработка PDF и аудио без использования облачных API.

Требования:
- Ollama с моделями llama3.3 (текст) и llava:34b (vision)
- mlx-whisper для транскрипции

Использование:
    # PDF через локальную LLM
    python local_processor.py pdf файл.pdf --mode full_analysis

    # PDF Vision через локальную LLM
    python local_processor.py vision скан.pdf --mode extract_text

    # Аудио через MLX-Whisper
    python local_processor.py audio запись.mp3
"""

import os
import sys
import json
import base64
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from io import BytesIO
import argparse
import requests

# Импортируем утилиты
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    get_logger, load_config, load_env_file,
    OUTPUT_DIR, TRANSCRIPTS_DIR, ensure_directories
)

# PDF processing
try:
    import pdfplumber
    from pypdf import PdfReader
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# PDF to image
try:
    from pdf2image import convert_from_path
    from PIL import Image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

# Инициализация
load_env_file()
ensure_directories()
logger = get_logger(__name__)

# Конфигурация по умолчанию
DEFAULT_CONFIG = {
    "ollama_base_url": "http://localhost:11434",
    "text_model": "llama3.3",           # Для анализа текста
    "vision_model": "llava:34b",         # Для анализа изображений
    "whisper_model": "large-v3",         # MLX-Whisper модель
    "max_tokens_per_chunk": 4000,
    "dpi": 150,
    "language": "ru"
}


class OllamaClient:
    """Клиент для работы с Ollama API."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self._check_connection()

    def _check_connection(self):
        """Проверяет подключение к Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Не удалось подключиться к Ollama. "
                "Убедитесь, что Ollama запущен: ollama serve"
            )

    def list_models(self) -> List[str]:
        """Возвращает список доступных моделей."""
        response = requests.get(f"{self.base_url}/api/tags")
        data = response.json()
        return [m["name"] for m in data.get("models", [])]

    def generate(self, model: str, prompt: str, system: str = None,
                 images: List[str] = None, stream: bool = False) -> str:
        """
        Генерирует ответ от модели.

        Args:
            model: Имя модели
            prompt: Промпт пользователя
            system: Системный промпт
            images: Список base64 изображений (для vision моделей)
            stream: Стриминг ответа
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "num_predict": 4096,
                "temperature": 0.7
            }
        }

        if system:
            payload["system"] = system

        if images:
            payload["images"] = images

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=300  # 5 минут таймаут
        )
        response.raise_for_status()

        return response.json()["response"]

    def chat(self, model: str, messages: List[Dict],
             images: List[str] = None) -> str:
        """
        Chat completion через Ollama.

        Args:
            model: Имя модели
            messages: Список сообщений [{"role": "user/assistant", "content": "..."}]
            images: Список base64 изображений
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": 4096,
                "temperature": 0.7
            }
        }

        # Для vision моделей добавляем изображения в последнее сообщение
        if images and messages:
            messages[-1]["images"] = images

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=300
        )
        response.raise_for_status()

        return response.json()["message"]["content"]


class LocalPDFProcessor:
    """Локальная обработка PDF через Ollama."""

    def __init__(self, config: dict = None):
        """Инициализация процессора."""
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.client = OllamaClient(self.config["ollama_base_url"])
        self.output_dir = OUTPUT_DIR

        # Проверяем доступность модели
        available_models = self.client.list_models()
        if self.config["text_model"] not in available_models:
            logger.warning(
                f"Модель {self.config['text_model']} не найдена. "
                f"Доступные: {available_models}"
            )

        logger.info(f"LocalPDFProcessor инициализирован, модель: {self.config['text_model']}")

    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """Извлекает текст из PDF."""
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError("Установите: pip install pdfplumber pypdf")

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF не найден: {pdf_path}")

        metadata = {
            "filename": pdf_path.name,
            "path": str(pdf_path),
            "extracted_at": datetime.now().isoformat(),
            "pages": 0,
            "characters": 0
        }

        full_text = []

        print(f"📄 Извлекаю текст из: {pdf_path.name}")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                metadata["pages"] = len(pdf.pages)

                for i, page in enumerate(pdf.pages, 1):
                    print(f"  Страница {i}/{metadata['pages']}...", end="\r")

                    text = page.extract_text() or ""
                    tables = page.extract_tables()

                    page_content = f"\n\n--- СТРАНИЦА {i} ---\n\n{text}"

                    if tables:
                        page_content += "\n\n[ТАБЛИЦЫ]\n"
                        for j, table in enumerate(tables, 1):
                            page_content += f"\nТаблица {j}:\n"
                            for row in table:
                                if row:
                                    page_content += " | ".join(str(cell) if cell else "" for cell in row) + "\n"

                    full_text.append(page_content)

        except Exception as e:
            # Fallback to pypdf
            reader = PdfReader(pdf_path)
            metadata["pages"] = len(reader.pages)
            for i, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                full_text.append(f"\n\n--- СТРАНИЦА {i} ---\n\n{text}")

        result_text = "\n".join(full_text)
        metadata["characters"] = len(result_text)

        print(f"\n✅ Извлечено {metadata['pages']} страниц, {metadata['characters']:,} символов")

        return result_text, metadata

    def smart_chunk_text(self, text: str, max_chars: int = None) -> List[Dict]:
        """Разбивает текст на чанки."""
        if max_chars is None:
            max_chars = self.config["max_tokens_per_chunk"] * 4

        import re

        chunks = []
        separators = [
            r'\n\n--- СТРАНИЦА \d+ ---\n\n',
            r'\n\n#{1,3} ',
            r'\n\n',
            r'\n',
            r'\. ',
        ]

        current_pos = 0
        chunk_num = 0

        while current_pos < len(text):
            chunk_end = min(current_pos + max_chars, len(text))

            if chunk_end < len(text):
                search_start = max(current_pos + max_chars // 2, current_pos)
                search_text = text[search_start:chunk_end]

                best_break = None
                for sep in separators:
                    matches = list(re.finditer(sep, search_text))
                    if matches:
                        best_break = search_start + matches[-1].end()
                        break

                if best_break:
                    chunk_end = best_break

            chunk_text = text[current_pos:chunk_end].strip()

            if chunk_text:
                chunk_num += 1
                page_matches = re.findall(r'--- СТРАНИЦА (\d+) ---', chunk_text)
                pages = [int(p) for p in page_matches] if page_matches else []

                chunks.append({
                    "chunk_id": chunk_num,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                    "pages": pages,
                    "page_range": f"{min(pages)}-{max(pages)}" if pages else "N/A"
                })

            current_pos = chunk_end

        print(f"📦 Создано {len(chunks)} чанков")
        return chunks

    def get_system_prompt(self, mode: str) -> str:
        """Создает системный промпт."""
        base = """Ты - AI-ассистент для программы Executive Coaching в СКОЛКОВО.
Твоя задача - обработать учебный материал и извлечь ценную информацию.

КОНТЕКСТ ПРОГРАММЫ:
- Программа по развитию коучинговых компетенций
- Фокус на executive-коучинг
- Развитие эмоционального интеллекта и лидерства

ПРИНЦИПЫ:
1. Сохраняй оригинальную терминологию
2. Выделяй практические инструменты
3. Структурируй информацию

"""

        mode_prompts = {
            "summary": """ЗАДАЧА: Создай структурированное саммари

## Основные темы
## Ключевые идеи
## Важные цитаты/определения
""",
            "key_concepts": """ЗАДАЧА: Извлеки ключевые концепции

## Концепции
| Название | Определение | Применение |

## Модели и фреймворки
## Инструменты
""",
            "coaching_tools": """ЗАДАЧА: Извлеки инструменты для коучинга

## Техники и упражнения
## Мощные вопросы
## Практические рекомендации
""",
            "full_analysis": """ЗАДАЧА: Полный анализ материала

## 1. САММАРИ
## 2. КЛЮЧЕВЫЕ КОНЦЕПЦИИ
## 3. ИНСТРУМЕНТЫ И ТЕХНИКИ
## 4. МОЩНЫЕ ВОПРОСЫ
## 5. ТЕГИ ДЛЯ ПОИСКА
"""
        }

        return base + mode_prompts.get(mode, mode_prompts["summary"])

    def process_chunk(self, chunk: Dict, mode: str = "full_analysis") -> Dict:
        """Обрабатывает один чанк через Ollama."""
        system_prompt = self.get_system_prompt(mode)

        user_message = f"""Обработай фрагмент учебного материала:

---
{chunk['text']}
---

Страницы: {chunk.get('page_range', 'N/A')}
"""

        try:
            print(f"   🔄 Отправляю в {self.config['text_model']}...")

            response = self.client.generate(
                model=self.config["text_model"],
                prompt=user_message,
                system=system_prompt
            )

            return {
                "chunk_id": chunk["chunk_id"],
                "page_range": chunk.get("page_range", "N/A"),
                "mode": mode,
                "processed_at": datetime.now().isoformat(),
                "response": response,
                "status": "success",
                "model": self.config["text_model"]
            }

        except Exception as e:
            return {
                "chunk_id": chunk["chunk_id"],
                "page_range": chunk.get("page_range", "N/A"),
                "mode": mode,
                "processed_at": datetime.now().isoformat(),
                "error": str(e),
                "status": "error"
            }

    def process_pdf(self, pdf_path: str, mode: str = "full_analysis") -> Dict:
        """Полный пайплайн обработки PDF."""
        pdf_path = Path(pdf_path)

        pdf_hash = hashlib.md5(pdf_path.name.encode()).hexdigest()[:8]
        session_id = f"{pdf_path.stem}_local_{pdf_hash}"

        print(f"""
╔══════════════════════════════════════════════════════════════╗
║     🏠 Local PDF Processor (Ollama)                          ║
║     Модель: {self.config['text_model']:<44}║
╚══════════════════════════════════════════════════════════════╝
""")

        # Извлекаем текст
        text, metadata = self.extract_text_from_pdf(pdf_path)

        # Разбиваем на чанки
        chunks = self.smart_chunk_text(text)

        # Обрабатываем
        results = []
        total = len(chunks)

        print(f"\n🚀 Обрабатываю {total} чанков в режиме '{mode}'")
        print("=" * 50)

        for i, chunk in enumerate(chunks, 1):
            print(f"\n📝 Чанк {i}/{total} (стр. {chunk.get('page_range', 'N/A')})...")
            result = self.process_chunk(chunk, mode)
            results.append(result)

            if result["status"] == "success":
                print(f"   ✅ Готово")
            else:
                print(f"   ❌ Ошибка: {result.get('error')}")

        # Сохраняем результаты
        final_result = {
            "session_id": session_id,
            "source_pdf": str(pdf_path),
            "metadata": metadata,
            "mode": mode,
            "processor": "local_ollama",
            "model": self.config["text_model"],
            "total_chunks": total,
            "successful_chunks": sum(1 for r in results if r["status"] == "success"),
            "processed_at": datetime.now().isoformat(),
            "results": results
        }

        output_json = self.output_dir / f"{session_id}_processed.json"
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)

        output_md = self.output_dir / f"{session_id}_processed.md"
        self._save_as_markdown(final_result, output_md)

        print("\n" + "=" * 50)
        print(f"✅ Обработка завершена!")
        print(f"📊 Чанков: {final_result['successful_chunks']}/{total}")
        print(f"📁 Результаты: {output_md}")

        return final_result

    def _save_as_markdown(self, result: Dict, output_path: Path):
        """Сохраняет результат в Markdown."""
        md = f"""# 🏠 Local Analysis: {Path(result['source_pdf']).name}

**Модель:** {result.get('model', 'N/A')}
**Дата:** {result['processed_at']}
**Режим:** {result['mode']}
**Страниц:** {result['metadata'].get('pages', 'N/A')}

---

"""
        for r in result["results"]:
            if r["status"] == "success":
                md += f"""
## Фрагмент {r['chunk_id']} (стр. {r['page_range']})

{r['response']}

---
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)


class LocalVisionProcessor:
    """Локальная обработка PDF через Ollama Vision."""

    def __init__(self, config: dict = None):
        """Инициализация процессора."""
        if not PDF2IMAGE_AVAILABLE:
            raise ImportError("Установите: pip install pdf2image pillow")

        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.client = OllamaClient(self.config["ollama_base_url"])
        self.output_dir = OUTPUT_DIR

        # Проверяем доступность vision модели
        available_models = self.client.list_models()
        if self.config["vision_model"] not in available_models:
            # Пробуем альтернативы
            vision_alternatives = ["llava:34b", "llava:13b", "llava:7b", "llava"]
            for alt in vision_alternatives:
                if alt in available_models:
                    self.config["vision_model"] = alt
                    break
            else:
                logger.warning(
                    f"Vision модель не найдена. Доступные: {available_models}. "
                    "Установите: ollama pull llava:34b"
                )

        logger.info(f"LocalVisionProcessor инициализирован, модель: {self.config['vision_model']}")

    def pdf_to_images(self, pdf_path: str, dpi: int = None) -> List[Image.Image]:
        """Конвертирует PDF в изображения."""
        dpi = dpi or self.config["dpi"]
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF не найден: {pdf_path}")

        print(f"📄 Конвертирую PDF: {pdf_path.name} (DPI: {dpi})")

        images = convert_from_path(str(pdf_path), dpi=dpi)

        print(f"   ✅ Создано {len(images)} страниц")

        return images

    def image_to_base64(self, image: Image.Image, max_size: int = 1024) -> str:
        """Конвертирует изображение в base64."""
        # Ресайзим для Ollama
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

    def get_system_prompt(self, mode: str) -> str:
        """Создает системный промпт для vision/OCR."""
        # Определяем, это OCR-оптимизированная модель или описательная
        model_name = self.config.get('vision_model', '').lower()
        is_ocr_model = any(kw in model_name for kw in [
            'granite', 'qwen2.5vl', 'qwen2-vl', 'deepseek-ocr', 'minicpm'
        ])

        if is_ocr_model:
            # Для OCR-моделей: прямая команда на извлечение текста
            base = """Ты — OCR-система. Твоя единственная задача — ДОСЛОВНО извлечь весь текст с изображения.

КРИТИЧЕСКИ ВАЖНО:
- НЕ описывай изображение
- НЕ интерпретируй содержимое
- ТОЛЬКО извлекай и копируй текст КАК ЕСТЬ
- Сохраняй оригинальный язык текста (русский, английский и т.д.)
- Используй markdown для форматирования (заголовки, списки, таблицы)

"""
        else:
            # Для llava и подобных: более подробные инструкции
            base = """Ты - AI-ассистент для программы Executive Coaching в СКОЛКОВО.
Твоя задача — ИЗВЛЕЧЬ ТЕКСТ с изображения, а НЕ описывать картинку.

ВАЖНО:
1. Извлекай ВЕСЬ видимый ТЕКСТ дословно
2. НЕ описывай что ты видишь ("на изображении показано...")
3. ПРОСТО копируй текст как он написан
4. Таблицы форматируй в markdown
5. Сохраняй язык оригинала

"""

        mode_prompts = {
            "extract_text": """Извлеки ВЕСЬ текст с этой страницы.
Формат:
- Заголовки как # Заголовок
- Списки как - пункт
- Таблицы в markdown формате
""",
            "full_analysis": """Извлеки текст и проанализируй:

## ТЕКСТ ДОКУМЕНТА
(здесь весь извлечённый текст)

## КЛЮЧЕВЫЕ ИДЕИ
(главные мысли из текста)

## ТЕГИ
(ключевые слова через запятую)
""",
            "summary": """Извлеки текст и создай саммари:

## ТЕКСТ
(извлечённый текст)

## КРАТКОЕ СОДЕРЖАНИЕ
(основные пункты)
""",
            "key_concepts": """Извлеки текст и выдели концепции:

## ТЕКСТ
(извлечённый текст)

## КОНЦЕПЦИИ
| Термин | Определение |
"""
        }

        return base + mode_prompts.get(mode, mode_prompts["extract_text"])

    def process_page(self, image: Image.Image, page_num: int, mode: str) -> Dict:
        """Обрабатывает одну страницу."""
        system_prompt = self.get_system_prompt(mode)

        try:
            img_base64 = self.image_to_base64(image)

            print(f"   🔄 Отправляю в {self.config['vision_model']}...")

            response = self.client.generate(
                model=self.config["vision_model"],
                prompt=f"Проанализируй страницу {page_num} документа:\n\n{system_prompt}",
                images=[img_base64]
            )

            return {
                "page": page_num,
                "mode": mode,
                "processed_at": datetime.now().isoformat(),
                "response": response,
                "status": "success",
                "model": self.config["vision_model"]
            }

        except Exception as e:
            return {
                "page": page_num,
                "mode": mode,
                "processed_at": datetime.now().isoformat(),
                "error": str(e),
                "status": "error"
            }

    def process_pdf(self, pdf_path: str, mode: str = "full_analysis",
                    page_range: Tuple[int, int] = None) -> Dict:
        """Полный пайплайн обработки PDF через Vision."""
        pdf_path = Path(pdf_path)

        pdf_hash = hashlib.md5(pdf_path.name.encode()).hexdigest()[:8]
        session_id = f"{pdf_path.stem}_local_vision_{pdf_hash}"

        print(f"""
╔══════════════════════════════════════════════════════════════╗
║     👁️ Local Vision Processor (Ollama)                       ║
║     Модель: {self.config['vision_model']:<44}║
╚══════════════════════════════════════════════════════════════╝
""")

        # Конвертируем PDF
        images = self.pdf_to_images(pdf_path)

        # Применяем диапазон
        if page_range:
            start, end = page_range
            images = images[start-1:end]
            page_start = start
        else:
            page_start = 1

        total = len(images)

        print(f"\n🚀 Обрабатываю {total} страниц в режиме '{mode}'")
        print("=" * 50)

        results = []

        for i, image in enumerate(images):
            page_num = page_start + i
            print(f"\n📝 Страница {page_num}/{page_start + total - 1}...")

            result = self.process_page(image, page_num, mode)
            results.append(result)

            if result["status"] == "success":
                print(f"   ✅ Готово")
            else:
                print(f"   ❌ Ошибка: {result.get('error')}")

        # Сохраняем результаты
        final_result = {
            "session_id": session_id,
            "source_pdf": str(pdf_path),
            "mode": mode,
            "processor": "local_ollama_vision",
            "model": self.config["vision_model"],
            "total_pages": total,
            "successful_pages": sum(1 for r in results if r["status"] == "success"),
            "processed_at": datetime.now().isoformat(),
            "results": results
        }

        output_json = self.output_dir / f"{session_id}_processed.json"
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)

        output_md = self.output_dir / f"{session_id}_processed.md"
        self._save_as_markdown(final_result, output_md)

        print("\n" + "=" * 50)
        print(f"✅ Обработка завершена!")
        print(f"📊 Страниц: {final_result['successful_pages']}/{total}")
        print(f"📁 Результаты: {output_md}")

        return final_result

    def _save_as_markdown(self, result: Dict, output_path: Path):
        """Сохраняет в Markdown."""
        md = f"""# 👁️ Local Vision: {Path(result['source_pdf']).name}

**Модель:** {result.get('model', 'N/A')}
**Дата:** {result['processed_at']}
**Режим:** {result['mode']}
**Страниц:** {result['total_pages']}

---

"""
        for r in result["results"]:
            if r["status"] == "success":
                md += f"""
## Страница {r['page']}

{r['response']}

---
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)


class LocalWhisperTranscriber:
    """Локальная транскрипция через MLX-Whisper."""

    def __init__(self, model: str = "large-v3"):
        """
        Args:
            model: Модель whisper (tiny, base, small, medium, large-v3)
        """
        self.model = model
        self.output_dir = OUTPUT_DIR
        self.transcripts_dir = TRANSCRIPTS_DIR

        # Проверяем наличие mlx_whisper
        try:
            import mlx_whisper
            self.mlx_whisper = mlx_whisper
        except ImportError:
            raise ImportError("Установите: pip install mlx-whisper")

        logger.info(f"LocalWhisperTranscriber инициализирован, модель: {model}")

    def transcribe(self, audio_path: str, language: str = "ru") -> Dict:
        """
        Транскрибирует аудиофайл локально через MLX-Whisper.

        Args:
            audio_path: Путь к аудиофайлу
            language: Язык (ru, en, etc.)

        Returns:
            Dict с текстом и сегментами
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Файл не найден: {audio_path}")

        print(f"""
╔══════════════════════════════════════════════════════════════╗
║     🎙️ Local Whisper Transcriber (MLX)                       ║
║     Модель: {self.model:<46}║
╚══════════════════════════════════════════════════════════════╝
""")
        print(f"📂 Файл: {audio_path.name}")
        print(f"🔄 Транскрибирую (это может занять несколько минут)...")

        # Транскрибируем через MLX-Whisper
        result = self.mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=f"mlx-community/whisper-{self.model}-mlx",
            language=language,
            verbose=False
        )

        # Формируем результат
        text = result.get("text", "")
        segments = []

        for seg in result.get("segments", []):
            segments.append({
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", "")
            })

        # Вычисляем длительность
        duration = segments[-1]["end"] if segments else 0

        output = {
            "text": text,
            "segments": segments,
            "duration": duration,
            "provider": "mlx_whisper",
            "model": self.model,
            "language": language
        }

        print(f"\n✅ Транскрипция завершена!")
        print(f"   📝 {len(text):,} символов")
        print(f"   ⏱️ {duration / 60:.1f} минут")

        return output

    def transcribe_and_save(self, audio_path: str, language: str = "ru") -> Tuple[Dict, Path]:
        """Транскрибирует и сохраняет результат."""
        result = self.transcribe(audio_path, language)

        # Сохраняем транскрипт
        audio_name = Path(audio_path).stem
        transcript_path = self.transcripts_dir / f"{audio_name}_local_transcript.txt"

        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(result["text"])

        print(f"   📁 Сохранено: {transcript_path}")

        return result, transcript_path


def check_local_requirements() -> Dict[str, bool]:
    """Проверяет доступность локальных компонентов."""
    status = {
        "ollama": False,
        "ollama_models": [],
        "mlx_whisper": False,
        "pdf2image": PDF2IMAGE_AVAILABLE,
        "pdfplumber": PDFPLUMBER_AVAILABLE
    }

    # Проверяем Ollama
    try:
        client = OllamaClient()
        status["ollama"] = True
        status["ollama_models"] = client.list_models()
    except:
        pass

    # Проверяем MLX-Whisper
    try:
        import mlx_whisper
        status["mlx_whisper"] = True
    except:
        pass

    return status


def main():
    """CLI интерфейс."""
    parser = argparse.ArgumentParser(
        description="Local Processor - локальная обработка через Ollama и MLX-Whisper"
    )

    subparsers = parser.add_subparsers(dest="command", help="Команды")

    # Команда pdf
    pdf_parser = subparsers.add_parser("pdf", help="Обработать PDF через Ollama (текст)")
    pdf_parser.add_argument("pdf_path", help="Путь к PDF")
    pdf_parser.add_argument("--mode", "-m", default="full_analysis",
                           choices=["summary", "key_concepts", "coaching_tools", "full_analysis"])
    pdf_parser.add_argument("--model", default="llama3.3", help="Модель Ollama")

    # Команда vision
    vision_parser = subparsers.add_parser("vision", help="Обработать PDF через Ollama Vision")
    vision_parser.add_argument("pdf_path", help="Путь к PDF")
    vision_parser.add_argument("--mode", "-m", default="full_analysis",
                              choices=["extract_text", "full_analysis"])
    vision_parser.add_argument("--pages", "-p", help="Диапазон страниц (1-10)")
    vision_parser.add_argument("--model", default="llava:34b", help="Vision модель")

    # Команда audio
    audio_parser = subparsers.add_parser("audio", help="Транскрибировать аудио через MLX-Whisper")
    audio_parser.add_argument("audio_path", help="Путь к аудиофайлу")
    audio_parser.add_argument("--model", default="large-v3",
                             choices=["tiny", "base", "small", "medium", "large-v3"])
    audio_parser.add_argument("--language", "-l", default="ru")

    # Команда check
    check_parser = subparsers.add_parser("check", help="Проверить локальные компоненты")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "check":
            status = check_local_requirements()
            print("\n🔍 Статус локальных компонентов:\n")
            print(f"  Ollama: {'✅' if status['ollama'] else '❌'}")
            if status['ollama']:
                print(f"    Модели: {', '.join(status['ollama_models'][:5])}...")
            print(f"  MLX-Whisper: {'✅' if status['mlx_whisper'] else '❌'}")
            print(f"  pdf2image: {'✅' if status['pdf2image'] else '❌'}")
            print(f"  pdfplumber: {'✅' if status['pdfplumber'] else '❌'}")

        elif args.command == "pdf":
            processor = LocalPDFProcessor({"text_model": args.model})
            processor.process_pdf(args.pdf_path, mode=args.mode)

        elif args.command == "vision":
            page_range = None
            if args.pages:
                parts = args.pages.split("-")
                page_range = (int(parts[0]), int(parts[1]))

            processor = LocalVisionProcessor({"vision_model": args.model})
            processor.process_pdf(args.pdf_path, mode=args.mode, page_range=page_range)

        elif args.command == "audio":
            transcriber = LocalWhisperTranscriber(model=args.model)
            transcriber.transcribe_and_save(args.audio_path, language=args.language)

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
