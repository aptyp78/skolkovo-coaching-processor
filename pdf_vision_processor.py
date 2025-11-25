#!/usr/bin/env python3
"""
PDF Vision Processor для SKOLKOVO Executive Coaching Program
=============================================================
Универсальный парсинг PDF через Claude Vision API.
Конвертирует страницы PDF в изображения и анализирует через Claude.

Преимущества:
- Работает с любым PDF (сканы, изображения, графики, таблицы)
- Единый механизм для всех типов контента
- Высокое качество распознавания

Использование:
    python pdf_vision_processor.py process файл.pdf
    python pdf_vision_processor.py process файл.pdf --mode full_analysis
"""

import os
import sys
import json
import base64
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import argparse
from io import BytesIO

# PDF to image conversion
try:
    from pdf2image import convert_from_path
    from PIL import Image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("⚠️ Установите: pip install pdf2image pillow")

# Claude API
try:
    import anthropic
except ImportError:
    print("Установите anthropic: pip install anthropic")
    sys.exit(1)


# Конфигурация
DEFAULT_CONFIG = {
    "model": "claude-sonnet-4-20250514",
    "max_output_tokens": 4096,
    "language": "ru",
    "dpi": 150,  # Разрешение для конвертации PDF -> изображение
    "max_pages_per_request": 4,  # Максимум страниц в одном запросе к Claude
    "processing_modes": {
        "extract_text": "Извлеки весь текст со страницы, сохраняя структуру",
        "summary": "Создай структурированное саммари",
        "key_concepts": "Извлеки ключевые концепции и модели",
        "coaching_tools": "Извлеки инструменты и техники для коучинга",
        "questions": "Сгенерируй вопросы для рефлексии",
        "full_analysis": "Полный анализ: текст + концепции + инструменты + вопросы"
    }
}


class PDFVisionProcessor:
    """Обработка PDF через Claude Vision API."""

    def __init__(self, api_key: str = None, config: dict = None):
        """Инициализация процессора."""
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API ключ не найден. Установите ANTHROPIC_API_KEY")

        if not PDF2IMAGE_AVAILABLE:
            raise ImportError("Установите pdf2image: pip install pdf2image pillow")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.config = {**DEFAULT_CONFIG, **(config or {})}

        # Пути для сохранения
        self.base_dir = Path(__file__).parent
        self.output_dir = self.base_dir / "output"
        self.output_dir.mkdir(exist_ok=True)

    def pdf_to_images(self, pdf_path: str, dpi: int = None) -> List[Image.Image]:
        """
        Конвертирует PDF в список изображений.

        Args:
            pdf_path: Путь к PDF файлу
            dpi: Разрешение (по умолчанию из конфига)

        Returns:
            List[Image.Image]: Список PIL изображений
        """
        dpi = dpi or self.config["dpi"]
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF не найден: {pdf_path}")

        print(f"📄 Конвертирую PDF в изображения: {pdf_path.name}")
        print(f"   DPI: {dpi}")

        images = convert_from_path(str(pdf_path), dpi=dpi)

        print(f"   ✅ Создано {len(images)} страниц")

        return images

    def image_to_base64(self, image: Image.Image, format: str = "PNG", max_size: int = 1568) -> str:
        """
        Конвертирует PIL Image в base64 строку.

        Args:
            image: PIL Image
            format: Формат изображения
            max_size: Максимальный размер стороны (Claude ограничение)

        Returns:
            str: Base64 encoded string
        """
        # Ресайзим если нужно (Claude лимит ~1568px для оптимальной работы)
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        buffer = BytesIO()
        image.save(buffer, format=format)
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

    def get_system_prompt(self, mode: str) -> str:
        """Создает системный промпт для обработки."""

        base_prompt = """Ты - AI-ассистент для программы Executive Coaching & Mentoring в СКОЛКОВО.
Анализируй изображения страниц PDF документа и извлекай информацию.

ВАЖНО:
1. Извлекай ВСЕ видимые элементы: текст, таблицы, схемы, графики
2. Описывай диаграммы и изображения текстом
3. Сохраняй структуру и форматирование документа
4. Указывай, если что-то нечитаемо или неясно

КОНТЕКСТ ПРОГРАММЫ:
- Executive-коучинг и развитие лидерства
- Психологические концепции
- Инструменты и техники коучинга

"""

        mode_prompts = {
            "extract_text": """ЗАДАЧА: Извлеки весь текст со страницы

ФОРМАТ ОТВЕТА:
- Сохраняй оригинальную структуру (заголовки, списки, абзацы)
- Таблицы представляй в markdown формате
- Описывай графики и схемы текстом [ИЗОБРАЖЕНИЕ: описание]
- Сохраняй нумерацию и маркировку
""",

            "summary": """ЗАДАЧА: Создай структурированное саммари

ФОРМАТ ОТВЕТА:
## Основные темы
[3-5 ключевых тем]

## Ключевые идеи
[Главные идеи документа]

## Важные определения
[Термины и их значения]
""",

            "key_concepts": """ЗАДАЧА: Извлеки ключевые концепции

ФОРМАТ ОТВЕТА:
## Концепции
| Название | Определение | Автор (если есть) |
|----------|-------------|-------------------|

## Модели и фреймворки
[Описание моделей с компонентами]

## Связи между концепциями
[Как концепции связаны друг с другом]
""",

            "coaching_tools": """ЗАДАЧА: Извлеки инструменты для коучинга

ФОРМАТ ОТВЕТА:
## Техники и упражнения
- **Название**: Пошаговое описание

## Мощные вопросы
[Вопросы из материала]

## Практические рекомендации
[Как применять на практике]
""",

            "questions": """ЗАДАЧА: Сгенерируй вопросы по материалу

ФОРМАТ ОТВЕТА:
## Вопросы для понимания
[5 вопросов на проверку]

## Вопросы для рефлексии
[5 глубоких вопросов]

## Вопросы для практики
[5 вопросов о применении]
""",

            "full_analysis": """ЗАДАЧА: Полный анализ материала

СТРУКТУРА ОТВЕТА:

## 1. ИЗВЛЕЧЕННЫЙ ТЕКСТ
[Весь текст со страницы с сохранением структуры]

## 2. ВИЗУАЛЬНЫЕ ЭЛЕМЕНТЫ
[Описание графиков, схем, изображений]

## 3. КЛЮЧЕВЫЕ КОНЦЕПЦИИ
| Концепция | Определение | Применение |
|-----------|-------------|------------|

## 4. ИНСТРУМЕНТЫ И ТЕХНИКИ
[Практические инструменты]

## 5. ВАЖНЫЕ ЦИТАТЫ
[Ключевые формулировки]

## 6. ТЕГИ
[5-10 ключевых слов для поиска]
"""
        }

        return base_prompt + mode_prompts.get(mode, mode_prompts["extract_text"])

    def process_pages(self, images: List[Image.Image], mode: str = "full_analysis",
                      page_start: int = 1) -> List[Dict]:
        """
        Обрабатывает страницы через Claude Vision.

        Args:
            images: Список PIL изображений
            mode: Режим обработки
            page_start: Номер первой страницы

        Returns:
            List[Dict]: Результаты обработки
        """
        results = []
        max_pages = self.config["max_pages_per_request"]
        system_prompt = self.get_system_prompt(mode)

        # Группируем страницы для обработки
        for i in range(0, len(images), max_pages):
            batch = images[i:i + max_pages]
            page_nums = list(range(page_start + i, page_start + i + len(batch)))

            print(f"\n📝 Обрабатываю страницы {page_nums[0]}-{page_nums[-1]}...")

            # Создаем контент с изображениями
            content = []

            # Добавляем текст с номерами страниц
            content.append({
                "type": "text",
                "text": f"Анализируй страницы {page_nums[0]}-{page_nums[-1]} документа:"
            })

            # Добавляем изображения
            for j, img in enumerate(batch):
                img_base64 = self.image_to_base64(img)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_base64
                    }
                })
                content.append({
                    "type": "text",
                    "text": f"[Страница {page_nums[j]}]"
                })

            try:
                response = self.client.messages.create(
                    model=self.config["model"],
                    max_tokens=self.config["max_output_tokens"],
                    system=system_prompt,
                    messages=[{"role": "user", "content": content}]
                )

                result = {
                    "pages": page_nums,
                    "page_range": f"{page_nums[0]}-{page_nums[-1]}",
                    "mode": mode,
                    "processed_at": datetime.now().isoformat(),
                    "response": response.content[0].text,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens
                    },
                    "status": "success"
                }

                print(f"   ✅ Готово ({result['usage']['input_tokens']}→{result['usage']['output_tokens']} токенов)")

            except Exception as e:
                result = {
                    "pages": page_nums,
                    "page_range": f"{page_nums[0]}-{page_nums[-1]}",
                    "mode": mode,
                    "processed_at": datetime.now().isoformat(),
                    "error": str(e),
                    "status": "error"
                }
                print(f"   ❌ Ошибка: {e}")

            results.append(result)

        return results

    def process_pdf(self, pdf_path: str, mode: str = "full_analysis",
                    page_range: Tuple[int, int] = None) -> Dict:
        """
        Полный пайплайн обработки PDF через Vision.

        Args:
            pdf_path: Путь к PDF
            mode: Режим обработки
            page_range: Диапазон страниц (start, end), None = все

        Returns:
            Dict: Результат обработки
        """
        pdf_path = Path(pdf_path)

        # Создаем ID сессии
        pdf_hash = hashlib.md5(pdf_path.name.encode()).hexdigest()[:8]
        session_id = f"{pdf_path.stem}_vision_{pdf_hash}"

        print(f"""
╔══════════════════════════════════════════════════════════════╗
║     👁️ PDF Vision Processor                                   ║
║     Claude Vision для любых PDF                               ║
╚══════════════════════════════════════════════════════════════╝
""")

        # Конвертируем PDF в изображения
        images = self.pdf_to_images(pdf_path)

        # Выбираем диапазон страниц
        if page_range:
            start, end = page_range
            images = images[start-1:end]
            page_start = start
        else:
            page_start = 1

        total_pages = len(images)

        print(f"\n🚀 Начинаю обработку {total_pages} страниц в режиме '{mode}'")
        print("=" * 50)

        # Обрабатываем страницы
        results = self.process_pages(images, mode, page_start)

        # Собираем финальный результат
        final_result = {
            "session_id": session_id,
            "source_pdf": str(pdf_path),
            "mode": mode,
            "total_pages": total_pages,
            "processed_at": datetime.now().isoformat(),
            "results": results,
            "successful_batches": sum(1 for r in results if r["status"] == "success"),
            "total_tokens": {
                "input": sum(r.get("usage", {}).get("input_tokens", 0) for r in results),
                "output": sum(r.get("usage", {}).get("output_tokens", 0) for r in results)
            }
        }

        # Сохраняем результаты
        output_json = self.output_dir / f"{session_id}_processed.json"
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)

        # Создаем Markdown
        output_md = self.output_dir / f"{session_id}_processed.md"
        self._save_as_markdown(final_result, output_md)

        print("\n" + "=" * 50)
        print(f"✅ Обработка завершена!")
        print(f"📊 Статистика:")
        print(f"   - Страниц: {total_pages}")
        print(f"   - Успешных батчей: {final_result['successful_batches']}/{len(results)}")
        print(f"   - Токенов: {final_result['total_tokens']['input']:,} вход / {final_result['total_tokens']['output']:,} выход")
        print(f"\n📁 Результаты:")
        print(f"   - JSON: {output_json}")
        print(f"   - Markdown: {output_md}")

        return final_result

    def _save_as_markdown(self, result: Dict, output_path: Path):
        """Сохраняет результат в Markdown."""

        md_content = f"""# 👁️ Vision Analysis: {Path(result['source_pdf']).name}

**Дата обработки:** {result['processed_at']}
**Режим:** {result['mode']}
**Страниц:** {result['total_pages']}
**Токенов:** {result['total_tokens']['input']:,} вход / {result['total_tokens']['output']:,} выход

---

"""

        for r in result["results"]:
            if r["status"] == "success":
                md_content += f"""
## Страницы {r['page_range']}

{r['response']}

---
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)


def main():
    """CLI интерфейс."""
    parser = argparse.ArgumentParser(
        description="PDF Vision Processor - анализ PDF через Claude Vision"
    )

    subparsers = parser.add_subparsers(dest="command", help="Команды")

    # Команда process
    process_parser = subparsers.add_parser("process", help="Обработать PDF через Vision")
    process_parser.add_argument("pdf_path", help="Путь к PDF файлу")
    process_parser.add_argument(
        "--mode", "-m",
        choices=["extract_text", "summary", "key_concepts", "coaching_tools", "questions", "full_analysis"],
        default="full_analysis",
        help="Режим обработки (default: full_analysis)"
    )
    process_parser.add_argument(
        "--pages", "-p",
        help="Диапазон страниц (например: 1-10)"
    )
    process_parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI для конвертации (default: 150)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        processor = PDFVisionProcessor()

        if args.command == "process":
            # Парсим диапазон страниц
            page_range = None
            if args.pages:
                parts = args.pages.split("-")
                page_range = (int(parts[0]), int(parts[1]))

            # Обновляем DPI если указан
            if args.dpi:
                processor.config["dpi"] = args.dpi

            processor.process_pdf(
                args.pdf_path,
                mode=args.mode,
                page_range=page_range
            )

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
