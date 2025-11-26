#!/usr/bin/env python3
"""
PDF AI Processor для SKOLKOVO Executive Coaching Program
=========================================================
Система для обработки больших PDF через Claude API с управлением контекстным окном.

Возможности:
- Извлечение текста из PDF (включая OCR для сканов)
- Умный чанкинг с учетом семантических границ
- Обработка через Claude API с сохранением промежуточных результатов
- Создание структурированной базы знаний
- Интерактивный режим для вопросов по материалам
"""

import os
import sys
import json
import hashlib
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import argparse

# PDF processing
try:
    import pdfplumber
    from pypdf import PdfReader
except ImportError:
    print("Установите зависимости: pip install pdfplumber pypdf")
    sys.exit(1)

# Claude API
try:
    import anthropic
except ImportError:
    print("Установите anthropic: pip install anthropic")
    sys.exit(1)

# Импортируем утилиты
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    get_logger, get_rate_limiter, load_config,
    OUTPUT_DIR, CHUNKS_DIR, KNOWLEDGE_DIR, load_env_file
)

# Инициализация
load_env_file()
logger = get_logger(__name__)
rate_limiter = get_rate_limiter()

# Configuration
DEFAULT_CONFIG = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens_per_chunk": 4000,  # ~16K символов текста
    "max_output_tokens": 4096,
    "overlap_tokens": 200,  # Перекрытие между чанками для контекста
    "language": "ru",
    "processing_modes": {
        "summary": "Создай структурированное саммари этого фрагмента",
        "key_concepts": "Извлеки ключевые концепции, модели и инструменты",
        "coaching_tools": "Извлеки инструменты и техники для коучинга",
        "questions": "Сгенерируй вопросы для рефлексии по материалу",
        "full_analysis": "Полный анализ: саммари + концепции + инструменты + вопросы"
    }
}

class PDFAIProcessor:
    """Основной класс для обработки PDF через Claude API."""

    def __init__(self, api_key: str = None, config: dict = None):
        """Инициализация процессора."""
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API ключ не найден. Передайте его напрямую или установите ANTHROPIC_API_KEY")

        self.client = anthropic.Anthropic(api_key=self.api_key)

        # Загружаем конфиг из utils, мерджим с переданным
        base_config = load_config()
        self.config = {**DEFAULT_CONFIG, **base_config, **(config or {})}

        # Используем пути из utils
        self.output_dir = OUTPUT_DIR
        self.chunks_dir = CHUNKS_DIR
        self.knowledge_dir = KNOWLEDGE_DIR

        logger.info(f"PDFAIProcessor инициализирован, модель: {self.config['model']}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """
        Извлекает текст из PDF с метаданными.
        
        Returns:
            Tuple[str, Dict]: (текст, метаданные)
        """
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
                    
                    # Извлекаем текст
                    text = page.extract_text() or ""
                    
                    # Извлекаем таблицы
                    tables = page.extract_tables()
                    
                    # Формируем контент страницы
                    page_content = f"\n\n--- СТРАНИЦА {i} ---\n\n{text}"
                    
                    if tables:
                        page_content += "\n\n[ТАБЛИЦЫ НА СТРАНИЦЕ]\n"
                        for j, table in enumerate(tables, 1):
                            page_content += f"\nТаблица {j}:\n"
                            for row in table:
                                if row:
                                    page_content += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
                    
                    full_text.append(page_content)
        
        except Exception as e:
            print(f"\n⚠️ Ошибка при чтении PDF: {e}")
            # Пробуем альтернативный метод
            try:
                reader = PdfReader(pdf_path)
                metadata["pages"] = len(reader.pages)
                for i, page in enumerate(reader.pages, 1):
                    text = page.extract_text() or ""
                    full_text.append(f"\n\n--- СТРАНИЦА {i} ---\n\n{text}")
            except Exception as e2:
                raise Exception(f"Не удалось извлечь текст: {e2}")
        
        result_text = "\n".join(full_text)
        metadata["characters"] = len(result_text)
        
        print(f"\n✅ Извлечено {metadata['pages']} страниц, {metadata['characters']:,} символов")
        
        return result_text, metadata
    
    def smart_chunk_text(self, text: str, max_chars: int = None) -> List[Dict]:
        """
        Умный чанкинг текста с учетом семантических границ.
        
        Returns:
            List[Dict]: Список чанков с метаданными
        """
        if max_chars is None:
            max_chars = self.config["max_tokens_per_chunk"] * 4  # ~4 символа на токен
        
        overlap_chars = self.config["overlap_tokens"] * 4
        
        chunks = []
        
        # Разделители в порядке приоритета
        separators = [
            r'\n\n--- СТРАНИЦА \d+ ---\n\n',  # Границы страниц
            r'\n\n#{1,3} ',  # Заголовки markdown
            r'\n\n[А-ЯA-Z][А-Яа-яA-Za-z\s]+:\n',  # Заголовки с двоеточием
            r'\n\n\d+\.\s',  # Нумерованные разделы
            r'\n\n',  # Двойной перенос
            r'\n',  # Одинарный перенос
            r'\. ',  # Точка с пробелом
        ]
        
        current_pos = 0
        chunk_num = 0
        
        while current_pos < len(text):
            chunk_end = min(current_pos + max_chars, len(text))
            
            # Если не конец текста, ищем хорошую точку разрыва
            if chunk_end < len(text):
                best_break = None
                
                # Ищем разрыв начиная с конца допустимого диапазона
                search_start = max(current_pos + max_chars // 2, current_pos)
                search_text = text[search_start:chunk_end]
                
                for sep in separators:
                    matches = list(re.finditer(sep, search_text))
                    if matches:
                        # Берем последнее совпадение в допустимом диапазоне
                        best_break = search_start + matches[-1].end()
                        break
                
                if best_break:
                    chunk_end = best_break
            
            # Создаем чанк
            chunk_text = text[current_pos:chunk_end].strip()
            
            if chunk_text:
                chunk_num += 1
                
                # Определяем номера страниц в чанке
                page_matches = re.findall(r'--- СТРАНИЦА (\d+) ---', chunk_text)
                pages = [int(p) for p in page_matches] if page_matches else []
                
                chunks.append({
                    "chunk_id": chunk_num,
                    "text": chunk_text,
                    "char_start": current_pos,
                    "char_end": chunk_end,
                    "char_count": len(chunk_text),
                    "pages": pages,
                    "page_range": f"{min(pages)}-{max(pages)}" if pages else "N/A"
                })
            
            # Двигаемся с учетом перекрытия
            current_pos = chunk_end - overlap_chars if chunk_end < len(text) else chunk_end
        
        print(f"📦 Создано {len(chunks)} чанков")
        return chunks
    
    def get_system_prompt(self, mode: str, context: str = "") -> str:
        """Создает системный промпт для обработки."""
        
        base_prompt = """Ты - AI-ассистент для программы Executive Coaching & Mentoring в СКОЛКОВО.
Твоя задача - обработать учебный материал и извлечь из него ценную информацию.

КОНТЕКСТ ПРОГРАММЫ:
- Программа по развитию коучинговых компетенций для руководителей
- Фокус на executive-коучинг (индивидуальный и групповой)
- Интеграция психологических концепций в коучинг
- Развитие эмоционального интеллекта и лидерства

ПРИНЦИПЫ ИЗВЛЕЧЕНИЯ:
1. Сохраняй оригинальную терминологию
2. Выделяй практические инструменты и техники
3. Отмечай связи между концепциями
4. Указывай авторов и источники, если упоминаются
5. Структурируй информацию для легкого поиска

"""
        
        mode_prompts = {
            "summary": """ЗАДАЧА: Создай структурированное саммари

ФОРМАТ ОТВЕТА:
## Основные темы
[Перечисли 3-5 ключевых тем]

## Ключевые идеи
[Краткое описание главных идей]

## Важные цитаты/определения
[Точные формулировки, которые стоит запомнить]

## Связи с другими концепциями
[Как это связано с коучингом/лидерством]
""",
            
            "key_concepts": """ЗАДАЧА: Извлеки ключевые концепции, модели и инструменты

ФОРМАТ ОТВЕТА (JSON):
{
    "concepts": [
        {
            "name": "Название концепции",
            "definition": "Краткое определение",
            "author": "Автор (если известен)",
            "application": "Применение в коучинге"
        }
    ],
    "models": [
        {
            "name": "Название модели",
            "components": ["компонент1", "компонент2"],
            "usage": "Как использовать"
        }
    ],
    "tools": [
        {
            "name": "Название инструмента",
            "description": "Описание",
            "when_to_use": "Когда применять"
        }
    ]
}
""",
            
            "coaching_tools": """ЗАДАЧА: Извлеки инструменты и техники для коучинговой практики

ФОРМАТ ОТВЕТА:
### Техники и упражнения
- **Название**: Описание и пошаговая инструкция

### Вопросы для коуча
[Мощные вопросы из материала]

### Фреймворки для сессий
[Структуры для проведения коучинговых сессий]

### Практические рекомендации
[Конкретные советы по применению]
""",
            
            "questions": """ЗАДАЧА: Сгенерируй вопросы для рефлексии

ФОРМАТ ОТВЕТА:
### Вопросы для понимания
[5 вопросов на проверку понимания материала]

### Вопросы для применения  
[5 вопросов о применении в практике]

### Вопросы для самоисследования
[5 глубоких вопросов для личной рефлексии]

### Вопросы для обсуждения
[3 вопроса для групповой дискуссии]
""",
            
            "full_analysis": """ЗАДАЧА: Проведи полный анализ материала

СТРУКТУРА ОТВЕТА:

## 1. САММАРИ
[Краткое содержание в 3-5 абзацах]

## 2. КЛЮЧЕВЫЕ КОНЦЕПЦИИ
| Концепция | Определение | Применение |
|-----------|-------------|------------|
[Таблица концепций]

## 3. ИНСТРУМЕНТЫ И ТЕХНИКИ
[Список с описаниями]

## 4. МОЩНЫЕ ВОПРОСЫ
[Вопросы из материала или по материалу]

## 5. СВЯЗИ И ИНСАЙТЫ
[Как это связано с другими темами коучинга]

## 6. ПРАКТИЧЕСКОЕ ПРИМЕНЕНИЕ
[Конкретные шаги для применения]

## 7. ТЕГИ ДЛЯ ПОИСКА
[5-10 ключевых слов/тегов]
"""
        }
        
        return base_prompt + mode_prompts.get(mode, mode_prompts["summary"]) + f"\n\n{context}"
    
    def process_chunk(self, chunk: Dict, mode: str = "full_analysis",
                      additional_context: str = "") -> Dict:
        """
        Обрабатывает один чанк через Claude API.

        Returns:
            Dict: Результат обработки с метаданными
        """
        system_prompt = self.get_system_prompt(mode, additional_context)

        user_message = f"""Обработай следующий фрагмент учебного материала:

---
{chunk['text']}
---

Страницы: {chunk.get('page_range', 'N/A')}
Фрагмент: {chunk['chunk_id']}
"""

        try:
            # Rate limiting перед API вызовом
            rate_limiter.wait()
            logger.debug(f"Обработка чанка {chunk['chunk_id']}, страницы {chunk.get('page_range', 'N/A')}")

            response = self.client.messages.create(
                model=self.config["model"],
                max_tokens=self.config["max_output_tokens"],
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )
            
            result = {
                "chunk_id": chunk["chunk_id"],
                "page_range": chunk.get("page_range", "N/A"),
                "mode": mode,
                "processed_at": datetime.now().isoformat(),
                "response": response.content[0].text,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                "status": "success"
            }
            
        except Exception as e:
            result = {
                "chunk_id": chunk["chunk_id"],
                "page_range": chunk.get("page_range", "N/A"),
                "mode": mode,
                "processed_at": datetime.now().isoformat(),
                "error": str(e),
                "status": "error"
            }
        
        return result
    
    def process_pdf(self, pdf_path: str, mode: str = "full_analysis",
                    save_intermediate: bool = True, resume: bool = True) -> Dict:
        """
        Полный пайплайн обработки PDF.
        
        Args:
            pdf_path: Путь к PDF
            mode: Режим обработки
            save_intermediate: Сохранять промежуточные результаты
            resume: Продолжить с места остановки
            
        Returns:
            Dict: Полный результат обработки
        """
        pdf_path = Path(pdf_path)
        
        # Создаем уникальный ID для этого PDF
        pdf_hash = hashlib.md5(pdf_path.name.encode()).hexdigest()[:8]
        session_id = f"{pdf_path.stem}_{pdf_hash}"
        
        # Проверяем, есть ли сохраненный прогресс
        progress_file = self.chunks_dir / f"{session_id}_progress.json"
        
        if resume and progress_file.exists():
            with open(progress_file) as f:
                progress = json.load(f)
            print(f"📂 Продолжаю обработку с чанка {progress['last_completed'] + 1}")
            start_chunk = progress["last_completed"] + 1
            chunks = progress["chunks"]
            results = progress["results"]
            metadata = progress["metadata"]
        else:
            # Извлекаем текст
            text, metadata = self.extract_text_from_pdf(pdf_path)
            
            # Сохраняем сырой текст
            raw_text_path = self.output_dir / f"{session_id}_raw.txt"
            with open(raw_text_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"💾 Сырой текст сохранен: {raw_text_path}")
            
            # Разбиваем на чанки
            chunks = self.smart_chunk_text(text)
            results = []
            start_chunk = 0
        
        # Обрабатываем чанки
        total_chunks = len(chunks)
        
        print(f"\n🚀 Начинаю обработку {total_chunks} чанков в режиме '{mode}'")
        print("=" * 50)
        
        for i, chunk in enumerate(chunks[start_chunk:], start_chunk + 1):
            print(f"\n📝 Обрабатываю чанк {i}/{total_chunks} (стр. {chunk.get('page_range', 'N/A')})...")
            
            result = self.process_chunk(chunk, mode)
            results.append(result)
            
            if result["status"] == "success":
                print(f"   ✅ Готово ({result['usage']['input_tokens']}→{result['usage']['output_tokens']} токенов)")
            else:
                print(f"   ❌ Ошибка: {result.get('error', 'Unknown')}")
            
            # Сохраняем прогресс
            if save_intermediate:
                progress = {
                    "session_id": session_id,
                    "pdf_path": str(pdf_path),
                    "mode": mode,
                    "last_completed": i,
                    "total_chunks": total_chunks,
                    "chunks": chunks,
                    "results": results,
                    "metadata": metadata,
                    "updated_at": datetime.now().isoformat()
                }
                with open(progress_file, "w", encoding="utf-8") as f:
                    json.dump(progress, f, ensure_ascii=False, indent=2)
        
        # Собираем финальный результат
        final_result = {
            "session_id": session_id,
            "source_pdf": str(pdf_path),
            "metadata": metadata,
            "mode": mode,
            "total_chunks": total_chunks,
            "successful_chunks": sum(1 for r in results if r["status"] == "success"),
            "processed_at": datetime.now().isoformat(),
            "results": results,
            "total_tokens": {
                "input": sum(r.get("usage", {}).get("input_tokens", 0) for r in results),
                "output": sum(r.get("usage", {}).get("output_tokens", 0) for r in results)
            }
        }
        
        # Сохраняем финальный результат
        output_path = self.output_dir / f"{session_id}_processed.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        # Создаем читаемый Markdown
        markdown_path = self.output_dir / f"{session_id}_processed.md"
        self._save_as_markdown(final_result, markdown_path)
        
        # Удаляем файл прогресса
        if progress_file.exists():
            progress_file.unlink()
        
        print("\n" + "=" * 50)
        print(f"✅ Обработка завершена!")
        print(f"📊 Статистика:")
        print(f"   - Обработано чанков: {final_result['successful_chunks']}/{total_chunks}")
        print(f"   - Использовано токенов: {final_result['total_tokens']['input']:,} вход / {final_result['total_tokens']['output']:,} выход")
        print(f"\n📁 Результаты сохранены:")
        print(f"   - JSON: {output_path}")
        print(f"   - Markdown: {markdown_path}")
        
        return final_result
    
    def _save_as_markdown(self, result: Dict, output_path: Path):
        """Сохраняет результат в читаемом Markdown формате."""
        
        md_content = f"""# Обработанный материал: {result['source_pdf']}

**Дата обработки:** {result['processed_at']}
**Режим:** {result['mode']}
**Страниц в PDF:** {result['metadata'].get('pages', 'N/A')}
**Обработано чанков:** {result['successful_chunks']}/{result['total_chunks']}

---

"""
        
        for r in result["results"]:
            if r["status"] == "success":
                md_content += f"""
## Фрагмент {r['chunk_id']} (стр. {r['page_range']})

{r['response']}

---
"""
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
    
    def build_knowledge_base(self, processed_files: List[str] = None) -> Dict:
        """
        Строит базу знаний из обработанных файлов.
        
        Args:
            processed_files: Список путей к JSON файлам результатов.
                           Если None, берет все файлы из output/
        
        Returns:
            Dict: Структурированная база знаний
        """
        if processed_files is None:
            processed_files = list(self.output_dir.glob("*_processed.json"))
        
        knowledge_base = {
            "created_at": datetime.now().isoformat(),
            "sources": [],
            "concepts": [],
            "tools": [],
            "questions": [],
            "summaries": []
        }
        
        for file_path in processed_files:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            
            source_info = {
                "filename": Path(data["source_pdf"]).name,
                "session_id": data["session_id"],
                "pages": data["metadata"].get("pages", 0)
            }
            knowledge_base["sources"].append(source_info)
            
            # Парсим результаты и извлекаем структурированную информацию
            for result in data["results"]:
                if result["status"] == "success":
                    response = result["response"]
                    
                    # Добавляем весь контент как саммари
                    knowledge_base["summaries"].append({
                        "source": source_info["filename"],
                        "chunk_id": result["chunk_id"],
                        "page_range": result["page_range"],
                        "content": response
                    })
        
        # Сохраняем базу знаний
        kb_path = self.knowledge_dir / "knowledge_base.json"
        with open(kb_path, "w", encoding="utf-8") as f:
            json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
        
        print(f"📚 База знаний создана: {kb_path}")
        print(f"   - Источников: {len(knowledge_base['sources'])}")
        print(f"   - Саммари: {len(knowledge_base['summaries'])}")
        
        return knowledge_base
    
    def ask_question(self, question: str, knowledge_base_path: str = None,
                     max_context_chunks: int = 5) -> str:
        """
        Задает вопрос по базе знаний.
        
        Args:
            question: Вопрос пользователя
            knowledge_base_path: Путь к базе знаний
            max_context_chunks: Максимум чанков для контекста
            
        Returns:
            str: Ответ от Claude
        """
        if knowledge_base_path is None:
            knowledge_base_path = self.knowledge_dir / "knowledge_base.json"
        
        with open(knowledge_base_path, encoding="utf-8") as f:
            kb = json.load(f)
        
        # Простой поиск релевантных чанков (можно улучшить с embeddings)
        relevant_chunks = []
        question_lower = question.lower()
        
        for summary in kb["summaries"]:
            content_lower = summary["content"].lower()
            # Простой скоринг по совпадению слов
            score = sum(1 for word in question_lower.split() if word in content_lower)
            if score > 0:
                relevant_chunks.append((score, summary))
        
        # Сортируем по релевантности и берем топ
        relevant_chunks.sort(key=lambda x: x[0], reverse=True)
        context_chunks = [c[1] for c in relevant_chunks[:max_context_chunks]]
        
        # Формируем контекст
        context = "\n\n---\n\n".join([
            f"Источник: {c['source']}, стр. {c['page_range']}\n{c['content']}"
            for c in context_chunks
        ])
        
        system_prompt = """Ты - AI-ассистент программы Executive Coaching & Mentoring в СКОЛКОВО.
Отвечай на вопросы, используя предоставленный контекст из учебных материалов.
Если информации в контексте недостаточно, честно скажи об этом.
Ссылайся на конкретные источники и страницы."""
        
        user_message = f"""КОНТЕКСТ ИЗ УЧЕБНЫХ МАТЕРИАЛОВ:

{context}

---

ВОПРОС: {question}"""
        
        response = self.client.messages.create(
            model=self.config["model"],
            max_tokens=self.config["max_output_tokens"],
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )
        
        return response.content[0].text


def main():
    """CLI интерфейс."""
    parser = argparse.ArgumentParser(
        description="PDF AI Processor для обработки учебных материалов через Claude API"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Команды")
    
    # Команда process
    process_parser = subparsers.add_parser("process", help="Обработать PDF")
    process_parser.add_argument("pdf_path", help="Путь к PDF файлу")
    process_parser.add_argument(
        "--mode", "-m",
        choices=["summary", "key_concepts", "coaching_tools", "questions", "full_analysis"],
        default="full_analysis",
        help="Режим обработки (default: full_analysis)"
    )
    process_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Начать обработку заново, игнорируя сохраненный прогресс"
    )
    
    # Команда ask
    ask_parser = subparsers.add_parser("ask", help="Задать вопрос по базе знаний")
    ask_parser.add_argument("question", help="Вопрос")
    
    # Команда build-kb
    kb_parser = subparsers.add_parser("build-kb", help="Построить базу знаний")
    
    # Общие аргументы
    parser.add_argument(
        "--api-key", "-k",
        help="API ключ Anthropic (или установите ANTHROPIC_API_KEY)"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        processor = PDFAIProcessor(api_key=args.api_key)
        
        if args.command == "process":
            processor.process_pdf(
                args.pdf_path,
                mode=args.mode,
                resume=not args.no_resume
            )
        
        elif args.command == "ask":
            answer = processor.ask_question(args.question)
            print("\n" + "=" * 50)
            print("ОТВЕТ:")
            print("=" * 50)
            print(answer)
        
        elif args.command == "build-kb":
            processor.build_knowledge_base()
    
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
