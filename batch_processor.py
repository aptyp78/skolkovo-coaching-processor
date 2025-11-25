#!/usr/bin/env python3
"""
Batch Processor - обработка всех PDF в папке и создание единой базы знаний.

Использование:
    python batch_processor.py /путь/к/папке/с/pdf
    python batch_processor.py /путь/к/папке/с/pdf --mode summary
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

from pdf_ai_processor import PDFAIProcessor


def process_all_pdfs(folder_path: str, mode: str = "full_analysis", 
                     api_key: str = None, skip_processed: bool = True):
    """
    Обрабатывает все PDF в указанной папке.
    
    Args:
        folder_path: Путь к папке с PDF
        mode: Режим обработки
        api_key: API ключ
        skip_processed: Пропускать уже обработанные файлы
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Папка не найдена: {folder}")
        return
    
    pdf_files = list(folder.glob("*.pdf")) + list(folder.glob("*.PDF"))
    
    if not pdf_files:
        print(f"❌ PDF файлы не найдены в: {folder}")
        return
    
    print(f"\n📂 Найдено {len(pdf_files)} PDF файлов в {folder}")
    print("=" * 60)
    
    processor = PDFAIProcessor(api_key=api_key)
    
    # Проверяем уже обработанные
    processed_files = set(f.stem.split("_")[0] for f in processor.output_dir.glob("*_processed.json"))
    
    results = []
    errors = []
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] 📄 {pdf_file.name}")
        
        # Пропускаем обработанные
        if skip_processed and pdf_file.stem in [p.split("_")[0] for p in processed_files]:
            print("   ⏭️ Уже обработан, пропускаю")
            continue
        
        try:
            result = processor.process_pdf(str(pdf_file), mode=mode)
            results.append({
                "file": pdf_file.name,
                "status": "success",
                "chunks": result["successful_chunks"],
                "tokens": result["total_tokens"]
            })
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            errors.append({
                "file": pdf_file.name,
                "error": str(e)
            })
    
    # Строим базу знаний
    print("\n" + "=" * 60)
    print("📚 Создаю единую базу знаний...")
    processor.build_knowledge_base()
    
    # Отчет
    print("\n" + "=" * 60)
    print("📊 ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 60)
    print(f"✅ Успешно обработано: {len(results)}")
    print(f"❌ Ошибок: {len(errors)}")
    
    if results:
        total_tokens = sum(r["tokens"]["input"] + r["tokens"]["output"] for r in results)
        print(f"📈 Всего токенов: {total_tokens:,}")
        print(f"💰 Примерная стоимость: ${total_tokens * 0.000003:.2f} (Sonnet)")
    
    if errors:
        print("\n⚠️ Файлы с ошибками:")
        for e in errors:
            print(f"   - {e['file']}: {e['error']}")
    
    # Сохраняем отчет
    report = {
        "processed_at": datetime.now().isoformat(),
        "folder": str(folder),
        "mode": mode,
        "successful": results,
        "errors": errors
    }
    
    report_path = processor.output_dir / "batch_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 Отчет сохранен: {report_path}")


def create_structured_knowledge_base(api_key: str = None):
    """
    Создает структурированную базу знаний с категоризацией.
    """
    processor = PDFAIProcessor(api_key=api_key)
    
    # Загружаем все обработанные файлы
    processed_files = list(processor.output_dir.glob("*_processed.json"))
    
    if not processed_files:
        print("❌ Нет обработанных файлов")
        return
    
    print(f"📚 Создаю структурированную базу из {len(processed_files)} файлов...")
    
    # Структура для категоризации
    structured_kb = {
        "meta": {
            "created_at": datetime.now().isoformat(),
            "program": "SKOLKOVO Executive Coaching & Mentoring",
            "total_sources": len(processed_files)
        },
        "categories": {
            "coaching_fundamentals": {
                "name": "Основы коучинга",
                "items": []
            },
            "psychological_concepts": {
                "name": "Психологические концепции",
                "items": []
            },
            "team_development": {
                "name": "Развитие команд",
                "items": []
            },
            "emotional_intelligence": {
                "name": "Эмоциональный интеллект",
                "items": []
            },
            "leadership": {
                "name": "Лидерство",
                "items": []
            },
            "tools_and_techniques": {
                "name": "Инструменты и техники",
                "items": []
            },
            "models_and_frameworks": {
                "name": "Модели и фреймворки",
                "items": []
            },
            "questions_bank": {
                "name": "Банк вопросов",
                "items": []
            }
        },
        "sources": [],
        "index": {}  # Для быстрого поиска
    }
    
    # Ключевые слова для категоризации
    category_keywords = {
        "coaching_fundamentals": ["коучинг", "коуч", "контракт", "сессия", "клиент"],
        "psychological_concepts": ["психолог", "бессознательн", "защит", "перенос", "проекц"],
        "team_development": ["команд", "группов", "динамик", "роль", "конфликт"],
        "emotional_intelligence": ["эмоц", "чувств", "эмпатия", "осознанность"],
        "leadership": ["лидер", "руковод", "управлен", "влияни"],
        "tools_and_techniques": ["техник", "упражнен", "метод", "практик", "инструмент"],
        "models_and_frameworks": ["модель", "фреймворк", "концепц", "теори", "подход"]
    }
    
    for file_path in processed_files:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        
        source_info = {
            "filename": Path(data["source_pdf"]).name,
            "session_id": data["session_id"],
            "pages": data["metadata"].get("pages", 0),
            "processed_at": data["processed_at"]
        }
        structured_kb["sources"].append(source_info)
        
        # Обрабатываем каждый результат
        for result in data["results"]:
            if result["status"] != "success":
                continue
            
            content = result["response"]
            content_lower = content.lower()
            
            # Категоризируем
            for category, keywords in category_keywords.items():
                if any(kw in content_lower for kw in keywords):
                    item = {
                        "source": source_info["filename"],
                        "page_range": result["page_range"],
                        "content": content,
                        "chunk_id": result["chunk_id"]
                    }
                    structured_kb["categories"][category]["items"].append(item)
            
            # Извлекаем вопросы
            if "?" in content:
                questions = [
                    line.strip() for line in content.split("\n")
                    if "?" in line and len(line) > 20
                ]
                for q in questions[:10]:  # Максимум 10 вопросов на чанк
                    structured_kb["categories"]["questions_bank"]["items"].append({
                        "question": q,
                        "source": source_info["filename"],
                        "page_range": result["page_range"]
                    })
            
            # Добавляем в индекс
            words = set(content_lower.split())
            for word in words:
                if len(word) > 4:  # Только слова длиннее 4 символов
                    if word not in structured_kb["index"]:
                        structured_kb["index"][word] = []
                    structured_kb["index"][word].append({
                        "source": source_info["filename"],
                        "chunk_id": result["chunk_id"]
                    })
    
    # Статистика
    print("\n📊 Статистика базы знаний:")
    for cat_id, cat_data in structured_kb["categories"].items():
        print(f"   {cat_data['name']}: {len(cat_data['items'])} элементов")
    
    # Сохраняем
    kb_path = processor.knowledge_dir / "structured_knowledge_base.json"
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(structured_kb, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Структурированная база сохранена: {kb_path}")
    
    # Создаем также Markdown версию для чтения
    create_markdown_knowledge_base(structured_kb, processor.knowledge_dir)
    
    return structured_kb


def create_markdown_knowledge_base(kb: dict, output_dir: Path):
    """Создает читаемую Markdown версию базы знаний."""
    
    md_content = f"""# 📚 База знаний SKOLKOVO Executive Coaching

**Создано:** {kb['meta']['created_at']}  
**Источников:** {kb['meta']['total_sources']}

---

## 📖 Содержание

"""
    
    # TOC
    for cat_id, cat_data in kb["categories"].items():
        if cat_data["items"]:
            md_content += f"- [{cat_data['name']}](#{cat_id}) ({len(cat_data['items'])})\n"
    
    md_content += "\n---\n\n"
    
    # Категории
    for cat_id, cat_data in kb["categories"].items():
        if not cat_data["items"]:
            continue
        
        md_content += f"## {cat_data['name']} {{#{cat_id}}}\n\n"
        
        if cat_id == "questions_bank":
            # Для вопросов особый формат
            for item in cat_data["items"][:50]:  # Первые 50
                md_content += f"- {item['question']}\n"
                md_content += f"  *[{item['source']}, стр. {item['page_range']}]*\n\n"
        else:
            for i, item in enumerate(cat_data["items"][:20], 1):  # Первые 20
                md_content += f"### {i}. [{item['source']}, стр. {item['page_range']}]\n\n"
                md_content += f"{item['content'][:1000]}...\n\n"
                md_content += "---\n\n"
    
    md_path = output_dir / "knowledge_base.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print(f"📄 Markdown версия: {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch обработка PDF и создание структурированной базы знаний"
    )
    
    subparsers = parser.add_subparsers(dest="command")
    
    # Команда process
    process_parser = subparsers.add_parser("process", help="Обработать все PDF в папке")
    process_parser.add_argument("folder", help="Путь к папке с PDF")
    process_parser.add_argument("--mode", "-m", default="full_analysis",
                               choices=["summary", "key_concepts", "coaching_tools", 
                                       "questions", "full_analysis"])
    process_parser.add_argument("--reprocess", action="store_true",
                               help="Переобработать уже обработанные файлы")
    
    # Команда structure
    structure_parser = subparsers.add_parser("structure", 
                                            help="Создать структурированную базу знаний")
    
    # Общие аргументы
    parser.add_argument("--api-key", "-k", help="API ключ Anthropic")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "process":
        process_all_pdfs(
            args.folder,
            mode=args.mode,
            api_key=args.api_key,
            skip_processed=not args.reprocess
        )
    
    elif args.command == "structure":
        create_structured_knowledge_base(api_key=args.api_key)


if __name__ == "__main__":
    main()
