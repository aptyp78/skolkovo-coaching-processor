#!/usr/bin/env python3
"""
Интерактивный режим работы с базой знаний SKOLKOVO Executive Coaching.

Этот скрипт позволяет:
- Загружать и просматривать обработанные материалы
- Задавать вопросы по базе знаний
- Искать по ключевым словам
- Генерировать саммари по темам
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

try:
    import anthropic
except ImportError:
    print("Установите: pip install anthropic")
    sys.exit(1)

# Импортируем утилиты
sys.path.insert(0, str(Path(__file__).parent))
from utils import get_model, get_logger, load_env_file, KNOWLEDGE_DIR, OUTPUT_DIR

# Инициализация
load_env_file()
logger = get_logger(__name__)


class InteractiveKnowledgeBase:
    """Интерактивная работа с базой знаний."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Установите ANTHROPIC_API_KEY или передайте api_key")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = get_model()  # Из конфига, не хардкод
        self.knowledge_dir = KNOWLEDGE_DIR
        self.output_dir = OUTPUT_DIR

        self.kb = None
        self.conversation_history = []

        logger.info(f"InteractiveKnowledgeBase инициализирован, модель: {self.model}")
        
    def load_knowledge_base(self) -> bool:
        """Загружает базу знаний."""
        kb_path = self.knowledge_dir / "knowledge_base.json"
        
        if not kb_path.exists():
            print("⚠️ База знаний не найдена. Сначала обработайте PDF файлы.")
            return False
        
        with open(kb_path, encoding="utf-8") as f:
            self.kb = json.load(f)
        
        print(f"✅ База знаний загружена")
        print(f"   Источников: {len(self.kb.get('sources', []))}")
        print(f"   Фрагментов: {len(self.kb.get('summaries', []))}")
        return True
    
    def search(self, query: str, top_k: int = 5) -> list:
        """Ищет релевантные фрагменты."""
        if not self.kb:
            return []
        
        results = []
        query_words = set(query.lower().split())
        
        for summary in self.kb.get("summaries", []):
            content_lower = summary["content"].lower()
            
            # Подсчет совпадений
            matches = sum(1 for word in query_words if word in content_lower)
            
            if matches > 0:
                results.append({
                    "score": matches,
                    "source": summary["source"],
                    "page_range": summary["page_range"],
                    "content": summary["content"][:500] + "..."
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def ask(self, question: str, include_history: bool = True) -> str:
        """Задает вопрос с контекстом из базы знаний."""
        
        # Находим релевантный контекст
        relevant = self.search(question, top_k=5)
        
        context = "\n\n---\n\n".join([
            f"[{r['source']}, стр. {r['page_range']}]\n{r['content']}"
            for r in relevant
        ]) if relevant else "Релевантный контекст не найден."
        
        # Формируем сообщения
        messages = []
        
        if include_history and self.conversation_history:
            # Добавляем последние 3 обмена
            for msg in self.conversation_history[-6:]:
                messages.append(msg)
        
        messages.append({
            "role": "user",
            "content": f"""КОНТЕКСТ ИЗ УЧЕБНЫХ МАТЕРИАЛОВ:
{context}

---

ВОПРОС: {question}"""
        })
        
        system_prompt = """Ты - AI-коуч и ментор программы Executive Coaching & Mentoring в СКОЛКОВО.

ТВОЯ РОЛЬ:
1. Отвечай на вопросы по учебным материалам программы
2. Помогай связывать концепции между собой
3. Предлагай практическое применение
4. Задавай уточняющие и углубляющие вопросы

ФОРМАТ ОТВЕТОВ:
- Ссылайся на конкретные источники и страницы
- Используй примеры из материалов
- Предлагай вопросы для дальнейшего исследования
- Если информации недостаточно, честно скажи об этом

КОНТЕКСТ ПРОГРАММЫ:
- 8 модулей с октября 2025 по июль 2026
- Фокус: executive-коучинг, эмоциональный интеллект, лидерство
- Практика: индивидуальный и групповой коучинг"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system_prompt,
            messages=messages
        )
        
        answer = response.content[0].text
        
        # Сохраняем в историю
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})
        
        return answer
    
    def list_sources(self):
        """Показывает список источников."""
        if not self.kb:
            print("База знаний не загружена")
            return
        
        print("\n📚 ИСТОЧНИКИ В БАЗЕ ЗНАНИЙ:\n")
        for i, source in enumerate(self.kb.get("sources", []), 1):
            print(f"  {i}. {source['filename']} ({source.get('pages', '?')} стр.)")
    
    def get_summary_for_source(self, source_name: str) -> str:
        """Получает все саммари для конкретного источника."""
        if not self.kb:
            return "База знаний не загружена"
        
        summaries = [
            s for s in self.kb.get("summaries", [])
            if source_name.lower() in s["source"].lower()
        ]
        
        if not summaries:
            return f"Не найдено материалов для: {source_name}"
        
        return "\n\n---\n\n".join([
            f"## Страницы {s['page_range']}\n\n{s['content']}"
            for s in summaries
        ])
    
    def run_interactive(self):
        """Запускает интерактивный режим."""
        print("""
╔══════════════════════════════════════════════════════════════╗
║     SKOLKOVO Executive Coaching - База знаний                ║
║                                                              ║
║  Команды:                                                    ║
║    /search <запрос>  - Поиск по материалам                   ║
║    /sources          - Список источников                     ║
║    /source <имя>     - Показать материал по источнику        ║
║    /clear            - Очистить историю диалога              ║
║    /help             - Показать справку                      ║
║    /quit             - Выход                                 ║
║                                                              ║
║  Или просто задайте вопрос!                                  ║
╚══════════════════════════════════════════════════════════════╝
""")
        
        if not self.load_knowledge_base():
            return
        
        while True:
            try:
                user_input = input("\n🎯 Вы: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=1)
                    command = parts[0].lower()
                    arg = parts[1] if len(parts) > 1 else ""
                    
                    if command == "/quit":
                        print("👋 До встречи!")
                        break
                    
                    elif command == "/help":
                        print("""
Команды:
  /search <запрос>  - Поиск фрагментов по ключевым словам
  /sources          - Показать все загруженные источники
  /source <имя>     - Показать саммари конкретного PDF
  /clear            - Очистить историю диалога
  /quit             - Выход

Без команды: задайте любой вопрос по материалам программы
""")
                    
                    elif command == "/sources":
                        self.list_sources()
                    
                    elif command == "/source":
                        if arg:
                            print(self.get_summary_for_source(arg))
                        else:
                            print("Укажите имя источника: /source <имя файла>")
                    
                    elif command == "/search":
                        if arg:
                            results = self.search(arg)
                            if results:
                                print(f"\n🔍 Найдено {len(results)} результатов:\n")
                                for i, r in enumerate(results, 1):
                                    print(f"{i}. [{r['source']}, стр. {r['page_range']}]")
                                    print(f"   {r['content'][:200]}...\n")
                            else:
                                print("Ничего не найдено")
                        else:
                            print("Укажите поисковый запрос: /search <запрос>")
                    
                    elif command == "/clear":
                        self.conversation_history = []
                        print("История очищена")
                    
                    else:
                        print(f"Неизвестная команда: {command}")
                
                else:
                    # Обычный вопрос
                    print("\n🤖 Думаю...")
                    answer = self.ask(user_input)
                    print(f"\n📚 Ответ:\n\n{answer}")
            
            except KeyboardInterrupt:
                print("\n\n👋 До встречи!")
                break
            
            except Exception as e:
                print(f"\n❌ Ошибка: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Интерактивная работа с базой знаний")
    parser.add_argument("--api-key", "-k", help="API ключ Anthropic")
    
    args = parser.parse_args()
    
    try:
        kb = InteractiveKnowledgeBase(api_key=args.api_key)
        kb.run_interactive()
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
