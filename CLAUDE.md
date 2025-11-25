# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Обзор проекта

SKOLKOVO Materials Processor — система обработки учебных материалов программы Executive Coaching & Mentoring через Claude API и OpenAI Whisper. Обрабатывает PDF-документы, аудио и видео записи семинаров, создаёт структурированную базу знаний.

## Команды

```bash
# Установка зависимостей
pip install -r requirements.txt
# ffmpeg требуется для аудио: brew install ffmpeg (Mac) / apt install ffmpeg (Linux)

# GUI интерфейс (рекомендуемый способ)
python gui.py  # Откроется на http://localhost:7860

# Универсальный обработчик (автоопределение типа файла)
python process.py файл.pdf
python process.py семинар.mp3 --title "Название" --speaker "Спикер"
python process.py папка/ --batch

# PDF обработка (текстовые PDF)
python pdf_ai_processor.py process файл.pdf --mode full_analysis
python pdf_ai_processor.py build-kb  # построить базу знаний

# PDF Vision (сканы, изображения, графики)
python pdf_vision_processor.py process скан.pdf --mode extract_text
python pdf_vision_processor.py process файл.pdf --mode full_analysis --dpi 200
python pdf_vision_processor.py process документ.pdf --pages 1-5  # конкретные страницы

# Аудио транскрипция
python transcribe.py запись.mp3 --title "Название" --speaker "Спикер"
python transcribe.py --provider local_whisper  # бесплатный локальный вариант

# Интерактивный Q&A по базе знаний
python interactive.py
```

## Архитектура

```
gui.py                  # Веб-интерфейс на Gradio (http://localhost:7860)
    ├── 4 вкладки: PDF, Аудио, Результаты, Настройки
    └── Интегрирует все процессоры ниже

process.py              # Точка входа CLI: определяет тип файла и вызывает нужный процессор
    ├── pdf_ai_processor.py   # PDFAIProcessor: извлечение текста, чанкинг, Claude API
    ├── pdf_vision_processor.py # PDFVisionProcessor: Claude Vision для сканов/изображений
    └── audio_transcriber.py  # AudioTranscriber: транскрипция через OpenAI/Whisper/AssemblyAI

interactive.py          # InteractiveKnowledgeBase: Q&A по базе знаний
batch_processor.py      # Массовая обработка
transcribe.py           # CLI для быстрой транскрипции
```

### Поток данных

1. **PDF**: `pdf_ai_processor.py` извлекает текст через pdfplumber/pypdf → разбивает на чанки (4000 токенов) → отправляет в Claude API → сохраняет в `output/` и `knowledge_base/`

2. **Аудио/Видео**: `audio_transcriber.py` конвертирует через pydub/ffmpeg → разбивает на сегменты (20 мин) → транскрибирует через Whisper API → анализирует через Claude → сохраняет в `output/` и `transcripts/`

### Ключевые классы

- `PDFAIProcessor` (pdf_ai_processor.py:56): извлечение текста, чанкинг, обработка через Claude
- `PDFVisionProcessor` (pdf_vision_processor.py:25): конвертация страниц в изображения, Claude Vision API
- `AudioTranscriber` (audio_transcriber.py:70): транскрипция с выбором провайдера (openai/local_whisper/assemblyai)
- `InteractiveKnowledgeBase` (interactive.py:25): поиск и Q&A по собранной базе

### Когда использовать Vision vs текстовый парсер

| Тип документа | Рекомендация |
|---------------|--------------|
| Обычные PDF с текстом | `pdf_ai_processor.py` — быстрее и дешевле |
| Сканы документов | `pdf_vision_processor.py` — единственный вариант |
| PDF с графиками/схемами | `pdf_vision_processor.py` — понимает визуал |
| Презентации | `pdf_vision_processor.py` — сохраняет layout |
| PDF с таблицами | `pdf_vision_processor.py` — лучше распознаёт структуру |

## Конфигурация

`config.json` содержит:
- `model`: модель Claude (claude-sonnet-4-20250514)
- `max_tokens_per_chunk`: размер чанка (4000)
- `transcription_provider`: провайдер транскрипции (openai)
- `processing_modes`: режимы анализа (summary, key_concepts, coaching_tools, questions, full_analysis)

## API ключи

```bash
export ANTHROPIC_API_KEY="sk-ant-..."  # обязательно
export OPENAI_API_KEY="sk-..."          # для Whisper транскрипции
```

## Выходные директории

- `output/` — результаты обработки (JSON + Markdown)
- `transcripts/` — чистые транскрипции
- `knowledge_base/` — единая база знаний (knowledge_base.json)
- `processed_chunks/` — промежуточные результаты чанков

## GUI интерфейс (gui.py)

Веб-интерфейс на Gradio для удобной работы с материалами:

```bash
python gui.py  # Откроется на http://localhost:7860
```

### Вкладки

1. **PDF Обработка**
   - Загрузка одного файла или указание пути к папке
   - Выбор режима: extract_text, summary, full_analysis, key_concepts
   - Выбор метода: Vision (для сканов/презентаций) или Text (для текстовых PDF)
   - Настройка DPI и диапазона страниц
   - Пакетная обработка всех PDF в папке

2. **Аудио Транскрипция**
   - Загрузка аудио файлов (mp3, m4a, wav, ogg)
   - Указание названия и спикера
   - Выбор провайдера: OpenAI Whisper API или локальный Whisper

3. **Результаты**
   - Просмотр всех обработанных файлов
   - Поиск по названию
   - Фильтрация по типу (PDF/Аудио)
   - Просмотр содержимого в форматах JSON и Markdown

4. **Настройки**
   - Проверка API ключей
   - Проверка системных зависимостей (ffmpeg, poppler)
   - Информация о версиях библиотек
