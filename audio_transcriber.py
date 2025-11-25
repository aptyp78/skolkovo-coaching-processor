#!/usr/bin/env python3
"""
Audio Transcription & Processing для SKOLKOVO Executive Coaching
=================================================================
Транскрипция аудиозаписей семинаров и обработка через Claude API.

Поддерживаемые провайдеры:
- OpenAI Whisper API (рекомендуется для качества)
- Локальный Whisper (бесплатно, требует установки)
- AssemblyAI (хорош для длинных записей)

Возможности:
- Транскрипция аудио/видео любой длины
- Автоматическая разбивка длинных файлов
- Обработка через Claude API
- Интеграция с базой знаний PDF
"""

import os
import sys
import json
import math
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import argparse

# Audio processing
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("⚠️ pydub не установлен. Установите: pip install pydub")

# API clients
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import assemblyai as aai
    ASSEMBLYAI_AVAILABLE = True
except ImportError:
    ASSEMBLYAI_AVAILABLE = False


# Configuration
DEFAULT_CONFIG = {
    "transcription_provider": "openai",  # openai, local_whisper, assemblyai
    "whisper_model": "whisper-1",  # Для OpenAI API
    "local_whisper_model": "large-v3",  # Для локального Whisper
    "language": "ru",
    "max_chunk_duration_minutes": 20,  # Whisper API лимит ~25MB
    "claude_model": "claude-sonnet-4-20250514",
    "max_output_tokens": 4096
}


class AudioTranscriber:
    """Транскрипция аудио с различными провайдерами."""
    
    def __init__(self, provider: str = "openai", api_key: str = None):
        """
        Args:
            provider: openai, local_whisper, assemblyai
            api_key: API ключ для выбранного провайдера
        """
        self.provider = provider
        self.api_key = api_key
        
        if provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("Установите openai: pip install openai")
            self.client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        
        elif provider == "assemblyai":
            if not ASSEMBLYAI_AVAILABLE:
                raise ImportError("Установите assemblyai: pip install assemblyai")
            aai.settings.api_key = api_key or os.environ.get("ASSEMBLYAI_API_KEY")
        
        elif provider == "local_whisper":
            # Проверяем наличие whisper
            try:
                result = subprocess.run(["whisper", "--help"], capture_output=True)
            except FileNotFoundError:
                raise ImportError("Локальный Whisper не найден. Установите: pip install openai-whisper")
    
    def transcribe(self, audio_path: str, language: str = "ru") -> Dict:
        """
        Транскрибирует аудиофайл.
        
        Returns:
            Dict: {
                "text": str,
                "segments": List[Dict],  # С таймкодами
                "duration": float,
                "provider": str
            }
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Файл не найден: {audio_path}")
        
        print(f"🎙️ Транскрибирую: {audio_path.name}")
        print(f"   Провайдер: {self.provider}")
        
        if self.provider == "openai":
            return self._transcribe_openai(audio_path, language)
        elif self.provider == "assemblyai":
            return self._transcribe_assemblyai(audio_path, language)
        elif self.provider == "local_whisper":
            return self._transcribe_local(audio_path, language)
        else:
            raise ValueError(f"Неизвестный провайдер: {self.provider}")
    
    def _transcribe_openai(self, audio_path: Path, language: str) -> Dict:
        """Транскрипция через OpenAI Whisper API."""
        
        # Проверяем размер файла (лимит 25MB)
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        
        if file_size_mb > 24:
            print(f"   ⚠️ Файл {file_size_mb:.1f}MB > 24MB, разбиваю на части...")
            return self._transcribe_openai_chunked(audio_path, language)
        
        with open(audio_path, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
        
        return {
            "text": response.text,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text
                }
                for s in (response.segments or [])
            ],
            "duration": response.duration,
            "provider": "openai"
        }
    
    def _transcribe_openai_chunked(self, audio_path: Path, language: str) -> Dict:
        """Транскрипция длинного файла по частям."""
        
        if not PYDUB_AVAILABLE:
            raise ImportError("Для разбивки файлов нужен pydub: pip install pydub")
        
        # Загружаем аудио
        print("   📂 Загружаю аудио...")
        audio = AudioSegment.from_file(str(audio_path))
        
        total_duration = len(audio) / 1000  # в секундах
        chunk_duration = DEFAULT_CONFIG["max_chunk_duration_minutes"] * 60 * 1000  # в миллисекундах
        
        chunks = []
        for i in range(0, len(audio), chunk_duration):
            chunks.append(audio[i:i + chunk_duration])
        
        print(f"   📦 Разбито на {len(chunks)} частей")
        
        all_text = []
        all_segments = []
        time_offset = 0
        
        for i, chunk in enumerate(chunks, 1):
            print(f"   🔄 Обрабатываю часть {i}/{len(chunks)}...")
            
            # Сохраняем временный файл
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                chunk.export(tmp.name, format="mp3")
                tmp_path = tmp.name
            
            try:
                with open(tmp_path, "rb") as audio_file:
                    response = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=language,
                        response_format="verbose_json",
                        timestamp_granularities=["segment"]
                    )
                
                all_text.append(response.text)
                
                # Корректируем таймкоды
                for s in (response.segments or []):
                    all_segments.append({
                        "start": s.start + time_offset,
                        "end": s.end + time_offset,
                        "text": s.text
                    })
                
                time_offset += len(chunk) / 1000
                
            finally:
                os.unlink(tmp_path)
        
        return {
            "text": " ".join(all_text),
            "segments": all_segments,
            "duration": total_duration,
            "provider": "openai"
        }
    
    def _transcribe_assemblyai(self, audio_path: Path, language: str) -> Dict:
        """Транскрипция через AssemblyAI."""
        
        config = aai.TranscriptionConfig(
            language_code=language,
            punctuate=True,
            format_text=True
        )
        
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(str(audio_path), config=config)
        
        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"AssemblyAI ошибка: {transcript.error}")
        
        segments = []
        if transcript.words:
            # Группируем слова в сегменты по ~30 секунд
            current_segment = {"start": 0, "end": 0, "text": ""}
            for word in transcript.words:
                if word.start / 1000 - current_segment["start"] > 30:
                    if current_segment["text"]:
                        segments.append(current_segment)
                    current_segment = {
                        "start": word.start / 1000,
                        "end": word.end / 1000,
                        "text": word.text
                    }
                else:
                    current_segment["end"] = word.end / 1000
                    current_segment["text"] += " " + word.text
            
            if current_segment["text"]:
                segments.append(current_segment)
        
        return {
            "text": transcript.text,
            "segments": segments,
            "duration": transcript.audio_duration,
            "provider": "assemblyai"
        }
    
    def _transcribe_local(self, audio_path: Path, language: str) -> Dict:
        """Транскрипция через локальный Whisper."""
        
        model = DEFAULT_CONFIG["local_whisper_model"]
        
        # Создаем временную директорию для вывода
        with tempfile.TemporaryDirectory() as tmp_dir:
            cmd = [
                "whisper",
                str(audio_path),
                "--model", model,
                "--language", language,
                "--output_format", "json",
                "--output_dir", tmp_dir
            ]
            
            print(f"   🔄 Запускаю локальный Whisper (модель: {model})...")
            subprocess.run(cmd, check=True)
            
            # Читаем результат
            json_file = Path(tmp_dir) / f"{audio_path.stem}.json"
            with open(json_file, encoding="utf-8") as f:
                result = json.load(f)
        
        return {
            "text": result.get("text", ""),
            "segments": [
                {
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"]
                }
                for s in result.get("segments", [])
            ],
            "duration": result.get("segments", [{}])[-1].get("end", 0) if result.get("segments") else 0,
            "provider": "local_whisper"
        }


class SeminarProcessor:
    """Обработка транскрипций семинаров через Claude API."""
    
    def __init__(self, anthropic_api_key: str = None):
        self.api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Установите ANTHROPIC_API_KEY")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        self.base_dir = Path(__file__).parent
        self.output_dir = self.base_dir / "output"
        self.transcripts_dir = self.base_dir / "transcripts"
        self.knowledge_dir = self.base_dir / "knowledge_base"
        
        for d in [self.output_dir, self.transcripts_dir, self.knowledge_dir]:
            d.mkdir(exist_ok=True)
    
    def process_seminar(self, transcript: Dict, metadata: Dict = None,
                        mode: str = "full_analysis") -> Dict:
        """
        Обрабатывает транскрипцию семинара.
        
        Args:
            transcript: Результат транскрипции
            metadata: Доп. информация (название, дата, спикер)
            mode: Режим обработки
        """
        metadata = metadata or {}
        
        text = transcript["text"]
        duration_min = transcript.get("duration", 0) / 60
        
        print(f"\n📝 Обрабатываю семинар ({duration_min:.0f} мин)...")
        
        # Разбиваем на чанки по ~15 минут текста
        chunks = self._chunk_by_time(transcript)
        
        print(f"   📦 Разбито на {len(chunks)} частей")
        
        results = []
        
        for i, chunk in enumerate(chunks, 1):
            print(f"   🔄 Обрабатываю часть {i}/{len(chunks)}...")
            
            result = self._process_chunk(chunk, metadata, mode, i, len(chunks))
            results.append(result)
        
        # Создаем финальное саммари
        print("   📊 Создаю общее саммари...")
        final_summary = self._create_final_summary(results, metadata)
        
        return {
            "metadata": metadata,
            "transcript": transcript,
            "chunk_results": results,
            "final_summary": final_summary,
            "processed_at": datetime.now().isoformat()
        }
    
    def _chunk_by_time(self, transcript: Dict, chunk_minutes: int = 15) -> List[Dict]:
        """Разбивает транскрипцию на части по времени."""
        
        segments = transcript.get("segments", [])
        
        if not segments:
            # Если нет сегментов, разбиваем по словам
            text = transcript["text"]
            words = text.split()
            words_per_chunk = len(words) // max(1, int(transcript.get("duration", 600) / 60 / chunk_minutes))
            words_per_chunk = max(500, words_per_chunk)
            
            chunks = []
            for i in range(0, len(words), words_per_chunk):
                chunk_text = " ".join(words[i:i + words_per_chunk])
                chunks.append({
                    "text": chunk_text,
                    "start_time": "N/A",
                    "end_time": "N/A"
                })
            return chunks
        
        # Группируем сегменты по времени
        chunk_seconds = chunk_minutes * 60
        chunks = []
        current_chunk = {"text": "", "start_time": 0, "end_time": 0, "segments": []}
        
        for segment in segments:
            if segment["start"] - current_chunk["start_time"] > chunk_seconds and current_chunk["text"]:
                current_chunk["end_time"] = current_chunk["segments"][-1]["end"]
                chunks.append(current_chunk)
                current_chunk = {"text": "", "start_time": segment["start"], "end_time": 0, "segments": []}
            
            current_chunk["text"] += " " + segment["text"]
            current_chunk["segments"].append(segment)
        
        if current_chunk["text"]:
            current_chunk["end_time"] = current_chunk["segments"][-1]["end"] if current_chunk["segments"] else 0
            chunks.append(current_chunk)
        
        return chunks
    
    def _format_time(self, seconds: float) -> str:
        """Форматирует время в HH:MM:SS."""
        return str(timedelta(seconds=int(seconds)))
    
    def _process_chunk(self, chunk: Dict, metadata: Dict, mode: str,
                       chunk_num: int, total_chunks: int) -> Dict:
        """Обрабатывает один чанк через Claude."""
        
        system_prompt = """Ты - AI-ассистент программы Executive Coaching & Mentoring в СКОЛКОВО.
Твоя задача - обработать транскрипцию семинара и извлечь ключевую информацию.

КОНТЕКСТ:
- Это аудиозапись семинара по коучингу
- Устная речь: могут быть повторы, оговорки, неполные предложения
- Важно выделить суть, а не дословный текст

ПРИНЦИПЫ ИЗВЛЕЧЕНИЯ:
1. Фокусируйся на ключевых идеях и концепциях
2. Выделяй практические инструменты и техники
3. Отмечай примеры и кейсы
4. Фиксируй мощные вопросы спикера
5. Игнорируй технические паузы, повторы, междометия

ФОРМАТ ОТВЕТА:

## Ключевые темы
[3-5 главных тем этого фрагмента]

## Основные идеи
[Краткое изложение основных мыслей]

## Инструменты и техники
[Практические методы, упомянутые спикером]

## Примеры и кейсы
[Истории, примеры из практики]

## Мощные вопросы
[Вопросы, которые задавал спикер]

## Цитаты
[Яркие формулировки, которые стоит запомнить]

## Связи с другими концепциями
[Как это связано с другими темами коучинга]
"""
        
        start_time = self._format_time(chunk.get("start_time", 0)) if isinstance(chunk.get("start_time"), (int, float)) else chunk.get("start_time", "N/A")
        end_time = self._format_time(chunk.get("end_time", 0)) if isinstance(chunk.get("end_time"), (int, float)) else chunk.get("end_time", "N/A")
        
        user_message = f"""ИНФОРМАЦИЯ О СЕМИНАРЕ:
- Название: {metadata.get('title', 'Не указано')}
- Спикер: {metadata.get('speaker', 'Не указан')}
- Дата: {metadata.get('date', 'Не указана')}
- Модуль: {metadata.get('module', 'Не указан')}

ФРАГМЕНТ {chunk_num}/{total_chunks}
Время: {start_time} - {end_time}

---
ТРАНСКРИПЦИЯ:

{chunk['text']}

---

Обработай этот фрагмент семинара и извлеки ключевую информацию."""

        response = self.client.messages.create(
            model=DEFAULT_CONFIG["claude_model"],
            max_tokens=DEFAULT_CONFIG["max_output_tokens"],
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )
        
        return {
            "chunk_num": chunk_num,
            "time_range": f"{start_time} - {end_time}",
            "response": response.content[0].text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        }
    
    def _create_final_summary(self, chunk_results: List[Dict], metadata: Dict) -> str:
        """Создает итоговое саммари из всех чанков."""
        
        all_content = "\n\n---\n\n".join([
            f"ЧАСТЬ {r['chunk_num']} ({r['time_range']}):\n{r['response']}"
            for r in chunk_results
        ])
        
        system_prompt = """Ты - AI-ассистент программы Executive Coaching & Mentoring в СКОЛКОВО.
Создай итоговое саммари семинара на основе обработанных фрагментов.

ЗАДАЧА:
1. Объедини информацию из всех частей
2. Убери дублирование
3. Структурируй по важности
4. Выдели главные takeaways

ФОРМАТ:

# 📚 Саммари семинара

## 🎯 Главные идеи (3-5 пунктов)
[Самое важное из семинара]

## 🔧 Инструменты и техники
[Практические методы]

## ❓ Мощные вопросы
[Вопросы для рефлексии и практики]

## 💡 Ключевые инсайты
[Неочевидные мысли, которые стоит запомнить]

## 🔗 Связи с другими темами
[Как это связано с программой]

## ✅ Практические шаги
[Что можно применить сразу]

## 🏷️ Теги
[5-10 ключевых слов для поиска]
"""
        
        user_message = f"""ИНФОРМАЦИЯ О СЕМИНАРЕ:
- Название: {metadata.get('title', 'Не указано')}
- Спикер: {metadata.get('speaker', 'Не указан')}
- Дата: {metadata.get('date', 'Не указана')}
- Модуль: {metadata.get('module', 'Не указан')}

ОБРАБОТАННЫЕ ФРАГМЕНТЫ:

{all_content}

---

Создай итоговое структурированное саммари всего семинара."""

        response = self.client.messages.create(
            model=DEFAULT_CONFIG["claude_model"],
            max_tokens=DEFAULT_CONFIG["max_output_tokens"],
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )
        
        return response.content[0].text
    
    def save_results(self, results: Dict, output_name: str = None) -> Tuple[Path, Path]:
        """Сохраняет результаты обработки."""
        
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            title = results.get("metadata", {}).get("title", "seminar")
            title_clean = "".join(c for c in title if c.isalnum() or c in " _-")[:30]
            output_name = f"{title_clean}_{timestamp}"
        
        # Сохраняем JSON
        json_path = self.output_dir / f"{output_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Сохраняем Markdown
        md_path = self.output_dir / f"{output_name}.md"
        
        md_content = f"""# {results.get('metadata', {}).get('title', 'Семинар')}

**Спикер:** {results.get('metadata', {}).get('speaker', 'Не указан')}  
**Дата:** {results.get('metadata', {}).get('date', 'Не указана')}  
**Модуль:** {results.get('metadata', {}).get('module', 'Не указан')}  
**Длительность:** {results.get('transcript', {}).get('duration', 0) / 60:.0f} минут  
**Обработано:** {results.get('processed_at', '')}

---

{results.get('final_summary', '')}

---

## Детальная обработка по частям

"""
        
        for chunk in results.get("chunk_results", []):
            md_content += f"""
### Часть {chunk['chunk_num']} ({chunk['time_range']})

{chunk['response']}

---
"""
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        # Сохраняем чистую транскрипцию
        transcript_path = self.transcripts_dir / f"{output_name}_transcript.txt"
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(results.get("transcript", {}).get("text", ""))
        
        print(f"\n✅ Результаты сохранены:")
        print(f"   📄 JSON: {json_path}")
        print(f"   📝 Markdown: {md_path}")
        print(f"   🎙️ Транскрипт: {transcript_path}")
        
        return json_path, md_path


def process_audio_file(audio_path: str, 
                       transcription_provider: str = "openai",
                       transcription_api_key: str = None,
                       anthropic_api_key: str = None,
                       metadata: Dict = None,
                       mode: str = "full_analysis") -> Dict:
    """
    Полный пайплайн: транскрипция + обработка.
    
    Args:
        audio_path: Путь к аудио/видео файлу
        transcription_provider: openai, local_whisper, assemblyai
        transcription_api_key: API ключ для транскрипции
        anthropic_api_key: API ключ для Claude
        metadata: Информация о семинаре
        mode: Режим обработки
    """
    # Транскрипция
    transcriber = AudioTranscriber(
        provider=transcription_provider,
        api_key=transcription_api_key
    )
    
    transcript = transcriber.transcribe(audio_path)
    
    print(f"\n✅ Транскрипция завершена:")
    print(f"   📝 {len(transcript['text'])} символов")
    print(f"   ⏱️ {transcript['duration'] / 60:.1f} минут")
    
    # Обработка через Claude
    processor = SeminarProcessor(anthropic_api_key=anthropic_api_key)
    
    results = processor.process_seminar(transcript, metadata, mode)
    
    # Сохранение
    processor.save_results(results)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Транскрипция и обработка аудиозаписей семинаров"
    )
    
    parser.add_argument("audio_path", help="Путь к аудио/видео файлу")
    
    parser.add_argument(
        "--provider", "-p",
        choices=["openai", "local_whisper", "assemblyai"],
        default="openai",
        help="Провайдер транскрипции (default: openai)"
    )
    
    parser.add_argument("--title", "-t", help="Название семинара")
    parser.add_argument("--speaker", "-s", help="Спикер")
    parser.add_argument("--date", "-d", help="Дата семинара")
    parser.add_argument("--module", "-m", help="Номер модуля")
    
    parser.add_argument("--openai-key", help="OpenAI API ключ")
    parser.add_argument("--anthropic-key", help="Anthropic API ключ")
    parser.add_argument("--assemblyai-key", help="AssemblyAI API ключ")
    
    parser.add_argument(
        "--transcript-only",
        action="store_true",
        help="Только транскрипция без обработки Claude"
    )
    
    args = parser.parse_args()
    
    # Собираем метаданные
    metadata = {}
    if args.title:
        metadata["title"] = args.title
    if args.speaker:
        metadata["speaker"] = args.speaker
    if args.date:
        metadata["date"] = args.date
    if args.module:
        metadata["module"] = args.module
    
    # Определяем API ключ для транскрипции
    if args.provider == "openai":
        transcription_key = args.openai_key
    elif args.provider == "assemblyai":
        transcription_key = args.assemblyai_key
    else:
        transcription_key = None
    
    try:
        if args.transcript_only:
            # Только транскрипция
            transcriber = AudioTranscriber(
                provider=args.provider,
                api_key=transcription_key
            )
            transcript = transcriber.transcribe(args.audio_path)
            
            # Сохраняем транскрипт
            output_path = Path(args.audio_path).with_suffix(".txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcript["text"])
            
            print(f"\n✅ Транскрипт сохранен: {output_path}")
            
        else:
            # Полная обработка
            process_audio_file(
                args.audio_path,
                transcription_provider=args.provider,
                transcription_api_key=transcription_key,
                anthropic_api_key=args.anthropic_key,
                metadata=metadata
            )
    
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
