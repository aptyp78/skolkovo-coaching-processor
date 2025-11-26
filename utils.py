#!/usr/bin/env python3
"""
Общие утилиты для SKOLKOVO Materials Processor.
================================================
Централизованные функции для конфигурации, логирования и валидации.
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from functools import wraps

# ============================================================================
# ПУТИ И ДИРЕКТОРИИ
# ============================================================================

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"
CHUNKS_DIR = BASE_DIR / "processed_chunks"
CONFIG_PATH = BASE_DIR / "config.json"


def ensure_directories():
    """Создаёт необходимые директории."""
    for d in [OUTPUT_DIR, TRANSCRIPTS_DIR, KNOWLEDGE_DIR, CHUNKS_DIR]:
        d.mkdir(exist_ok=True)


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

DEFAULT_CONFIG = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens_per_chunk": 4000,
    "max_output_tokens": 4096,
    "overlap_tokens": 200,
    "language": "ru",
    "dpi": 150,
    "max_pages_per_request": 4,
    "rate_limit_delay": 0.5,  # Задержка между API вызовами
    "transcription_provider": "openai",
    "whisper_model": "whisper-1",
    "local_whisper_model": "large-v3",
    "max_chunk_duration_minutes": 20,
    "processing_modes": {
        "summary": "Создай структурированное саммари",
        "key_concepts": "Извлеки ключевые концепции и модели",
        "coaching_tools": "Извлеки инструменты и техники для коучинга",
        "questions": "Сгенерируй вопросы для рефлексии",
        "full_analysis": "Полный анализ: саммари + концепции + инструменты + вопросы"
    }
}


def load_config() -> Dict[str, Any]:
    """Загружает конфигурацию из config.json с fallback на defaults."""
    config = DEFAULT_CONFIG.copy()

    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                file_config = json.load(f)
            config.update(file_config)
        except (json.JSONDecodeError, IOError) as e:
            get_logger(__name__).warning(f"Ошибка загрузки config.json: {e}")

    return config


def get_model() -> str:
    """Возвращает имя модели из конфига."""
    return load_config().get("model", DEFAULT_CONFIG["model"])


# ============================================================================
# ЛОГИРОВАНИЕ
# ============================================================================

_loggers: Dict[str, logging.Logger] = {}


def setup_logging(level: int = logging.INFO) -> None:
    """Настраивает глобальное логирование."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def get_logger(name: str) -> logging.Logger:
    """Получает или создаёт логгер."""
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
    return _loggers[name]


# ============================================================================
# API КЛЮЧИ
# ============================================================================

def load_env_file(env_path: Path = None) -> None:
    """
    Безопасно загружает .env файл.
    Использует python-dotenv если доступен, иначе безопасный fallback.
    """
    if env_path is None:
        env_path = BASE_DIR / ".env"

    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        # Fallback: безопасный ручной парсинг
        _safe_load_env(env_path)


def _safe_load_env(env_path: Path) -> None:
    """Безопасный парсинг .env без python-dotenv."""
    # Белый список разрешённых переменных
    allowed_keys = {
        'ANTHROPIC_API_KEY',
        'OPENAI_API_KEY',
        'ASSEMBLYAI_API_KEY'
    }

    try:
        with open(env_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Пропускаем комментарии и пустые строки
                if not line or line.startswith("#"):
                    continue

                if "=" not in line:
                    continue

                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                # Проверяем белый список
                if key in allowed_keys:
                    os.environ[key] = value

    except IOError as e:
        get_logger(__name__).warning(f"Ошибка чтения .env: {e}")


def get_api_key(key_name: str) -> Optional[str]:
    """Получает API ключ из переменных окружения."""
    return os.environ.get(key_name)


def mask_api_key(key: str, show_last: int = 4) -> str:
    """
    Маскирует API ключ для безопасного отображения.

    Args:
        key: API ключ
        show_last: Количество символов в конце для показа

    Returns:
        Замаскированный ключ: "sk-...xxxx"
    """
    if not key:
        return "не установлен"

    if len(key) <= show_last + 3:
        return "***"

    prefix = key[:3] if key.startswith("sk-") else ""
    return f"{prefix}...{key[-show_last:]}"


def check_api_keys() -> Dict[str, Dict[str, Any]]:
    """
    Проверяет наличие API ключей.

    Returns:
        Dict с информацией о ключах (без раскрытия значений)
    """
    keys_info = {}

    for key_name in ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'ASSEMBLYAI_API_KEY']:
        key = get_api_key(key_name)
        keys_info[key_name] = {
            "present": bool(key),
            "masked": mask_api_key(key) if key else "не установлен"
        }

    return keys_info


# ============================================================================
# ВАЛИДАЦИЯ
# ============================================================================

def validate_page_range(page_range: str, max_pages: int = None) -> Tuple[int, int]:
    """
    Валидирует и парсит диапазон страниц.

    Args:
        page_range: Строка вида "1-10" или "5"
        max_pages: Максимальное количество страниц (для проверки)

    Returns:
        Tuple[start, end]: Валидный диапазон

    Raises:
        ValueError: При невалидном диапазоне
    """
    if not page_range or not page_range.strip():
        raise ValueError("Диапазон страниц не указан")

    page_range = page_range.strip()

    try:
        if "-" in page_range:
            parts = page_range.split("-")
            if len(parts) != 2:
                raise ValueError(f"Неверный формат: {page_range}. Используйте: 1-10")

            start = int(parts[0].strip())
            end = int(parts[1].strip())
        else:
            start = end = int(page_range)
    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError(f"Страницы должны быть числами: {page_range}")
        raise

    # Валидация значений
    if start < 1:
        raise ValueError(f"Начальная страница должна быть >= 1, получено: {start}")

    if end < start:
        raise ValueError(f"Конечная страница ({end}) меньше начальной ({start})")

    if max_pages is not None and end > max_pages:
        raise ValueError(f"Страница {end} превышает количество страниц в документе ({max_pages})")

    return start, end


def validate_file_path(path: str, allowed_extensions: set = None) -> Path:
    """
    Валидирует путь к файлу.

    Args:
        path: Путь к файлу
        allowed_extensions: Множество разрешённых расширений (например {'.pdf', '.mp3'})

    Returns:
        Path: Валидный путь

    Raises:
        ValueError: При невалидном пути
    """
    if not path:
        raise ValueError("Путь к файлу не указан")

    file_path = Path(path)

    if not file_path.exists():
        raise ValueError(f"Файл не найден: {path}")

    if not file_path.is_file():
        raise ValueError(f"Не является файлом: {path}")

    if allowed_extensions:
        if file_path.suffix.lower() not in allowed_extensions:
            raise ValueError(
                f"Неподдерживаемый формат: {file_path.suffix}. "
                f"Разрешены: {', '.join(allowed_extensions)}"
            )

    return file_path


# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """Простой rate limiter для API вызовов."""

    def __init__(self, delay: float = None):
        """
        Args:
            delay: Задержка между вызовами в секундах
        """
        self.delay = delay or load_config().get("rate_limit_delay", 0.5)
        self.last_call = 0.0

    def wait(self) -> None:
        """Ожидает необходимое время перед следующим вызовом."""
        elapsed = time.time() - self.last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_call = time.time()

    def __call__(self, func):
        """Декоратор для автоматического rate limiting."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.wait()
            return func(*args, **kwargs)
        return wrapper


# Глобальный rate limiter
_rate_limiter = None


def get_rate_limiter() -> RateLimiter:
    """Получает глобальный rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def rate_limited_call(func):
    """Декоратор для rate limiting API вызовов."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        get_rate_limiter().wait()
        return func(*args, **kwargs)
    return wrapper


# ============================================================================
# ИНИЦИАЛИЗАЦИЯ
# ============================================================================

def init_app():
    """Инициализация приложения: директории, логи, env."""
    setup_logging()
    ensure_directories()
    load_env_file()


# Автоматическая инициализация при импорте
ensure_directories()
