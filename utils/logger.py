import logging
import os
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from functools import wraps


class CryptoLogger:
    """
    Уніфікований логер для crypto_prediction_bot.
    Підтримує логування в консоль та файли з різними рівнями деталізації,
    ротацію логів за часом та розміром, а також спеціалізовані методи
    для логування криптовалютних операцій.
    """

    LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }

    def __init__(self, name, log_level='INFO', log_to_console=True, log_to_file=True,
                 log_dir='logs', max_file_size=10 * 1024 * 1024, backup_count=10):
        """
        Ініціалізація логера.

        Args:
            name (str): Назва логера (зазвичай назва модуля)
            log_level (str): Рівень логування ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            log_to_console (bool): Чи логувати в консоль
            log_to_file (bool): Чи логувати у файл
            log_dir (str): Директорія для файлів логів
            max_file_size (int): Максимальний розмір файлу логу в байтах
            backup_count (int): Кількість резервних копій логів
        """
        self.name = name
        self.log_level = self.LEVELS.get(log_level.upper(), logging.INFO)
        self.log_to_console = log_to_console
        self.log_to_file = log_to_file
        self.log_dir = log_dir
        self.max_file_size = max_file_size
        self.backup_count = backup_count

        # Створення логера
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        self.logger.propagate = False

        # Очищення існуючих хендлерів
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Створення форматера
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Логування в консоль
        if self.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(self.log_level)
            self.logger.addHandler(console_handler)

        # Логування в файл
        if self.log_to_file:
            # Створення директорії для логів
            os.makedirs(self.log_dir, exist_ok=True)

            # Логування за днем
            daily_log_file = os.path.join(self.log_dir, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
            daily_handler = TimedRotatingFileHandler(
                daily_log_file,
                when="midnight",
                interval=1,
                backupCount=30
            )
            daily_handler.setFormatter(formatter)
            daily_handler.setLevel(self.log_level)
            self.logger.addHandler(daily_handler)

            # Логування за розміром файлу
            size_log_file = os.path.join(self.log_dir, f"{name}.log")
            size_handler = RotatingFileHandler(
                size_log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            size_handler.setFormatter(formatter)
            size_handler.setLevel(self.log_level)
            self.logger.addHandler(size_handler)

    def debug(self, message, extra=None):
        """Логування debug-повідомлення"""
        self.logger.debug(message, extra=extra)

    def info(self, message, extra=None):
        """Логування інформаційного повідомлення"""
        self.logger.info(message, extra=extra)

    def warning(self, message, extra=None):
        """Логування попередження"""
        self.logger.warning(message, extra=extra)

    def error(self, message, exc_info=False, extra=None):
        """Логування помилки"""
        self.logger.error(message, exc_info=exc_info, extra=extra)

    def critical(self, message, exc_info=True, extra=None):
        """Логування критичної помилки"""
        self.logger.critical(message, exc_info=exc_info, extra=extra)

    def log_execution_time(self, message=""):
        """
        Декоратор для логування часу виконання функції

        Usage:
            @logger.log_execution_time("Обробка даних:")
            def process_data(data):
                # обробка даних
                return result
        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time

                log_message = f"{message} Функція {func.__name__} виконалась за {execution_time:.2f} секунд"
                self.info(log_message)

                return result

            return wrapper

        return decorator

    # Специфічні методи для крипто-проекту
    def log_api_call(self, api_name, params=None, response_status=None):
        """
        Логування API виклику

        Args:
            api_name (str): Назва API
            params (dict, optional): Параметри запиту
            response_status (int, optional): Статус відповіді
        """
        params_str = str(params) if params else ""
        status_str = f", статус: {response_status}" if response_status else ""
        self.info(f"API виклик: {api_name}, параметри: {params_str}{status_str}")

    def log_trading_signal(self, symbol, signal_type, price, quantity=None, reason=None):
        """
        Логування торгового сигналу

        Args:
            symbol (str): Символ криптовалюти
            signal_type (str): Тип сигналу (BUY, SELL)
            price (float): Ціна
            quantity (float, optional): Кількість
            reason (str, optional): Причина сигналу
        """
        extra = {
            'symbol': symbol,
            'signal_type': signal_type,
            'price': price,
            'quantity': quantity,
            'reason': reason
        }
        self.info(f"Торговий сигнал: {signal_type} для {symbol} за ціною {price}", extra=extra)

    def log_prediction(self, symbol, time_frame, predicted_value, actual_value=None, model_name=None):
        """
        Логування прогнозу моделі

        Args:
            symbol (str): Символ криптовалюти
            time_frame (str): Часовий фрейм
            predicted_value (float): Прогнозоване значення
            actual_value (float, optional): Фактичне значення
            model_name (str, optional): Назва моделі
        """
        extra = {
            'symbol': symbol,
            'time_frame': time_frame,
            'predicted_value': predicted_value,
            'actual_value': actual_value,
            'model_name': model_name
        }
        self.info(f"Прогноз для {symbol} ({time_frame}): {predicted_value}", extra=extra)

    def log_model_metrics(self, model_name, metrics):
        """
        Логування метрик моделі

        Args:
            model_name (str): Назва моделі
            metrics (dict): Словник метрик моделі
        """
        metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
        self.info(f"Метрики моделі {model_name}: {metrics_str}")

    def log_data_stats(self, data_type, symbol, count, start_date=None, end_date=None):
        """
        Логування статистики даних

        Args:
            data_type (str): Тип даних
            symbol (str): Символ криптовалюти
            count (int): Кількість записів
            start_date (str, optional): Початкова дата
            end_date (str, optional): Кінцева дата
        """
        date_range = ""
        if start_date and end_date:
            date_range = f" за період {start_date} - {end_date}"

        self.info(f"Дані {data_type} для {symbol}: {count} записів{date_range}")

    def log_websocket_event(self, event_type, symbol=None, status=None, details=None):
        """
        Логування подій веб-сокету

        Args:
            event_type (str): Тип події (connect, disconnect, message)
            symbol (str, optional): Символ криптовалюти
            status (str, optional): Статус події
            details (dict, optional): Додаткові деталі
        """
        symbol_str = f" для {symbol}" if symbol else ""
        status_str = f" (статус: {status})" if status else ""
        details_str = f": {details}" if details else ""

        self.info(f"WebSocket {event_type}{symbol_str}{status_str}{details_str}")

    def log_sentiment_analysis(self, source, sentiment_score, volume=None, keywords=None):
        """
        Логування результатів аналізу настроїв

        Args:
            source (str): Джерело даних (Twitter, Reddit, News)
            sentiment_score (float): Оцінка настрою (-1.0 до 1.0)
            volume (int, optional): Об'єм даних
            keywords (list, optional): Ключові слова
        """
        volume_str = f", об'єм: {volume}" if volume else ""
        keywords_str = f", ключові слова: {keywords}" if keywords else ""

        sentiment_type = "позитивний" if sentiment_score > 0.3 else "негативний" if sentiment_score < -0.3 else "нейтральний"

        self.info(f"Аналіз настроїв ({source}): {sentiment_type} ({sentiment_score:.2f}){volume_str}{keywords_str}")

    def log_database_operation(self, operation, table, records_affected=None, details=None):
        """
        Логування операцій з базою даних

        Args:
            operation (str): Тип операції (INSERT, UPDATE, DELETE, SELECT)
            table (str): Назва таблиці
            records_affected (int, optional): Кількість задіяних записів
            details (str, optional): Додаткові деталі
        """
        records_str = f", записів: {records_affected}" if records_affected is not None else ""
        details_str = f", деталі: {details}" if details else ""

        self.info(f"БД операція: {operation} для таблиці {table}{records_str}{details_str}")


# Словник для кешування логерів
_loggers = {}


def setup_logger(name=None, log_level='INFO', log_to_console=True, log_to_file=True,
                 log_dir='logs', max_file_size=10 * 1024 * 1024, backup_count=10):

    if name is None:
        import inspect
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        name = module.__name__ if module else 'root'

    # Перевіряємо, чи не створено вже логер з такою назвою
    if name in _loggers:
        return _loggers[name]

    # Створюємо новий логер
    logger = CryptoLogger(
        name=name,
        log_level=log_level,
        log_to_console=log_to_console,
        log_to_file=log_to_file,
        log_dir=log_dir,
        max_file_size=max_file_size,
        backup_count=backup_count
    )

    # Кешуємо логер
    _loggers[name] = logger

    return logger


def get_logger(name=None, log_level='INFO'):
    """
    Отримання існуючого або створення нового логера.

    Args:
        name (str, optional): Назва логера (якщо None, буде використано назву модуля)
        log_level (str): Рівень логування

    Returns:
        CryptoLogger: Інстанс логера
    """
    return setup_logger(name, log_level)