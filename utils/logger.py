import logging
import os
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class CryptoLogger:


    LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }

    def __init__(self, name, log_level='INFO', log_to_console=True, log_to_file=True,
                 log_dir='logs', max_file_size=10 * 1024 * 1024, backup_count=10):

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
        """Декоратор для логування часу виконання функції"""

        def decorator(func):
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

    def log_api_call(self, api_name, params=None):
        """Логування API виклику"""
        params_str = str(params) if params else ""
        self.info(f"API виклик: {api_name}, параметри: {params_str}")

    def log_trading_signal(self, symbol, signal_type, price, quantity=None, reason=None):
        """Логування торгового сигналу"""
        extra = {
            'symbol': symbol,
            'signal_type': signal_type,
            'price': price,
            'quantity': quantity,
            'reason': reason
        }
        self.info(f"Торговий сигнал: {signal_type} для {symbol} за ціною {price}", extra=extra)

    def log_prediction(self, symbol, time_frame, predicted_value, actual_value=None, model_name=None):
        """Логування прогнозу моделі"""
        extra = {
            'symbol': symbol,
            'time_frame': time_frame,
            'predicted_value': predicted_value,
            'actual_value': actual_value,
            'model_name': model_name
        }
        self.info(f"Прогноз для {symbol} ({time_frame}): {predicted_value}", extra=extra)

    def log_model_metrics(self, model_name, metrics):
        """Логування метрик моделі"""
        metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
        self.info(f"Метрики моделі {model_name}: {metrics_str}")

    def log_data_stats(self, data_type, symbol, count, start_date=None, end_date=None):
        """Логування статистики даних"""
        date_range = ""
        if start_date and end_date:
            date_range = f" за період {start_date} - {end_date}"

        self.info(f"Дані {data_type} для {symbol}: {count} записів{date_range}")


# Створення глобального інстансу для імпортів
def get_logger(name, log_level='INFO'):

    return CryptoLogger(name, log_level=log_level)