import pandas as pd
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from data.db import DatabaseManager
from timeseriesmodels.ModelEvaluator import ModelEvaluator
from timeseriesmodels.TimeSeriesTransformer import TimeSeriesTransformer
from timeseriesmodels.Forecaster import Forecaster
class TimeSeriesModels:

    def __init__(self, log_level=logging.INFO):

        self.db_manager = DatabaseManager()
        self.models = {}  # Словник для збереження навчених моделей
        self.transformations = {}  # Словник для збереження параметрів трансформацій
        self.modeler = ModelEvaluator()
        self.transfromer = TimeSeriesTransformer()
        # Налаштування логування
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.forecaster = Forecaster()
        # Якщо немає обробників логів, додаємо обробник для виведення в консоль
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info("TimeSeriesModels initialized")

    def load_crypto_data(self, db_manager: Any,
                         symbol: str,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         interval: str = '1d') -> pd.DataFrame:

        try:
            self.logger.info(f"Loading {symbol} data with interval {interval} from database")

            # Якщо кінцева дата не вказана, використовуємо поточну дату
            if end_date is None:
                end_date = datetime.now()
                self.logger.debug(f"End date not specified, using current date: {end_date}")

            # Використовуємо переданий db_manager або збережений в класі
            manager = db_manager if db_manager is not None else self.db_manager

            if manager is None:
                error_msg = "Database manager not available. Please provide a valid db_manager."
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Отримуємо дані з бази даних
            klines_data = manager.get_klines_processed(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date
            )

            if klines_data is None or klines_data.empty:
                self.logger.warning(f"No data found for {symbol} with interval {interval}")
                return pd.DataFrame()

            # Перевіряємо наявність часового індексу
            if not isinstance(klines_data.index, pd.DatetimeIndex):
                # Шукаємо колонку з часовим індексом (timestamp, time, date, etc.)
                time_cols = [col for col in klines_data.columns if any(
                    time_str in col.lower() for time_str in ['time', 'date', 'timestamp'])]

                if time_cols:
                    # Використовуємо першу знайдену колонку часу
                    klines_data = klines_data.set_index(pd.DatetimeIndex(pd.to_datetime(klines_data[time_cols[0]])))
                    self.logger.info(f"Set index using column: {time_cols[0]}")
                else:
                    self.logger.warning("No time column found in data. Using default index.")

            # Сортуємо дані за часовим індексом
            klines_data = klines_data.sort_index()

            # Виводимо інформацію про отримані дані
            self.logger.info(f"Loaded {len(klines_data)} records for {symbol} "
                             f"from {klines_data.index.min()} to {klines_data.index.max()}")

            return klines_data

        except Exception as e:
            self.logger.error(f"Error loading crypto data: {str(e)}")
            raise

    def save_forecast_to_db(self, db_manager: Any, symbol: str,
                            forecast_data: pd.Series, model_key: str) -> bool:

        try:
            self.logger.info(f"Saving forecast for {symbol} using model {model_key}")

            if forecast_data is None or len(forecast_data) == 0:
                self.logger.error("No forecast data provided")
                return False

            # Перевіряємо чи forecast_data є pd.Series
            if not isinstance(forecast_data, pd.Series):
                try:
                    forecast_data = pd.Series(forecast_data)
                    self.logger.warning("Converted forecast data to pandas Series")
                except Exception as convert_error:
                    self.logger.error(f"Could not convert forecast data to pandas Series: {str(convert_error)}")
                    return False

            # Використовуємо переданий db_manager або збережений в класі
            manager = db_manager if db_manager is not None else self.db_manager

            if manager is None:
                error_msg = "Database manager not available. Please provide a valid db_manager."
                self.logger.error(error_msg)
                return False

            # Перевіряємо наявність моделі в базі даних
            model_info = manager.get_model_by_key(model_key)

            if model_info is None:
                self.logger.error(f"Model with key {model_key} not found in database")
                return False

            # Перетворюємо дані прогнозу у формат для збереження
            forecast_dict = {
                "model_key": model_key,
                "symbol": symbol,
                "timestamp": datetime.now(),
                "forecast_data": {
                    timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp): value
                    for timestamp, value in forecast_data.items()
                },
                "forecast_horizon": len(forecast_data),
                "forecast_start": forecast_data.index[0].isoformat() if isinstance(forecast_data.index[0], datetime)
                else str(forecast_data.index[0]),
                "forecast_end": forecast_data.index[-1].isoformat() if isinstance(forecast_data.index[-1], datetime)
                else str(forecast_data.index[-1])
            }

            # Зберігаємо прогноз у базі даних
            success = manager.save_model_forecasts(model_key, forecast_dict)

            if success:
                self.logger.info(f"Successfully saved forecast for {symbol} using model {model_key}")
            else:
                self.logger.error(f"Failed to save forecast for {symbol}")

            return success

        except Exception as e:
            self.logger.error(f"Error saving forecast to database: {str(e)}")
            return False

    def load_forecast_from_db(self, db_manager: Any, symbol: str,
                              model_key: str) -> Optional[pd.Series]:
        try:
            self.logger.info(f"Loading forecast for {symbol} from model {model_key}")

            # Використовуємо переданий db_manager або збережений в класі
            manager = db_manager if db_manager is not None else self.db_manager

            if manager is None:
                error_msg = "Database manager not available. Please provide a valid db_manager."
                self.logger.error(error_msg)
                return None

            # Перевіряємо наявність моделі в базі даних
            model_info = manager.get_model_by_key(model_key)

            if model_info is None:
                self.logger.warning(f"Model with key {model_key} not found in database")
                return None

            # Отримуємо прогнози з бази даних
            forecast_dict = manager.get_model_forecasts(model_key)

            if not forecast_dict:
                self.logger.warning(f"No forecasts found for model {model_key}")
                return None

            # Перевіряємо, чи є прогнози для заданого символу
            if symbol.upper() != forecast_dict.get('symbol', '').upper():
                self.logger.warning(f"Forecast for symbol {symbol} not found in model {model_key}")
                return None

            # Перетворюємо словник прогнозів на pd.Series
            try:
                forecast_data = forecast_dict.get('forecast_data', {})

                # Перетворюємо ключі на datetime, якщо вони є датами
                index = []
                values = []

                for timestamp_str, value in forecast_data.items():
                    try:
                        # Спробуємо перетворити на datetime
                        timestamp = pd.to_datetime(timestamp_str)
                    except:
                        # Якщо не вийшло, використовуємо як є
                        timestamp = timestamp_str

                    index.append(timestamp)
                    values.append(float(value))

                # Створюємо pandas Series з правильним індексом
                forecast_series = pd.Series(values, index=index)

                # Сортуємо за індексом
                forecast_series = forecast_series.sort_index()

                self.logger.info(f"Successfully loaded forecast with {len(forecast_series)} points for {symbol}")

                return forecast_series

            except Exception as e:
                self.logger.error(f"Error converting forecast data to Series: {str(e)}")
                return None

        except Exception as e:
            self.logger.error(f"Error loading forecast from database: {str(e)}")
            return None

    def get_available_crypto_symbols(self, db_manager: Any) -> List[str]:

        try:
            self.logger.info("Getting available cryptocurrency symbols from database")

            # Використовуємо переданий db_manager або збережений в класі
            manager = db_manager if db_manager is not None else self.db_manager

            if manager is None:
                error_msg = "Database manager not available. Please provide a valid db_manager."
                self.logger.error(error_msg)
                return []

            # Отримуємо список доступних символів
            symbols = manager.get_available_symbols()

            if symbols is None:
                self.logger.warning("No symbols returned from database")
                return []

            # Перевіряємо, що отримані дані є списком
            if not isinstance(symbols, list):
                try:
                    symbols = list(symbols)
                except Exception as e:
                    self.logger.error(f"Could not convert symbols to list: {str(e)}")
                    return []

            # Перевіряємо, що символи не порожні
            symbols = [s for s in symbols if s]

            # Видаляємо дублікати і сортуємо
            symbols = sorted(set(symbols))

            self.logger.info(f"Found {len(symbols)} available cryptocurrency symbols")

            return symbols

        except Exception as e:
            self.logger.error(f"Error getting available cryptocurrency symbols: {str(e)}")
            return []

    def get_last_update_time(self, db_manager: Any, symbol: str,
                             interval: str = '1d') -> Optional[datetime]:

        self.logger.info(f"Getting last update time for {symbol} with interval {interval}")

        if db_manager is None:
            self.logger.error("Database manager is not provided")
            return None

        try:
            # Перевіряємо, чи переданий db_manager був заданий при ініціалізації класу
            # якщо ні, використовуємо переданий
            db = self.db_manager if self.db_manager is not None else db_manager

            # Припускаємо, що в db_manager є метод для отримання останніх даних свічок
            latest_kline = db.get_latest_kline(symbol=symbol, interval=interval)

            if latest_kline is not None and hasattr(latest_kline, 'timestamp'):
                # Якщо отримали дані і є відмітка часу
                self.logger.info(f"Last update time for {symbol} ({interval}): {latest_kline.timestamp}")
                return latest_kline.timestamp

            # Якщо немає прямого методу, спробуємо отримати через оброблені свічки
            klines_data = db.get_klines_processed(
                symbol=symbol,
                interval=interval,
                limit=1,  # Беремо тільки одну (останню) свічку
                sort_order="DESC"  # Сортуємо за спаданням дати
            )

            if klines_data is not None and not klines_data.empty:
                # Отримуємо індекс останньої свічки (який повинен бути datetime)
                if isinstance(klines_data.index[0], datetime):
                    last_update = klines_data.index[0]
                else:
                    # Якщо індекс не datetime, спробуємо знайти стовпець з часовою міткою
                    for col in ['timestamp', 'time', 'date', 'datetime']:
                        if col in klines_data.columns:
                            last_update = klines_data[col].iloc[0]
                            if not isinstance(last_update, datetime):
                                # Конвертуємо в datetime, якщо це не datetime
                                if isinstance(last_update, (int, float)):
                                    # Припускаємо, що це UNIX timestamp в мілісекундах або секундах
                                    if last_update > 1e11:  # Якщо в мілісекундах
                                        last_update = datetime.fromtimestamp(last_update / 1000)
                                    else:  # Якщо в секундах
                                        last_update = datetime.fromtimestamp(last_update)
                                else:
                                    # Якщо це строка, пробуємо парсити
                                    try:
                                        from dateutil import parser
                                        last_update = parser.parse(str(last_update))
                                    except:
                                        self.logger.warning(f"Cannot parse datetime from {last_update}")
                                        continue
                            break
                    else:
                        self.logger.warning(f"No datetime column found in klines data for {symbol}")
                        return None

                self.logger.info(f"Last update time for {symbol} ({interval}): {last_update}")
                return last_update

            self.logger.warning(f"No data found for {symbol} with interval {interval}")
            return None

        except Exception as e:
            self.logger.error(f"Error getting last update time for {symbol}: {str(e)}")
            return None

    def batch_process_symbols(self, db_manager: Any, symbols: List[str],
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None,
                              interval: str = '1d') -> Dict[str, Dict]:

        self.logger.info(f"Starting batch processing for {len(symbols)} symbols")

        if db_manager is None:
            self.logger.error("Database manager is not provided")
            return {"status": "error", "message": "Database manager is not provided"}

        # Ініціалізуємо словник для результатів
        results = {}

        # Встановлюємо значення за замовчуванням для дат
        if end_date is None:
            end_date = datetime.now()
            self.logger.info(f"End date not provided, using current time: {end_date}")

        # Обробляємо кожен символ
        for symbol in symbols:
            self.logger.info(f"Processing symbol: {symbol}")

            try:
                # Перевіряємо, чи є дані для цього символу
                if not self._check_symbol_data_available(db_manager, symbol, interval):
                    self.logger.warning(f"No data available for {symbol}, skipping")
                    results[symbol] = {
                        "status": "error",
                        "message": f"No data available for {symbol}",
                        "timestamp": datetime.now()
                    }
                    continue

                # Якщо початкова дата не вказана, отримуємо останню дату оновлення
                # і віднімаємо певний період (наприклад, 365 днів для денних даних)
                if start_date is None:
                    last_update = self.get_last_update_time(db_manager, symbol, interval)
                    if last_update is not None:
                        if interval == '1d':
                            start_date = last_update - timedelta(days=365)  # Рік даних для денного інтервалу
                        elif interval == '1h':
                            start_date = last_update - timedelta(days=30)  # 30 днів для годинного інтервалу
                        elif interval in ['15m', '5m', '1m']:
                            start_date = last_update - timedelta(days=7)  # Тиждень для хвилинних інтервалів
                        else:
                            start_date = last_update - timedelta(days=180)  # Півроку за замовчуванням

                        self.logger.info(f"Calculated start date for {symbol}: {start_date}")
                    else:
                        self.logger.warning(f"Cannot determine last update time for {symbol}, using default")
                        # За замовчуванням, беремо дані за останній рік
                        start_date = end_date - timedelta(days=365)

                # Завантажуємо дані для аналізу
                data = self.load_crypto_data(
                    db_manager=db_manager,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval
                )

                if data is None or data.empty:
                    self.logger.warning(f"No data loaded for {symbol}, skipping")
                    results[symbol] = {
                        "status": "error",
                        "message": f"No data loaded for {symbol}",
                        "timestamp": datetime.now()
                    }
                    continue

                # Вибираємо цільову колонку для аналізу (зазвичай 'close')
                target_column = 'close'
                if target_column not in data.columns:
                    # Шукаємо альтернативу, якщо 'close' немає
                    possible_columns = ['Close', 'price', 'Price', 'value', 'Value']
                    for col in possible_columns:
                        if col in data.columns:
                            target_column = col
                            break
                    else:
                        # Якщо немає відповідної колонки, використовуємо першу колонку з числовими даними
                        for col in data.columns:
                            if pd.api.types.is_numeric_dtype(data[col]):
                                target_column = col
                                break
                        else:
                            self.logger.error(f"No suitable numeric column found for {symbol}")
                            results[symbol] = {
                                "status": "error",
                                "message": f"No suitable numeric column found for {symbol}",
                                "timestamp": datetime.now()
                            }
                            continue

                self.logger.info(f"Using column '{target_column}' for analysis of {symbol}")

                # Запускаємо автоматичне прогнозування
                forecast_result = self.forecaster.run_auto_forecast(
                    data=data[target_column],
                    test_size=0.2,  # 20% даних для тестування
                    forecast_steps=24,  # Прогноз на 24 періоди вперед
                    symbol=symbol
                )

                # Зберігаємо результати прогнозування
                if forecast_result.get("status") == "success" and "model_key" in forecast_result:
                    model_key = forecast_result["model_key"]

                    # Зберігаємо комплексну інформацію про модель в БД
                    if self.db_manager is not None:
                        try:
                            self.db_manager.save_complete_model(model_key, forecast_result.get("model_info", {}))
                            self.logger.info(f"Model for {symbol} saved to database with key {model_key}")
                        except Exception as db_error:
                            self.logger.error(f"Error saving model for {symbol} to database: {str(db_error)}")

                    # Додаємо результат до загального словника
                    results[symbol] = {
                        "status": "success",
                        "message": f"Successfully processed {symbol}",
                        "model_key": model_key,
                        "timestamp": datetime.now(),
                        **forecast_result
                    }
                else:
                    # Якщо прогнозування не вдалося, додаємо інформацію про помилку
                    results[symbol] = {
                        "status": "error",
                        "message": forecast_result.get("message", f"Error processing {symbol}"),
                        "timestamp": datetime.now()
                    }

            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {str(e)}")
                results[symbol] = {
                    "status": "error",
                    "message": f"Exception: {str(e)}",
                    "timestamp": datetime.now()
                }

        # Додаємо загальну статистику
        success_count = sum(1 for symbol, result in results.items() if result.get("status") == "success")
        error_count = len(symbols) - success_count

        summary = {
            "total_symbols": len(symbols),
            "success_count": success_count,
            "error_count": error_count,
            "success_rate": success_count / len(symbols) if symbols else 0,
            "processed_at": datetime.now()
        }

        results["_summary"] = summary

        self.logger.info(f"Batch processing completed. Success: {success_count}, Errors: {error_count}")

        return results

    def _check_symbol_data_available(self, db_manager: Any, symbol: str, interval: str) -> bool:

        try:
            # Перевіряємо, чи є такий символ у списку доступних
            available_symbols = self.get_available_crypto_symbols(db_manager)
            if symbol not in available_symbols:
                self.logger.warning(f"Symbol {symbol} not in available symbols list")
                return False

            # Перевіряємо, чи є хоча б деякі дані для цього символу
            last_update = self.get_last_update_time(db_manager, symbol, interval)
            if last_update is None:
                self.logger.warning(f"No last update time for {symbol}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking data availability for {symbol}: {str(e)}")
            return False
def main():
    import pprint

    symbol = "BTCUSDT"
    interval = "1d"
    forecast_steps = 7

    model = TimeSeriesModels()
    db = model.db_manager

    if db is None:
        print(" Менеджер бази даних не налаштований.")
        return

    try:
        df = db.get_klines_processed(symbol, interval)
    except Exception as e:
        print(f" Помилка при отриманні даних: {e}")
        return

    if df is None or df.empty or "close" not in df.columns:
        print(" Немає даних для тренування моделі.")
        return

    price_series = df["close"]

    stat_info = model.check_stationarity(price_series)
    if not stat_info["is_stationary"]:
        price_series = self.transformer.difference_series(price_series)

    #  ARIMA
    arima_key = None
    arima_params = model.find_optimal_params(price_series, seasonal=False)
    if arima_params["status"] == "success":
        arima_result = model.fit_arima(price_series, arima_params["parameters"]["order"], symbol=symbol)
        arima_key = arima_result["model_key"]
        print(f" ARIMA збережено як {arima_key}")
    else:
        print("️ ARIMA: не вдалося підібрати параметри")

    #  SARIMA
    sarima_key = None
    sarima_params = model.find_optimal_params(price_series, seasonal=True)
    if sarima_params["status"] == "success":
        sarima_result = model.fit_sarima(
            price_series,
            order=sarima_params["parameters"]["order"],
            seasonal_order=sarima_params["parameters"]["seasonal_order"],
            symbol=symbol
        )
        sarima_key = sarima_result["model_key"]
        print(f" SARIMA збережено як {sarima_key}")
    else:
        print(" SARIMA: не вдалося підібрати параметри")

    #  Порівняння моделей
    if arima_key and sarima_key:
        aic_arima = arima_result["model_info"]["stats"]["aic"]
        aic_sarima = sarima_result["model_info"]["stats"]["aic"]
        better_key = arima_key if aic_arima < aic_sarima else sarima_key
        print(f"🏆 Краща модель за AIC: {better_key}")
    else:
        better_key = arima_key or sarima_key

    #  Прогнозування
    if better_key:
        forecast = model.forecast(better_key, steps=forecast_steps)
        if not forecast.empty:
            print("\n Прогноз:")
            pprint.pprint(forecast.to_dict())
        else:
            print(" Прогнозування не вдалося.")
    else:
        print(" Немає доступної моделі для прогнозування.")

    print("\n Завершено.")

if __name__ == "__main__":
    main()
