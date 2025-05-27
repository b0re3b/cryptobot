import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import logging
from data.db import DatabaseManager
from featureengineering.DimensionalityReducer import DimensionalityReducer
from featureengineering.CrossFeatures import CrossFeatures
from featureengineering.StatisticalFeatures import StatisticalFeatures
from featureengineering.TechnicalFeatures import TechnicalFeatures
from featureengineering.TimeFeatures import TimeFeatures
from utils.logger import CryptoLogger


class FeatureEngineering:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.supported_symbols = self.db_manager.supported_symbols
        self.logger = CryptoLogger('Feature Engineering')
        self.logger.info("Ініціалізація Feature Engineering Pipeline...")
        self.ready = True

        # Ініціалізація підмодулів
        self.dimensionality_reducer = DimensionalityReducer()
        self.cross_features = CrossFeatures()
        self.statistical_features = StatisticalFeatures()
        self.technical_features = TechnicalFeatures()
        self.time_features = TimeFeatures()

    def create_target_variable(self, data: pd.DataFrame,
                               price_column: str = 'close',
                               horizon: int = 1,
                               target_type: str = 'return') -> pd.DataFrame:

        self.logger.info(f"Створення цільової змінної типу '{target_type}' з горизонтом {horizon}")

        # Створюємо копію, щоб не модифікувати оригінальні дані
        result_df = data.copy()

        # Перевіряємо, що price_column існує в даних
        if price_column not in result_df.columns:
            self.logger.error(f"Стовпець {price_column} не знайдено в даних")
            raise ValueError(f"Стовпець {price_column} не знайдено в даних")

        # Перевіряємо, що індекс часовий для правильного зсуву
        if not isinstance(result_df.index, pd.DatetimeIndex):
            self.logger.warning("Індекс даних не є часовим (DatetimeIndex). Цільова змінна може бути неточною.")

        # Перевіряємо наявність пропущених значень у стовпці ціни
        if result_df[price_column].isna().any():
            self.logger.warning(f"Стовпець {price_column} містить NaN значення, вони будуть заповнені")
            result_df[price_column] = result_df[price_column].fillna(method='ffill').fillna(method='bfill')

        # Створюємо цільову змінну в залежності від типу
        if target_type == 'return':
            # Процентна зміна ціни через horizon періодів
            target_name = f'target_return_{horizon}p'
            result_df[target_name] = result_df[price_column].pct_change(periods=-horizon).shift(horizon)
            self.logger.info(f"Створено цільову змінну '{target_name}' як процентну зміну ціни")

        elif target_type == 'log_return':
            # Логарифмічна зміна ціни
            target_name = f'target_log_return_{horizon}p'
            result_df[target_name] = np.log(result_df[price_column].shift(-horizon) / result_df[price_column])
            self.logger.info(f"Створено цільову змінну '{target_name}' як логарифмічну зміну ціни")

        elif target_type == 'direction':
            # Напрямок зміни ціни (1 - ріст, 0 - падіння)
            target_name = f'target_direction_{horizon}p'
            future_price = result_df[price_column].shift(-horizon)
            result_df[target_name] = np.where(future_price > result_df[price_column], 1, 0)
            self.logger.info(f"Створено цільову змінну '{target_name}' як напрямок зміни ціни")

        elif target_type == 'volatility':
            # Майбутня волатільність як стандартне відхилення прибутковості за період
            target_name = f'target_volatility_{horizon}p'
            # Розраховуємо логарифмічну прибутковість
            log_returns = np.log(result_df[price_column] / result_df[price_column].shift(1))
            # Розраховуємо волатільність за наступні horizon періодів
            result_df[target_name] = log_returns.rolling(window=horizon).std().shift(-horizon)
            self.logger.info(f"Створено цільову змінну '{target_name}' як майбутню волатільність")

        elif target_type == 'price':
            # Майбутня ціна
            target_name = f'target_price_{horizon}p'
            result_df[target_name] = result_df[price_column].shift(-horizon)
            self.logger.info(f"Створено цільову змінну '{target_name}' як майбутню ціну")

        elif target_type == 'range':
            # Діапазон зміни ціни (high-low) за наступні horizon періодів
            target_name = f'target_range_{horizon}p'
            # Для точного розрахунку діапазону потрібні high і low колонки
            if 'high' in result_df.columns and 'low' in result_df.columns:
                # Знаходимо максимальне high і мінімальне low за наступні horizon періодів
                high_values = result_df['high'].rolling(window=horizon, min_periods=1).max().shift(-horizon)
                low_values = result_df['low'].rolling(window=horizon, min_periods=1).min().shift(-horizon)
                result_df[target_name] = (high_values - low_values) / result_df[price_column]
                self.logger.info(f"Створено цільову змінну '{target_name}' як відносний діапазон ціни")
            else:
                self.logger.warning(
                    "Колонки 'high' або 'low' відсутні, використовуємо близьку ціну для розрахунку діапазону")
                # Використовуємо амплітуду зміни ціни close
                price_max = result_df[price_column].rolling(window=horizon, min_periods=1).max().shift(-horizon)
                price_min = result_df[price_column].rolling(window=horizon, min_periods=1).min().shift(-horizon)
                result_df[target_name] = (price_max - price_min) / result_df[price_column]
                self.logger.info(f"Створено цільову змінну '{target_name}' як відносний діапазон ціни")
        else:
            self.logger.error(f"Невідомий тип цільової змінної: {target_type}")
            raise ValueError(
                f"Невідомий тип цільової змінної: {target_type}. Допустимі значення: 'return', 'log_return', 'direction', 'volatility', 'price', 'range'")

        # Заповнюємо NaN значення в цільовій змінній
        if result_df[target_name].isna().any():
            self.logger.warning(
                f"Цільова змінна {target_name} містить {result_df[target_name].isna().sum()} NaN значень")
            # Для цільових змінних краще видалити рядки з NaN, ніж заповнювати їх
            if target_type in ['return', 'log_return', 'price', 'range', 'volatility']:
                # Для числових цільових змінних можна спробувати заповнити медіаною
                # Але це не рекомендується для навчання моделей
                median_val = result_df[target_name].median()
                result_df[target_name] = result_df[target_name].fillna(median_val)
                self.logger.warning(f"NaN значення в цільовій змінній заповнені медіаною: {median_val}")
            elif target_type == 'direction':
                # Для бінарної класифікації можна заповнити найбільш поширеним класом
                mode_val = result_df[target_name].mode()[0] if len(result_df[target_name].mode()) > 0 else 0
                result_df[target_name] = result_df[target_name].fillna(mode_val)
                self.logger.warning(f"NaN значення в цільовій змінній заповнені модою: {mode_val}")

        return result_df

    def select_features(self, X: pd.DataFrame, y: pd.Series,
                        n_features: Optional[int] = None,
                        method: str = 'f_regression') -> Tuple[pd.DataFrame, List[str]]:
        """Відбір найважливіших ознак."""
        return self.dimensionality_reducer.select_features(X, y, n_features, method)

    def reduce_dimensions(self, data: pd.DataFrame,
                          n_components: Optional[int] = None,
                          method: str = 'pca') -> Tuple[pd.DataFrame, object]:
        """Зменшення розмірності даних."""
        return self.dimensionality_reducer.reduce_dimensions(data, n_components, method)

    def create_polynomial_features(self, data: pd.DataFrame,
                                   columns: Optional[List[str]] = None,
                                   degree: int = 2,
                                   interaction_only: bool = False) -> pd.DataFrame:
        """Створення поліноміальних ознак."""
        return self.dimensionality_reducer.create_polynomial_features(data, columns, degree, interaction_only)

    def create_cluster_features(self, data: pd.DataFrame,
                                n_clusters: int = 5,
                                method: str = 'kmeans') -> pd.DataFrame:
        """Створення ознак на основі кластеризації."""
        return self.dimensionality_reducer.create_cluster_features(data, n_clusters, method)

    def create_ratio_features(self, data: pd.DataFrame,
                              numerators: List[str],
                              denominators: List[str],
                              clip_percentiles: Tuple[float, float] = (0.01, 0.99)) -> pd.DataFrame:
        """Створення ознак-співвідношень."""
        return self.cross_features.create_ratio_features(data, numerators, denominators, clip_percentiles)

    def create_crossover_features(self, data: pd.DataFrame,
                                  fast_columns: List[str],
                                  slow_columns: List[str],
                                  slope_periods: int = 3) -> pd.DataFrame:
        """Створення ознак перетинів індикаторів."""
        return self.cross_features.create_crossover_features(data, fast_columns, slow_columns, slope_periods)

    def create_volatility_features(self, data: pd.DataFrame,
                                   price_column: str = 'close',
                                   window_sizes: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """Створення ознак волатільності."""
        return self.statistical_features.create_volatility_features(data, price_column, window_sizes)

    def create_return_features(self, data: pd.DataFrame,
                               price_column: str = 'close',
                               periods: List[int] = [1, 3, 5, 7, 14]) -> pd.DataFrame:
        """Створення ознак прибутковості."""
        return self.statistical_features.create_return_features(data, price_column, periods)

    def create_volume_features(self, data: pd.DataFrame,
                               window_sizes: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """Створення ознак на основі об'єму."""
        return self.statistical_features.create_volume_features(data, window_sizes)

    def create_technical_features(self, data: pd.DataFrame,
                                  indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """Створення технічних індикаторів з додаткової обробкою помилок."""
        try:
            # Перевіряємо, чи є необхідні колонки
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                self.logger.warning(f"Відсутні колонки для технічних індикаторів: {missing_columns}")
                # Створюємо мінімальний набір ознак з доступних даних
                result_df = data.copy()

                # Якщо є хоча б колонка close, створюємо базові індикатори
                if 'close' in data.columns:
                    result_df['close_sma_10'] = data['close'].rolling(window=10).mean()
                    result_df['close_sma_20'] = data['close'].rolling(window=20).mean()

                return result_df

            return self.technical_features.create_technical_features(data, indicators)

        except Exception as e:
            self.logger.error(f"Помилка в create_technical_features: {str(e)}")
            # Повертаємо оригінальні дані при помилці
            return data.copy()

    def create_candle_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Створення ознак на основі патернів свічок з обробкою помилок."""
        try:
            return self.technical_features.create_candle_pattern_features(data)
        except Exception as e:
            self.logger.error(f"Помилка в create_candle_pattern_features: {str(e)}")
            return data.copy()

    def create_custom_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Створення спеціальних індикаторів для криптовалют з обробкою помилок."""
        try:
            return self.technical_features.create_custom_indicators(data)
        except Exception as e:
            self.logger.error(f"Помилка в create_custom_indicators: {str(e)}")
            return data.copy()

    def create_lagged_features(self, data: pd.DataFrame,
                               columns: Optional[List[str]] = None,
                               lag_periods: List[int] = [1, 3, 5, 7, 14]) -> pd.DataFrame:
        """Створення лагових ознак з обробкою помилок."""
        try:
            return self.time_features.create_lagged_features(data, columns, lag_periods)
        except Exception as e:
            self.logger.error(f"Помилка в create_lagged_features: {str(e)}")
            return data.copy()

    def create_rolling_features(self, data: pd.DataFrame,
                                columns: Optional[List[str]] = None,
                                window_sizes: List[int] = [5, 10, 20, 50],
                                functions: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
        """Створення ознак на основі ковзного вікна з обробкою помилок."""
        try:
            return self.time_features.create_rolling_features(data, columns, window_sizes, functions)
        except Exception as e:
            self.logger.error(f"Помилка в create_rolling_features: {str(e)}")
            return data.copy()

    def create_ewm_features(self, data: pd.DataFrame,
                            columns: Optional[List[str]] = None,
                            spans: List[int] = [5, 10, 20, 50],
                            functions: List[str] = ['mean', 'std']) -> pd.DataFrame:
        """Створення ознак на основі експоненційно зваженого вікна з обробкою помилок."""
        try:
            return self.time_features.create_ewm_features(data, columns, spans, functions)
        except Exception as e:
            self.logger.error(f"Помилка в create_ewm_features: {str(e)}")
            return data.copy()

    def create_datetime_features(self, data: pd.DataFrame, cyclical: bool = True) -> pd.DataFrame:
        """Створення ознак на основі дати й часу з обробкою помилок."""
        try:
            return self.time_features.create_datetime_features(data, cyclical)
        except Exception as e:
            self.logger.error(f"Помилка в create_datetime_features: {str(e)}")
            return data.copy()

    def prepare_features_pipeline(self, data: pd.DataFrame,
                                  target_column: str = 'close',
                                  horizon: int = 1,
                                  feature_groups: Optional[List[str]] = None,
                                  reduce_dimensions: bool = False,
                                  n_features: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Уніфікований пайплайн підготовки ознак з покращеною обробкою помилок.

        Args:
            data: Вхідний DataFrame
            target_column: Назва цільової колонки
            horizon: Горизонт прогнозування
            feature_groups: Список груп ознак для створення
            reduce_dimensions: Чи виконувати зменшення розмірності
            n_features: Кількість ознак для відбору

        Returns:
            Кортеж (DataFrame з ознаками, Series з цільовою змінною)
        """
        self.logger.info("Запуск уніфікованого пайплайну підготовки даних...")

        # Стандартні групи ознак
        standard_groups = [
            'lagged', 'rolling', 'ewm', 'returns', 'technical', 'volatility',
            'ratio', 'crossover', 'datetime', 'candle_patterns', 'custom_indicators',
            'volume', 'polynomial', 'cluster'
        ]

        if feature_groups is None:
            feature_groups = standard_groups
            self.logger.info(f"Використовуються всі стандартні групи ознак: {feature_groups}")
        else:
            invalid_groups = [group for group in feature_groups if group not in standard_groups]
            if invalid_groups:
                self.logger.warning(f"Невідомі групи ознак {invalid_groups} будуть пропущені")
                feature_groups = [group for group in feature_groups if group in standard_groups]

        # Створюємо копію даних
        result_df = data.copy()
        initial_feature_count = len(result_df.columns)

        # Послідовне створення ознак з покращеною обробкою помилок
        for group in feature_groups:
            try:
                self.logger.info(f"Обробка групи ознак: {group}")
                previous_shape = result_df.shape

                if group == 'lagged':
                    result_df = self.create_lagged_features(result_df)
                elif group == 'rolling':
                    result_df = self.create_rolling_features(result_df)
                elif group == 'ewm':
                    result_df = self.create_ewm_features(result_df)
                elif group == 'returns':
                    result_df = self.create_return_features(result_df, target_column)
                elif group == 'technical':
                    result_df = self.create_technical_features(result_df)
                elif group == 'volatility':
                    result_df = self.create_volatility_features(result_df, target_column)
                elif group == 'ratio':
                    price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in result_df.columns]
                    if 'volume' in result_df.columns:
                        price_cols.append('volume')
                    if len(price_cols) >= 2:  # Потрібно мінімум 2 колонки для співвідношень
                        result_df = self.create_ratio_features(result_df, price_cols, price_cols)
                    else:
                        self.logger.warning(f"Недостатньо колонок для створення співвідношень: {price_cols}")
                elif group == 'crossover':
                    sma_cols = [col for col in result_df.columns if col.startswith('sma_')]
                    ema_cols = [col for col in result_df.columns if col.startswith('ema_')]
                    if sma_cols and ema_cols:
                        result_df = self.create_crossover_features(result_df, ema_cols, sma_cols)
                    else:
                        self.logger.warning(
                            f"Недостатньо SMA/EMA колонок для перетинів: SMA={len(sma_cols)}, EMA={len(ema_cols)}")
                elif group == 'datetime':
                    result_df = self.create_datetime_features(result_df)
                elif group == 'candle_patterns':
                    result_df = self.create_candle_pattern_features(result_df)
                elif group == 'custom_indicators':
                    result_df = self.create_custom_indicators(result_df)
                elif group == 'volume':
                    if 'volume' in result_df.columns:
                        result_df = self.create_volume_features(result_df)
                    else:
                        self.logger.warning("Колонка 'volume' відсутня для створення об'ємних ознак")
                elif group == 'polynomial':
                    result_df = self.create_polynomial_features(result_df)
                elif group == 'cluster':
                    result_df = self.create_cluster_features(result_df)

                # Логування змін
                new_features = result_df.shape[1] - previous_shape[1]
                if new_features > 0:
                    self.logger.info(f"Група {group}: додано {new_features} нових ознак")
                else:
                    self.logger.warning(f"Група {group}: нові ознаки не додані")

            except Exception as e:
                self.logger.error(f"Критична помилка при створенні групи ознак {group}: {str(e)}")
                # Продовжуємо з наступною групою замість зупинки всього процесу

        # Створюємо цільову змінну
        try:
            self.logger.info(f"Створення цільової змінної для {target_column} з горизонтом {horizon}")
            result_df = self.create_target_variable(result_df, target_column, horizon, 'return')
            target_name = f'target_return_{horizon}p'
            target = result_df[target_name].copy()
            result_df.drop(target_name, axis=1, inplace=True)
        except Exception as e:
            self.logger.error(f"Помилка при створенні цільової змінної: {str(e)}")
            raise

        # Видаляємо рядки з NaN у цільовій змінній
        valid_idx = ~target.isna()
        if not valid_idx.all():
            self.logger.info(f"Видалено {sum(~valid_idx)} рядків з NaN у цільовій змінній")
            result_df = result_df.loc[valid_idx]
            target = target.loc[valid_idx]

        # Заповнюємо NaN у ознаках
        nan_cols = result_df.columns[result_df.isna().any()].tolist()
        if nan_cols:
            self.logger.warning(f"Заповнення NaN у {len(nan_cols)} стовпцях")
            for col in nan_cols:
                try:
                    if result_df[col].dtype == 'object':
                        mode_values = result_df[col].mode()
                        fill_value = mode_values[0] if len(mode_values) > 0 else 'unknown'
                        result_df[col] = result_df[col].fillna(fill_value)
                    else:
                        median_value = result_df[col].median()
                        if pd.isna(median_value):
                            median_value = 0  # Якщо медіана також NaN
                        result_df[col] = result_df[col].fillna(median_value)
                except Exception as e:
                    self.logger.error(f"Помилка при заповненні NaN в колонці {col}: {str(e)}")
                    result_df[col] = result_df[col].fillna(0)

        # Видаляємо колонки з нескінченними значеннями
        inf_cols = []
        for col in result_df.columns:
            if result_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                if np.isinf(result_df[col]).any():
                    inf_cols.append(col)

        if inf_cols:
            self.logger.warning(f"Видалено {len(inf_cols)} колонок з нескінченними значеннями: {inf_cols[:5]}...")
            result_df = result_df.drop(columns=inf_cols)

        # Зменшення розмірності, якщо потрібно
        if reduce_dimensions and result_df.shape[1] > 0:
            try:
                if n_features is None:
                    n_features = min(100, result_df.shape[1] // 2)
                n_features = min(n_features, result_df.shape[1])  # Не більше ніж доступно

                self.logger.info(f"Виконується відбір {n_features} найважливіших ознак")
                result_df, _ = self.select_features(result_df, target, n_features=n_features)
            except Exception as e:
                self.logger.error(f"Помилка при відборі ознак: {str(e)}")

        # Фінальна інформація
        features_added = len(result_df.columns) - initial_feature_count
        self.logger.info(
            f"Пайплайн завершено. Додано {features_added} нових ознак. Загальна кількість ознак: {len(result_df.columns)}")

        # Перевіряємо фінальний результат
        if result_df.empty:
            self.logger.error("Результуючий DataFrame порожній!")
            raise ValueError("Не вдалося створити жодної ознаки")

        if target.empty:
            self.logger.error("Цільова змінна порожня!")
            raise ValueError("Не вдалося створити цільову змінну")

        return result_df, target