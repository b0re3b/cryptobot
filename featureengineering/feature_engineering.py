import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
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

        # Скейлери для нейронних мереж
        self.scalers = {}


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

    # НОВІ МЕТОДИ ДЛЯ НЕЙРОННИХ МЕРЕЖ

    def _get_scaler(self, scaler_type: str = 'standard'):
        """Отримання скейлера заданого типу."""
        if scaler_type == 'standard':
            return StandardScaler()
        elif scaler_type == 'minmax':
            return MinMaxScaler()
        elif scaler_type == 'robust':
            return RobustScaler()
        else:
            self.logger.warning(f"Невідомий тип скейлера {scaler_type}, використовується StandardScaler")
            return StandardScaler()

    def _scale_features(self, data: pd.DataFrame, scaler_type: str = 'standard',
                        fit_scaler: bool = True) -> pd.DataFrame:
        """Масштабування ознак."""
        try:
            scaler_key = f"{scaler_type}_scaler"

            if fit_scaler or scaler_key not in self.scalers:
                self.scalers[scaler_key] = self._get_scaler(scaler_type)
                scaled_data = self.scalers[scaler_key].fit_transform(data)
                self.logger.info(f"Навчено новий скейлер {scaler_type}")
            else:
                scaled_data = self.scalers[scaler_key].transform(data)
                self.logger.info(f"Використано існуючий скейлер {scaler_type}")

            return pd.DataFrame(scaled_data, columns=data.columns, index=data.index)

        except Exception as e:
            self.logger.error(f"Помилка при масштабуванні даних: {str(e)}")
            return data.copy()

    def create_sequences_lstm_gru(self, data: pd.DataFrame, target: pd.Series,
                                  sequence_length: int = 60,
                                  step_size: int = 1,
                                  scale_features: bool = True,
                                  scaler_type: str = 'standard') -> Tuple[np.ndarray, np.ndarray]:
        """
        Створення послідовностей для LSTM/GRU моделей.

        Args:
            data: DataFrame з ознаками
            target: Series з цільовими значеннями
            sequence_length: Довжина послідовності
            step_size: Крок між послідовностями
            scale_features: Чи масштабувати ознаки
            scaler_type: Тип скейлера ('standard', 'minmax', 'robust')

        Returns:
            Кортеж (X_sequences, y_sequences) у вигляді numpy arrays
        """
        self.logger.info(
            f"Створення послідовностей для LSTM/GRU: sequence_length={sequence_length}, step_size={step_size}")

        try:
            # Масштабування ознак
            if scale_features:
                scaled_data = self._scale_features(data, scaler_type, fit_scaler=True)
            else:
                scaled_data = data.copy()

            # Перевіряємо на NaN після масштабування
            if scaled_data.isna().any().any():
                self.logger.warning("Знайдено NaN після масштабування, заповнюємо нулями")
                scaled_data = scaled_data.fillna(0)

            # Вирівнюємо розміри даних і цільової змінної
            min_length = min(len(scaled_data), len(target))
            scaled_data = scaled_data.iloc[:min_length]
            target_aligned = target.iloc[:min_length]

            # Створюємо послідовності
            X_sequences = []
            y_sequences = []

            for i in range(0, min_length - sequence_length + 1, step_size):
                # Беремо sequence_length рядків для X
                X_seq = scaled_data.iloc[i:i + sequence_length].values
                # Беремо відповідне цільове значення (останнє в послідовності)
                y_val = target_aligned.iloc[i + sequence_length - 1]

                # Перевіряємо на NaN у цільовій змінній
                if not pd.isna(y_val):
                    X_sequences.append(X_seq)
                    y_sequences.append(y_val)

            X_sequences = np.array(X_sequences)
            y_sequences = np.array(y_sequences)

            self.logger.info(f"Створено послідовності: X_shape={X_sequences.shape}, y_shape={y_sequences.shape}")

            return X_sequences, y_sequences

        except Exception as e:
            self.logger.error(f"Помилка при створенні послідовностей для LSTM/GRU: {str(e)}")
            raise

    def create_sequences_transformer(self, data: pd.DataFrame, target: pd.Series,
                                     sequence_length: int = 60,
                                     step_size: int = 1,
                                     scale_features: bool = True,
                                     scaler_type: str = 'standard',
                                     add_positional_encoding: bool = True) -> Tuple[
        np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Створення послідовностей для Transformer моделей.

        Args:
            data: DataFrame з ознаками
            target: Series з цільовими значеннями
            sequence_length: Довжина послідовності
            step_size: Крок між послідовностями
            scale_features: Чи масштабувати ознаки
            scaler_type: Тип скейлера
            add_positional_encoding: Чи додавати позиційне кодування

        Returns:
            Кортеж (X_sequences, y_sequences, positional_encoding)
        """
        self.logger.info(f"Створення послідовностей для Transformer: sequence_length={sequence_length}")

        try:
            # Створюємо базові послідовності
            X_sequences, y_sequences = self.create_sequences_lstm_gru(
                data, target, sequence_length, step_size, scale_features, scaler_type
            )

            positional_encoding = None
            if add_positional_encoding:
                # Створюємо позиційне кодування
                positional_encoding = self._create_positional_encoding(sequence_length, data.shape[1])
                self.logger.info(f"Створено позиційне кодування: shape={positional_encoding.shape}")

            return X_sequences, y_sequences, positional_encoding

        except Exception as e:
            self.logger.error(f"Помилка при створенні послідовностей для Transformer: {str(e)}")
            raise

    def _create_positional_encoding(self, seq_len: int, d_model: int) -> np.ndarray:
        """Створення позиційного кодування для Transformer."""
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pos_encoding = np.zeros((seq_len, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)

        return pos_encoding

    def create_multi_step_targets(self, target: pd.Series, steps: List[int]) -> pd.DataFrame:
        """
        Створення багатокрокових цільових змінних.

        Args:
            target: Оригінальна цільова змінна
            steps: Список кроків для передбачення

        Returns:
            DataFrame з цільовими змінними для кожного кроку
        """
        self.logger.info(f"Створення багатокрокових цільових змінних для кроків: {steps}")

        try:
            multi_targets = pd.DataFrame(index=target.index)

            for step in steps:
                target_name = f'target_step_{step}'
                multi_targets[target_name] = target.shift(-step)

            # Видаляємо рядки з NaN
            multi_targets = multi_targets.dropna()

            self.logger.info(f"Створено багатокрокові цілі: shape={multi_targets.shape}")
            return multi_targets

        except Exception as e:
            self.logger.error(f"Помилка при створенні багатокрокових цілей: {str(e)}")
            raise

    def split_sequences_for_training(self, X: np.ndarray, y: np.ndarray,
                                     train_ratio: float = 0.7,
                                     val_ratio: float = 0.15,
                                     test_ratio: float = 0.15) -> Dict[str, np.ndarray]:
        """
        Розподіл послідовностей на навчальну, валідаційну та тестову вибірки.

        Args:
            X: Масив ознак
            y: Масив цільових значень
            train_ratio: Частка навчальних даних
            val_ratio: Частка валідаційних даних
            test_ratio: Частка тестових даних

        Returns:
            Словник з розподіленими даними
        """
        self.logger.info(f"Розподіл даних: train={train_ratio}, val={val_ratio}, test={test_ratio}")

        try:
            # Перевіряємо, що сума коефіцієнтів дорівнює 1
            if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
                raise ValueError("Сума train_ratio, val_ratio та test_ratio повинна дорівнювати 1.0")

            total_samples = len(X)

            # Розраховуємо індекси для розподілу
            train_end = int(total_samples * train_ratio)
            val_end = int(total_samples * (train_ratio + val_ratio))

            # Розподіляємо дані
            X_train = X[:train_end]
            X_val = X[train_end:val_end]
            X_test = X[val_end:]

            y_train = y[:train_end]
            y_val = y[train_end:val_end]
            y_test = y[val_end:]

            self.logger.info(f"Розподіл завершено: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

            return {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test
            }

        except Exception as e:
            self.logger.error(f"Помилка при розподілі даних: {str(e)}")
            raise

    def prepare_features_pipeline(self, data: pd.DataFrame,
                                  target_column: str = 'close',
                                  horizon: int = 1,
                                  model_type: str = 'lstm',
                                  sequence_length: int = 60,
                                  step_size: int = 1,
                                  feature_groups: Optional[List[str]] = None,
                                  reduce_dimensions: bool = False,
                                  n_features: Optional[int] = None,
                                  scale_features: bool = True,
                                  scaler_type: str = 'standard',
                                  multi_step_prediction: Optional[List[int]] = None,
                                  split_data: bool = True,
                                  train_ratio: float = 0.7,
                                  val_ratio: float = 0.15,
                                  test_ratio: float = 0.15) -> Dict:
        """
        Уніфікований пайплайн підготовки ознак для нейронних мереж.

        Args:
            data: Вхідний DataFrame
            target_column: Назва цільової колонки
            horizon: Горизонт прогнозування
            model_type: Тип моделі ('lstm', 'gru', 'transformer')
            sequence_length: Довжина послідовності для нейронних мереж
            step_size: Крок між послідовностями
            feature_groups: Список груп ознак для створення
            reduce_dimensions: Чи виконувати зменшення розмірності
            n_features: Кількість ознак для відбору
            scale_features: Чи масштабувати ознаки
            scaler_type: Тип скейлера
            multi_step_prediction: Список кроків для багатокрокового передбачення
            split_data: Чи розподіляти дані на train/val/test
            train_ratio: Частка навчальних даних
            val_ratio: Частка валідаційних даних
            test_ratio: Частка тестових даних

        Returns:
            Словник з підготовленими даними та метаданими
        """
        self.logger.info("Початок виконання уніфікованого пайплайну підготовки ознак")

        try:
            # Ініціалізуємо результат
            result = {
                'metadata': {
                    'model_type': model_type,
                    'sequence_length': sequence_length,
                    'horizon': horizon,
                    'feature_groups': feature_groups,
                    'original_shape': data.shape,
                    'scaler_type': scaler_type if scale_features else None,
                    'multi_step': multi_step_prediction is not None
                }
            }

            # Крок 1: Копіюємо дані та базова перевірка
            processed_data = data.copy()
            self.logger.info(f"Початкова форма даних: {processed_data.shape}")

            # Перевіряємо наявність цільової колонки
            if target_column not in processed_data.columns:
                raise ValueError(f"Цільова колонка '{target_column}' не знайдена в даних")

            # Крок 2: Створення базових груп ознак
            if feature_groups is None:
                feature_groups = ['technical', 'statistical', 'time', 'lagged']

            for group in feature_groups:
                self.logger.info(f"Створення групи ознак: {group}")

                if group == 'technical':
                    processed_data = self.create_technical_features(processed_data)
                    processed_data = self.create_candle_pattern_features(processed_data)
                    processed_data = self.create_custom_indicators(processed_data)

                elif group == 'statistical':
                    processed_data = self.create_volatility_features(processed_data, target_column)
                    processed_data = self.create_return_features(processed_data, target_column)
                    if 'volume' in processed_data.columns:
                        processed_data = self.create_volume_features(processed_data)

                elif group == 'time':
                    processed_data = self.create_datetime_features(processed_data, cyclical=True)
                    processed_data = self.create_rolling_features(processed_data)
                    processed_data = self.create_ewm_features(processed_data)

                elif group == 'lagged':
                    processed_data = self.create_lagged_features(processed_data)

                elif group == 'cross':
                    # Створюємо кросс-ознаки між технічними індикаторами
                    tech_cols = [col for col in processed_data.columns if any(
                        indicator in col.lower() for indicator in ['sma', 'ema', 'rsi', 'macd']
                    )]
                    if len(tech_cols) >= 2:
                        processed_data = self.create_ratio_features(
                            processed_data, tech_cols[:5], tech_cols[5:10] if len(tech_cols) > 5 else tech_cols[:5]
                        )

                elif group == 'polynomial':
                    # Створюємо поліноміальні ознаки для ключових індикаторів
                    key_cols = [col for col in processed_data.columns if any(
                        key in col.lower() for key in ['rsi', 'close', 'volume', 'return']
                    )][:5]  # Обмежуємо кількість, щоб уникнути вибуху ознак
                    if key_cols:
                        processed_data = self.create_polynomial_features(
                            processed_data, key_cols, degree=2, interaction_only=True
                        )

            self.logger.info(f"Форма даних після створення ознак: {processed_data.shape}")

            # Крок 3: Очищення даних
            # Видаляємо колонки з занадто багатьма NaN
            nan_threshold = 0.5  # Якщо більше 50% NaN, видаляємо колонку
            cols_to_drop = []
            for col in processed_data.columns:
                nan_ratio = processed_data[col].isna().sum() / len(processed_data)
                if nan_ratio > nan_threshold:
                    cols_to_drop.append(col)

            if cols_to_drop:
                self.logger.warning(f"Видаляємо колонки з великою кількістю NaN: {len(cols_to_drop)} колонок")
                processed_data = processed_data.drop(columns=cols_to_drop)

            # Заповнюємо решту NaN
            for col in processed_data.columns:
                if processed_data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # Для числових колонок використовуємо forward fill, потім backward fill
                    processed_data[col] = processed_data[col].fillna(method='ffill').fillna(method='bfill')
                    # Якщо все ще є NaN, заповнюємо медіаною
                    if processed_data[col].isna().any():
                        processed_data[col] = processed_data[col].fillna(processed_data[col].median())

            # Видаляємо рядки де все ще є NaN
            processed_data = processed_data.dropna()
            self.logger.info(f"Форма даних після очищення: {processed_data.shape}")

            # Крок 4: Створення цільової змінної
            processed_data = self.create_target_variable(
                processed_data,
                price_column=target_column,
                horizon=horizon,
                target_type='log_return'
            )

            target_col_name = f'target_log_return_{horizon}p'

            # Крок 5: Багатокрокове передбачення (якщо потрібно)
            if multi_step_prediction:
                multi_targets = self.create_multi_step_targets(
                    processed_data[target_col_name], multi_step_prediction
                )
                # Вирівнюємо індекси
                common_index = processed_data.index.intersection(multi_targets.index)
                processed_data = processed_data.loc[common_index]
                target_data = multi_targets.loc[common_index]
                result['metadata']['target_columns'] = list(multi_targets.columns)
            else:
                target_data = processed_data[target_col_name]
                result['metadata']['target_columns'] = [target_col_name]

            # Крок 6: Відокремлюємо ознаки від цільових змінних
            feature_columns = [col for col in processed_data.columns
                               if not col.startswith('target_')]
            X_data = processed_data[feature_columns]

            self.logger.info(f"Кількість ознак: {len(feature_columns)}")

            # Крок 7: Відбір ознак (якщо потрібно)
            if reduce_dimensions and n_features:
                self.logger.info(f"Відбір {n_features} найважливіших ознак")
                if isinstance(target_data, pd.DataFrame):
                    # Для багатокрокового передбачення використовуємо першу цільову змінну
                    y_for_selection = target_data.iloc[:, 0]
                else:
                    y_for_selection = target_data

                X_selected, selected_features = self.select_features(
                    X_data, y_for_selection, n_features=n_features
                )
                X_data = X_selected
                result['metadata']['selected_features'] = selected_features
                self.logger.info(f"Відібрано {len(selected_features)} ознак")

            # Крок 8: Створення послідовностей для нейронних мереж
            if model_type.lower() in ['lstm', 'gru']:
                if isinstance(target_data, pd.DataFrame):
                    # Для багатокрокового передбачення створюємо послідовності для кожної цілі
                    sequences_data = {}
                    for target_col in target_data.columns:
                        X_seq, y_seq = self.create_sequences_lstm_gru(
                            X_data, target_data[target_col],
                            sequence_length=sequence_length,
                            step_size=step_size,
                            scale_features=scale_features,
                            scaler_type=scaler_type
                        )
                        sequences_data[f'X_{target_col}'] = X_seq
                        sequences_data[f'y_{target_col}'] = y_seq
                    result['sequences'] = sequences_data
                else:
                    X_sequences, y_sequences = self.create_sequences_lstm_gru(
                        X_data, target_data,
                        sequence_length=sequence_length,
                        step_size=step_size,
                        scale_features=scale_features,
                        scaler_type=scaler_type
                    )
                    result['X_sequences'] = X_sequences
                    result['y_sequences'] = y_sequences

            elif model_type.lower() == 'transformer':
                if isinstance(target_data, pd.DataFrame):
                    sequences_data = {}
                    for target_col in target_data.columns:
                        X_seq, y_seq, pos_enc = self.create_sequences_transformer(
                            X_data, target_data[target_col],
                            sequence_length=sequence_length,
                            step_size=step_size,
                            scale_features=scale_features,
                            scaler_type=scaler_type,
                            add_positional_encoding=True
                        )
                        sequences_data[f'X_{target_col}'] = X_seq
                        sequences_data[f'y_{target_col}'] = y_seq
                        if pos_enc is not None:
                            sequences_data['positional_encoding'] = pos_enc
                    result['sequences'] = sequences_data
                else:
                    X_sequences, y_sequences, positional_encoding = self.create_sequences_transformer(
                        X_data, target_data,
                        sequence_length=sequence_length,
                        step_size=step_size,
                        scale_features=scale_features,
                        scaler_type=scaler_type,
                        add_positional_encoding=True
                    )
                    result['X_sequences'] = X_sequences
                    result['y_sequences'] = y_sequences
                    if positional_encoding is not None:
                        result['positional_encoding'] = positional_encoding

            # Крок 9: Розподіл даних (якщо потрібно)
            if split_data:
                if isinstance(target_data, pd.DataFrame):
                    # Для багатокрокового передбачення розподіляємо кожну послідовність
                    if 'sequences' in result:
                        split_data_dict = {}
                        for target_col in target_data.columns:
                            X_key = f'X_{target_col}'
                            y_key = f'y_{target_col}'
                            if X_key in result['sequences'] and y_key in result['sequences']:
                                split_result = self.split_sequences_for_training(
                                    result['sequences'][X_key],
                                    result['sequences'][y_key],
                                    train_ratio=train_ratio,
                                    val_ratio=val_ratio,
                                    test_ratio=test_ratio
                                )
                                for key, value in split_result.items():
                                    split_data_dict[f"{key}_{target_col}"] = value
                        result['split_data'] = split_data_dict
                else:
                    # Для одиночного передбачення
                    if 'X_sequences' in result and 'y_sequences' in result:
                        result['split_data'] = self.split_sequences_for_training(
                            result['X_sequences'],
                            result['y_sequences'],
                            train_ratio=train_ratio,
                            val_ratio=val_ratio,
                            test_ratio=test_ratio
                        )

            # Крок 10: Збереження метаданих
            result['metadata'].update({
                'final_shape': processed_data.shape,
                'feature_columns': feature_columns,
                'n_features_final': len(feature_columns),
                'sequence_shape': result.get('X_sequences', np.array([])).shape if 'X_sequences' in result else None,
                'data_split': split_data,
                'scaler_keys': list(self.scalers.keys()) if hasattr(self, 'scalers') else []
            })

            # Додаємо також необроблені дані для аналізу
            result['processed_data'] = processed_data
            result['feature_data'] = X_data
            result['target_data'] = target_data

            self.logger.info("Пайплайн підготовки ознак завершено успішно")
            self.logger.info(f"Підсумок: {result['metadata']['n_features_final']} ознак, "
                             f"модель: {model_type}, горизонт: {horizon}")

            if 'X_sequences' in result:
                self.logger.info(f"Форма послідовностей: X={result['X_sequences'].shape}, "
                                 f"y={result['y_sequences'].shape}")

            return result

        except Exception as e:
            self.logger.error(f"Помилка в пайплайні підготовки ознак: {str(e)}")
            raise