from datetime import datetime
from typing import Union, Tuple, List, Dict
from data.db import DatabaseManager
import numpy as np
import pandas as pd
from scipy import stats

from utils.logger import CryptoLogger


class TimeSeriesTransformer:
    def __init__(self):
        self.logger = CryptoLogger('INFO')
        self.transformations = {}
        self.db_manager = DatabaseManager()
    def difference_series(self, data: pd.Series, order: int = 1) -> pd.Series:

        if order < 1:
            self.logger.warning("Differencing order must be at least 1, using order=1 instead")
            order = 1

        if data.isnull().any():
            self.logger.warning("Data contains NaN values. Removing them before differencing.")
            data = data.dropna()

        if len(data) <= order:
            self.logger.error(f"Not enough data points for {order}-order differencing")
            return pd.Series([], index=pd.DatetimeIndex([]))

        # Застосовуємо послідовне диференціювання
        diff_data = data.copy()
        for i in range(order):
            diff_data = diff_data.diff().dropna()

            # Перевіряємо чи залишились дані після диференціювання
            if len(diff_data) == 0:
                self.logger.error(f"No data left after {i + 1}-order differencing")
                return pd.Series([], index=pd.DatetimeIndex([]))

        self.logger.info(
            f"Applied {order}-order differencing. Original length: {len(data)}, Result length: {len(diff_data)}")

        return diff_data

    def transform_data(self, data: pd.Series, method: str = 'log') -> Union[pd.Series, Tuple[pd.Series, float]]:

        self.logger.info(f"Applying {method} transformation to data")

        # Перевірка наявності нульових та від'ємних значень для певних трансформацій
        if method in ['log', 'boxcox']:
            min_value = data.min()
            if min_value <= 0:
                self.logger.warning(f"Data contains non-positive values, adding offset for {method} transformation")
                offset = abs(min_value) + 1
                data = data + offset

                # Зберігаємо інформацію про зсув для подальшої зворотної трансформації
                self.transformations[method] = {'offset': offset}
            else:
                self.transformations[method] = {'offset': 0}

        if method == 'none':
            # Без трансформації
            return data

        elif method == 'log':
            # Логарифмічна трансформація
            transformed_data = np.log(data)
            return transformed_data

        elif method == 'sqrt':
            # Квадратний корінь
            transformed_data = np.sqrt(data)
            return transformed_data

        elif method == 'boxcox':
            # Трансформація Бокса-Кокса
            transformed_data, lambda_param = stats.boxcox(data)

            # Зберігаємо lambda параметр для зворотної трансформації
            if method in self.transformations:
                self.transformations[method]['lambda'] = lambda_param
            else:
                self.transformations[method] = {'lambda': lambda_param, 'offset': 0}

            self.logger.info(f"BoxCox transformation applied with lambda = {lambda_param}")
            return transformed_data, lambda_param

        elif method == 'yeo-johnson':
            # Трансформація Йео-Джонсона (працює з від'ємними значеннями)
            from sklearn.preprocessing import PowerTransformer

            # Підготовка даних для трансформації
            data_reshaped = data.values.reshape(-1, 1)

            # Застосування трансформації
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            transformed_data_array = pt.fit_transform(data_reshaped)
            transformed_data = pd.Series(transformed_data_array.flatten(), index=data.index)

            # Зберігаємо параметри трансформації
            lambda_param = pt.lambdas_[0]
            self.transformations['yeo-johnson'] = {'lambda': lambda_param, 'transformer': pt}

            self.logger.info(f"Yeo-Johnson transformation applied with lambda = {lambda_param}")
            return transformed_data, lambda_param

        else:
            error_msg = f"Unknown transformation method: {method}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def inverse_transform(self, data: pd.Series, method: str = 'log', lambda_param: float = None) -> pd.Series:

        self.logger.info(f"Applying inverse {method} transformation")

        if method == 'none':
            # Без трансформації
            return data

        # Отримуємо збережені параметри трансформації
        transform_params = self.transformations.get(method, {})
        offset = transform_params.get('offset', 0)

        if method == 'log':
            # Зворотна логарифмічна трансформація
            inverse_data = np.exp(data)

            # Відновлюємо зсув, якщо він був застосований
            if offset > 0:
                inverse_data = inverse_data - offset

            return inverse_data

        elif method == 'sqrt':
            # Зворотна трансформація квадратного кореня
            inverse_data = data ** 2

            # Відновлюємо зсув, якщо він був застосований
            if offset > 0:
                inverse_data = inverse_data - offset

            return inverse_data

        elif method == 'boxcox':
            # Зворотна трансформація Бокса-Кокса

            # Якщо lambda_param не вказано, намагаємося використати збережений
            if lambda_param is None:
                lambda_param = transform_params.get('lambda')

                if lambda_param is None:
                    error_msg = "lambda parameter for BoxCox inverse transformation is not provided or saved"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)

            # Застосовуємо зворотну трансформацію
            inverse_data = stats.inv_boxcox(data, lambda_param)

            # Відновлюємо зсув, якщо він був застосований
            if offset > 0:
                inverse_data = inverse_data - offset

            return inverse_data

        elif method == 'yeo-johnson':
            # Зворотна трансформація Йео-Джонсона

            # Отримуємо збережений трансформер
            transformer = transform_params.get('transformer')

            if transformer is None:
                error_msg = "Transformer object for Yeo-Johnson inverse transformation is not saved"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Підготовка даних
            data_reshaped = data.values.reshape(-1, 1)

            # Застосування зворотної трансформації
            inverse_data_array = transformer.inverse_transform(data_reshaped)
            inverse_data = pd.Series(inverse_data_array.flatten(), index=data.index)

            return inverse_data

        else:
            error_msg = f"Unknown transformation method: {method}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def apply_preprocessing_pipeline(self, data: pd.Series, operations: List[Dict]) -> pd.Series:

        self.logger.info(f"Applying preprocessing pipeline with {len(operations)} operations")

        if data.isnull().any():
            self.logger.warning("Data contains NaN values. Removing them before preprocessing.")
            data = data.dropna()

        if len(data) == 0:
            self.logger.error("Data is empty after removing NaN values")
            return pd.Series([], index=pd.DatetimeIndex([]))

        original_data = data.copy()
        processed_data = data.copy()
        transformations_info = []

        try:
            for i, operation in enumerate(operations):
                op_type = operation.get('op', '').lower()

                if not op_type:
                    self.logger.warning(f"Operation {i + 1} has no 'op' field, skipping")
                    continue

                self.logger.info(f"Applying operation {i + 1}: {op_type}")

                if op_type == 'log':
                    # Перевірка на наявність нульових або від'ємних значень
                    if (processed_data <= 0).any():
                        min_positive = processed_data[processed_data > 0].min() if (processed_data > 0).any() else 1e-6
                        offset = abs(processed_data.min()) + min_positive if processed_data.min() <= 0 else 0
                        self.logger.warning(f"Negative or zero values found in data. Adding offset {offset}")
                        processed_data = processed_data + offset
                        transformations_info.append({
                            "method": "log",
                            "params": {"offset": offset}
                        })
                    else:
                        transformations_info.append({
                            "method": "log",
                            "params": {}
                        })
                    processed_data = np.log(processed_data)

                elif op_type == 'sqrt':
                    # Перевірка на наявність від'ємних значень
                    if (processed_data < 0).any():
                        offset = abs(processed_data.min()) + 1e-6
                        self.logger.warning(f"Negative values found in data. Adding offset {offset}")
                        processed_data = processed_data + offset
                        transformations_info.append({
                            "method": "sqrt",
                            "params": {"offset": offset}
                        })
                    else:
                        transformations_info.append({
                            "method": "sqrt",
                            "params": {}
                        })
                    processed_data = np.sqrt(processed_data)

                elif op_type == 'boxcox':
                    from scipy import stats
                    # BoxCox працює тільки з додатними значеннями
                    if (processed_data <= 0).any():
                        min_positive = processed_data[processed_data > 0].min() if (processed_data > 0).any() else 1e-6
                        offset = abs(processed_data.min()) + min_positive
                        self.logger.warning(f"Non-positive values found in data. Adding offset {offset}")
                        processed_data = processed_data + offset
                        processed_data, lambda_param = stats.boxcox(processed_data)
                        transformations_info.append({
                            "method": "boxcox",
                            "params": {
                                "lambda": lambda_param,
                                "offset": offset
                            }
                        })
                    else:
                        processed_data, lambda_param = stats.boxcox(processed_data)
                        transformations_info.append({
                            "method": "boxcox",
                            "params": {
                                "lambda": lambda_param
                            }
                        })

                elif op_type == 'diff':
                    order = operation.get('order', 1)
                    if not isinstance(order, int) or order < 1:
                        self.logger.warning(f"Invalid differencing order {order}, using 1 instead")
                        order = 1

                    processed_data = processed_data.diff(order).dropna()
                    transformations_info.append({
                        "method": "diff",
                        "params": {"order": order}
                    })

                elif op_type == 'seasonal_diff':
                    lag = operation.get('lag', 7)  # За замовчуванням тижнева сезонність
                    if not isinstance(lag, int) or lag < 1:
                        self.logger.warning(f"Invalid seasonal lag {lag}, using 7 instead")
                        lag = 7

                    processed_data = processed_data.diff(lag).dropna()
                    transformations_info.append({
                        "method": "seasonal_diff",
                        "params": {"lag": lag}
                    })

                elif op_type == 'remove_outliers':
                    method = operation.get('method', 'iqr')
                    threshold = operation.get('threshold', 1.5)

                    if method == 'iqr':
                        # Метод міжквартильного розмаху
                        q1 = processed_data.quantile(0.25)
                        q3 = processed_data.quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - threshold * iqr
                        upper_bound = q3 + threshold * iqr

                        # Зберігаємо інформацію про межі для можливого відновлення
                        outlier_mask = (processed_data < lower_bound) | (processed_data > upper_bound)
                        outlier_indices = outlier_mask[outlier_mask].index.tolist()
                        outlier_values = processed_data[outlier_mask].tolist()

                        # Заміна викидів на межі
                        processed_data = processed_data.clip(lower=lower_bound, upper=upper_bound)

                        transformations_info.append({
                            "method": "remove_outliers",
                            "params": {
                                "method": method,
                                "threshold": threshold,
                                "replaced_outliers": {
                                    "indices": outlier_indices,
                                    "values": outlier_values
                                }
                            }
                        })

                    elif method == 'zscore':
                        # Метод z-оцінки
                        z_scores = (processed_data - processed_data.mean()) / processed_data.std()
                        outlier_mask = abs(z_scores) > threshold

                        # Зберігаємо інформацію про викиди
                        outlier_indices = outlier_mask[outlier_mask].index.tolist()
                        outlier_values = processed_data[outlier_mask].tolist()

                        # Заміна викидів на None і інтерполяція
                        processed_data[outlier_mask] = None
                        processed_data = processed_data.interpolate(method='linear')

                        transformations_info.append({
                            "method": "remove_outliers",
                            "params": {
                                "method": method,
                                "threshold": threshold,
                                "replaced_outliers": {
                                    "indices": outlier_indices,
                                    "values": outlier_values
                                }
                            }
                        })

                    else:
                        self.logger.warning(f"Unknown outlier removal method: {method}, skipping")

                elif op_type == 'moving_average':
                    window = operation.get('window', 3)
                    center = operation.get('center', False)

                    if not isinstance(window, int) or window < 2:
                        self.logger.warning(f"Invalid window size {window} for moving average, using 3 instead")
                        window = 3

                    processed_data = processed_data.rolling(window=window, center=center).mean().dropna()
                    transformations_info.append({
                        "method": "moving_average",
                        "params": {
                            "window": window,
                            "center": center
                        }
                    })

                elif op_type == 'ewm':
                    # Експоненційно зважене середнє
                    span = operation.get('span', 5)

                    if not isinstance(span, int) or span < 2:
                        self.logger.warning(f"Invalid span {span} for EWM, using 5 instead")
                        span = 5

                    processed_data = processed_data.ewm(span=span).mean()
                    transformations_info.append({
                        "method": "ewm",
                        "params": {"span": span}
                    })

                elif op_type == 'normalize':
                    method = operation.get('method', 'minmax')

                    if method == 'minmax':
                        # Min-Max масштабування
                        min_val = processed_data.min()
                        max_val = processed_data.max()
                        if max_val > min_val:
                            processed_data = (processed_data - min_val) / (max_val - min_val)
                        else:
                            processed_data = processed_data * 0  # Якщо всі значення однакові

                        transformations_info.append({
                            "method": "normalize",
                            "params": {
                                "method": method,
                                "min": min_val,
                                "max": max_val
                            }
                        })

                    elif method == 'zscore':
                        # Z-score стандартизація
                        mean_val = processed_data.mean()
                        std_val = processed_data.std()
                        if std_val > 0:
                            processed_data = (processed_data - mean_val) / std_val
                        else:
                            processed_data = processed_data * 0  # Якщо стандартне відхилення нульове

                        transformations_info.append({
                            "method": "normalize",
                            "params": {
                                "method": method,
                                "mean": mean_val,
                                "std": std_val
                            }
                        })

                    else:
                        self.logger.warning(f"Unknown normalization method: {method}, skipping")

                else:
                    self.logger.warning(f"Unknown operation type: {op_type}, skipping")

                if len(processed_data) == 0:
                    self.logger.error(f"No data left after operation {i + 1}: {op_type}")
                    return pd.Series([], index=pd.DatetimeIndex([]))

            # Зберігаємо інформацію про трансформації, якщо є db_manager
            if self.db_manager is not None and transformations_info:
                try:
                    transformation_key = f"transform_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    self.db_manager.save_data_transformations(
                        key=transformation_key,
                        transformations=transformations_info,
                        metadata={
                            "original_length": len(original_data),
                            "processed_length": len(processed_data),
                            "timestamp": datetime.now().isoformat(),
                            "operations": operations
                        }
                    )
                    # Зберігаємо трансформації в локальному словнику також
                    self.transformations[transformation_key] = transformations_info

                    self.logger.info(f"Saved transformation pipeline with key: {transformation_key}")
                except Exception as db_error:
                    self.logger.error(f"Error saving transformation pipeline to database: {str(db_error)}")

            self.logger.info(
                f"Preprocessing pipeline applied successfully. Original length: {len(original_data)}, Processed length: {len(processed_data)}")
            return processed_data

        except Exception as e:
            self.logger.error(f"Error during preprocessing pipeline: {str(e)}")
            return pd.Series([], index=pd.DatetimeIndex([]))

    def extract_volatility(self, data: pd.Series, window: int = 20) -> pd.Series:

        self.logger.info(f"Calculating volatility with window size {window}")

        if data.isnull().any():
            self.logger.warning("Data contains NaN values. Removing them before volatility calculation.")
            data = data.dropna()

        if len(data) < window:
            self.logger.error(f"Not enough data points for volatility calculation with window={window}")
            return pd.Series([], index=pd.DatetimeIndex([]))

        try:
            # Розрахунок логарифмічних прибутків
            log_returns = np.log(data / data.shift(1)).dropna()

            # Розрахунок волатильності як ковзного стандартного відхилення логарифмічних прибутків
            volatility = log_returns.rolling(window=window).std()

            # Переведення стандартного відхилення у волатильність (анулізована)
            # Для різних частот даних множник буде різним:
            # - Денні дані: множник = sqrt(252) - кількість торгових днів у році
            # - Годинні дані: множник = sqrt(252 * 24)
            # - Хвилинні дані: множник = sqrt(252 * 24 * 60)

            # За замовчуванням припускаємо денні дані
            if isinstance(data.index, pd.DatetimeIndex):
                # Визначаємо частоту даних
                if len(data) >= 2:
                    time_diff = data.index[1:] - data.index[:-1]
                    median_diff = pd.Series(time_diff).median()

                    if median_diff <= pd.Timedelta(minutes=5):
                        annualization_factor = np.sqrt(252 * 24 * 12)  # 5-хвилинні дані
                    elif median_diff <= pd.Timedelta(hours=1):
                        annualization_factor = np.sqrt(252 * 24)  # Годинні дані
                    elif median_diff <= pd.Timedelta(days=1):
                        annualization_factor = np.sqrt(252)  # Денні дані
                    else:
                        annualization_factor = 1  # Не анулізуємо для нестандартних інтервалів
                else:
                    annualization_factor = np.sqrt(252)  # За замовчуванням
            else:
                annualization_factor = np.sqrt(252)  # За замовчуванням

            volatility = volatility * annualization_factor

            self.logger.info(f"Volatility calculation completed. Annualization factor: {annualization_factor}")

            return volatility

        except Exception as e:
            self.logger.error(f"Error during volatility calculation: {str(e)}")
            return pd.Series([], index=data.index)