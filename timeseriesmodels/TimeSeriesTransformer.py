from typing import Union, Tuple, List, Dict
from data.db import DatabaseManager
import numpy as np
import pandas as pd
from scipy import stats

from utils.logger import CryptoLogger


class TimeSeriesTransformer:
    def __init__(self):
        self.logger = CryptoLogger('TimeseriesModelEvaluator')
        self.transformations = {}
        self.db_manager = DatabaseManager()

    def difference_series(self, data: pd.Series, order: int = 1) -> pd.Series:
        if order < 1:
            self.logger.warning("Differencing order must be at least 1, using order=1 instead")
            order = 1

        # Перевірка на нульові значення (не потрібно спеціальної обробки для операції різниці)
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

        # Перевірка на null значення
        if data.isnull().any():
            self.logger.warning(f"Data contains NaN values. Removing them before {method} transformation.")
            data = data.dropna()

        # Перевірка наявності нульових та від'ємних значень для певних трансформацій
        if method in ['log', 'boxcox']:
            min_value = data.min()

            # Більш безпечна перевірка на нульові та від'ємні значення
            if min_value <= 0:
                # Знаходимо мінімальне позитивне значення для більш обґрунтованого зсуву
                positive_values = data[data > 0]
                min_positive = positive_values.min() if len(positive_values) > 0 else 1e-6

                # Встановлюємо зсув щоб всі значення були додатними
                offset = abs(min_value) + min_positive
                self.logger.warning(
                    f"Data contains non-positive values ({min_value}), adding offset {offset} for {method} transformation")
                data = data + offset

                # Зберігаємо інформацію про зсув для подальшої зворотної трансформації
                self.transformations[method] = {'offset': offset}
            else:
                self.transformations[method] = {'offset': 0}

        if method == 'none':
            # Без трансформації
            return data

        elif method == 'log':
            # Логарифмічна трансформація з перевіркою на нульові значення
            # Null значення вже відфільтровані, а нульові значення зсунуті вище
            transformed_data = np.log(data)
            return transformed_data

        elif method == 'sqrt':
            # Квадратний корінь
            # Перевіряємо наявність від'ємних значень
            if (data < 0).any():
                min_value = data.min()
                offset = abs(min_value) + 1e-6
                self.logger.warning(
                    f"Data contains negative values ({min_value}), adding offset {offset} for sqrt transformation")
                data = data + offset
                self.transformations['sqrt'] = {'offset': offset}
            else:
                self.transformations['sqrt'] = {'offset': 0}

            transformed_data = np.sqrt(data)
            return transformed_data

        elif method == 'boxcox':
            # Трансформація Бокса-Кокса
            # Нульові та від'ємні значення вже оброблені вище
            try:
                transformed_data, lambda_param = stats.boxcox(data)
            except Exception as e:
                self.logger.error(f"BoxCox transformation error: {str(e)}. Using log transformation instead.")
                transformed_data = np.log(data)
                lambda_param = 0  # log transform is boxcox with lambda=0

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

            # Перевірка на нульові значення не потрібна для Yeo-Johnson
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
            try:
                inverse_data = stats.inv_boxcox(data, lambda_param)
            except Exception as e:
                self.logger.error(f"Error in inverse BoxCox transformation: {str(e)}")
                # Якщо lambda близька до 0, використовуємо експоненційну функцію (зворотну до log)
                if abs(lambda_param) < 1e-5:
                    self.logger.info("Lambda is close to 0, using exp as inverse transformation")
                    inverse_data = np.exp(data)
                else:
                    raise

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

    def apply_preprocessing_pipeline(self, data: pd.Series, operations: List[Dict], model_id: int = None) -> pd.Series:
        self.logger.info(f"Applying preprocessing pipeline with {len(operations)} operations")

        # Спочатку перевіряємо наявність NaN значень
        if data.isnull().any():
            self.logger.warning(f"Data contains {data.isnull().sum()} NaN values. Removing them before preprocessing.")
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
                        min_value = processed_data.min()
                        # Знаходимо найменше позитивне значення для кращого зсуву
                        positive_values = processed_data[processed_data > 0]
                        min_positive = positive_values.min() if len(positive_values) > 0 else 1e-6

                        # Встановлюємо зсув щоб усі значення стали позитивними
                        offset = abs(min_value) + min_positive if min_value <= 0 else 0
                        self.logger.warning(
                            f"Negative or zero values found in data. Min value: {min_value}. Adding offset {offset}")

                        processed_data = processed_data + offset
                        transformations_info.append({
                            "type": "log",
                            "params": {"offset": offset},
                            "order": i + 1
                        })
                    else:
                        transformations_info.append({
                            "type": "log",
                            "params": {},
                            "order": i + 1
                        })
                    processed_data = np.log(processed_data)

                elif op_type == 'sqrt':
                    # Перевірка на наявність від'ємних значень
                    if (processed_data < 0).any():
                        min_value = processed_data.min()
                        offset = abs(min_value) + 1e-6
                        self.logger.warning(
                            f"Negative values found in data. Min value: {min_value}. Adding offset {offset}")
                        processed_data = processed_data + offset
                        transformations_info.append({
                            "type": "sqrt",
                            "params": {"offset": offset},
                            "order": i + 1
                        })
                    else:
                        transformations_info.append({
                            "type": "sqrt",
                            "params": {},
                            "order": i + 1
                        })
                    processed_data = np.sqrt(processed_data)

                elif op_type == 'boxcox':
                    from scipy import stats
                    # BoxCox працює тільки з додатними значеннями
                    if (processed_data <= 0).any():
                        min_value = processed_data.min()
                        # Знаходимо найменше позитивне значення для кращого зсуву
                        positive_values = processed_data[processed_data > 0]
                        min_positive = positive_values.min() if len(positive_values) > 0 else 1e-6

                        offset = abs(min_value) + min_positive
                        self.logger.warning(
                            f"Non-positive values found in data. Min value: {min_value}. Adding offset {offset}")

                        processed_data = processed_data + offset

                        try:
                            processed_data, lambda_param = stats.boxcox(processed_data)
                            transformations_info.append({
                                "type": "boxcox",
                                "params": {
                                    "lambda": lambda_param,
                                    "offset": offset
                                },
                                "order": i + 1
                            })
                        except Exception as e:
                            self.logger.error(f"BoxCox error: {str(e)}. Using log transformation instead.")
                            processed_data = np.log(processed_data)
                            transformations_info.append({
                                "type": "log",
                                "params": {
                                    "offset": offset
                                },
                                "order": i + 1
                            })
                    else:
                        try:
                            processed_data, lambda_param = stats.boxcox(processed_data)
                            transformations_info.append({
                                "type": "boxcox",
                                "params": {
                                    "lambda": lambda_param
                                },
                                "order": i + 1
                            })
                        except Exception as e:
                            self.logger.error(f"BoxCox error: {str(e)}. Using log transformation instead.")
                            processed_data = np.log(processed_data)
                            transformations_info.append({
                                "type": "log",
                                "params": {},
                                "order": i + 1
                            })

                elif op_type == 'diff':
                    order = operation.get('order', 1)
                    if not isinstance(order, int) or order < 1:
                        self.logger.warning(f"Invalid differencing order {order}, using 1 instead")
                        order = 1

                    processed_data = processed_data.diff(order).dropna()
                    transformations_info.append({
                        "type": "diff",
                        "params": {"order": order},
                        "order": i + 1
                    })

                elif op_type == 'seasonal_diff':
                    lag = operation.get('lag', 7)  # За замовчуванням тижнева сезонність
                    if not isinstance(lag, int) or lag < 1:
                        self.logger.warning(f"Invalid seasonal lag {lag}, using 7 instead")
                        lag = 7

                    processed_data = processed_data.diff(lag).dropna()
                    transformations_info.append({
                        "type": "seasonal_diff",
                        "params": {"lag": lag},
                        "order": i + 1
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
                            "type": "remove_outliers",
                            "params": {
                                "method": method,
                                "threshold": threshold,
                                "replaced_outliers": {
                                    "indices": outlier_indices,
                                    "values": outlier_values
                                }
                            },
                            "order": i + 1
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
                            "type": "remove_outliers",
                            "params": {
                                "method": method,
                                "threshold": threshold,
                                "replaced_outliers": {
                                    "indices": outlier_indices,
                                    "values": outlier_values
                                }
                            },
                            "order": i + 1
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
                        "type": "moving_average",
                        "params": {
                            "window": window,
                            "center": center
                        },
                        "order": i + 1
                    })

                elif op_type == 'ewm':
                    # Експоненційно зважене середнє
                    span = operation.get('span', 5)

                    if not isinstance(span, int) or span < 2:
                        self.logger.warning(f"Invalid span {span} for EWM, using 5 instead")
                        span = 5

                    processed_data = processed_data.ewm(span=span).mean()
                    transformations_info.append({
                        "type": "ewm",
                        "params": {"span": span},
                        "order": i + 1
                    })

                elif op_type == 'normalize':
                    method = operation.get('method', 'minmax')

                    if method == 'minmax':
                        # Min-Max масштабування
                        min_val = processed_data.min()
                        max_val = processed_data.max()

                        # Захист від однакових значень (max=min)
                        if max_val > min_val:
                            processed_data = (processed_data - min_val) / (max_val - min_val)
                        else:
                            self.logger.warning(
                                "All values are the same (max=min). Setting all values to 0.5 for minmax scaling.")
                            processed_data = pd.Series(0.5, index=processed_data.index)

                        transformations_info.append({
                            "type": "normalize",
                            "params": {
                                "method": method,
                                "min": min_val,
                                "max": max_val
                            },
                            "order": i + 1
                        })

                    elif method == 'zscore':
                        # Z-score стандартизація
                        mean_val = processed_data.mean()
                        std_val = processed_data.std()

                        # Захист від нульового стандартного відхилення
                        if std_val > 0:
                            processed_data = (processed_data - mean_val) / std_val
                        else:
                            self.logger.warning(
                                "Standard deviation is zero. Setting all values to 0 for z-score normalization.")
                            processed_data = pd.Series(0, index=processed_data.index)

                        transformations_info.append({
                            "type": "normalize",
                            "params": {
                                "method": method,
                                "mean": mean_val,
                                "std": std_val
                            },
                            "order": i + 1
                        })

                    else:
                        self.logger.warning(f"Unknown normalization method: {method}, skipping")

                else:
                    self.logger.warning(f"Unknown operation type: {op_type}, skipping")

                if len(processed_data) == 0:
                    self.logger.error(f"No data left after operation {i + 1}: {op_type}")
                    return pd.Series([], index=pd.DatetimeIndex([]))

            # Зберігаємо інформацію про трансформації, якщо є model_id
            if self.db_manager is not None and transformations_info and model_id is not None:
                try:
                    # Викликаємо метод збереження трансформацій з правильними параметрами
                    save_result = self.db_manager.save_data_transformations(
                        model_id=model_id,
                        transformations=transformations_info
                    )

                    if save_result:
                        self.logger.info(f"Successfully saved transformation pipeline for model_id: {model_id}")
                    else:
                        self.logger.error(f"Failed to save transformation pipeline for model_id: {model_id}")

                    # Зберігаємо трансформації в локальному словнику також
                    self.transformations[f"model_{model_id}"] = transformations_info

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

        # Перевірка на нульові значення
        if data.isnull().any():
            self.logger.warning("Data contains NaN values. Removing them before volatility calculation.")
            data = data.dropna()

        # Перевірка на нульові або негативні значення
        if (data <= 0).any():
            min_value = data.min()
            self.logger.warning(
                f"Data contains zero or negative values (min={min_value}). These points will be excluded from volatility calculation.")
            data = data[data > 0]  # Відфільтровуємо всі нульові та від'ємні значення

        if len(data) < window:
            self.logger.error(f"Not enough data points for volatility calculation with window={window}")
            return pd.Series([], index=pd.DatetimeIndex([]))

        try:
            # Розрахунок логарифмічних прибутків
            # Захист від ділення на нуль при розрахунку відношення
            log_returns = np.log(data / data.shift(1)).dropna()

            # Розрахунок волатильності як ковзного стандартного відхилення логарифмічних прибутків
            volatility = log_returns.rolling(window=window).std()

            # Переведення стандартного відхилення у волатильність (анулізована)
            # Для різних частот даних множник буде різним:
            # - Денні дані: множник = sqrt(252) - кількість торгових днів у році
            # - Годинні дані: множник = sqrt(252 * 24)
            # - Хвилинні дані: множник = sqrt(252 * 24 * 60)

            # За замовчуванням припускаємо денні дані
            annualization_factor = np.sqrt(252)  # За замовчуванням

            if isinstance(data.index, pd.DatetimeIndex):
                # Визначаємо частоту даних
                if len(data) >= 2:
                    try:
                        time_diff = data.index[1:] - data.index[:-1]
                        median_diff = pd.Series(time_diff).median()

                        if median_diff <= pd.Timedelta(minutes=5):
                            annualization_factor = np.sqrt(252 * 24 * 12)  # 5-хвилинні дані
                        elif median_diff <= pd.Timedelta(hours=1):
                            annualization_factor = np.sqrt(252 * 24)  # Годинні дані
                        elif median_diff <= pd.Timedelta(days=1):
                            annualization_factor = np.sqrt(252)  # Денні дані
                        else:
                            # Не анулізуємо для нестандартних інтервалів
                            self.logger.warning(
                                f"Non-standard time interval detected ({median_diff}). Using default annualization factor.")
                    except Exception as e:
                        self.logger.warning(
                            f"Error determining time frequency: {str(e)}. Using default annualization factor.")

            volatility = volatility * annualization_factor

            self.logger.info(f"Volatility calculation completed. Annualization factor: {annualization_factor}")

            return volatility

        except Exception as e:
            self.logger.error(f"Error during volatility calculation: {str(e)}")
            return pd.Series([], index=data.index)