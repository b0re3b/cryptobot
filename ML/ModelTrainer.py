import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from ML.DataPreprocessor import DataPreprocessor
from ML.base import BaseDeepModel
from data.db import DatabaseManager
from utils.logger import CryptoLogger


@dataclass
class ModelConfig:
    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 2
    output_dim: int = 1
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    sequence_length: int = 60

    # Transformer specific parameters
    num_heads: int = 8
    dim_feedforward: int = 256

    @classmethod
    def get_sequence_length_for_timeframe(cls, timeframe: str) -> int:
        """
        Повертає рекомендовану довжину послідовності для таймфрейму
        """
        sequence_mapping = {
            '1m': 60,
            '1h': 24,
            '4h': 60,
            '1d': 30,
            '1w': 12
        }
        return sequence_mapping.get(timeframe, 60)


@dataclass
class CryptoConfig:
    symbols: List[str] = field(default_factory=lambda: ['BTC', 'ETH', 'SOL'])
    timeframes: List[str] = field(default_factory=lambda: ['4h'])
    model_types: List[str] = field(default_factory=lambda: ['lstm', 'gru', 'transformer'])


class ModelTrainer:

    def __init__(self, device: Optional[torch.device] = None, models_dir: str = "models"):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, BaseDeepModel] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.model_metrics: Dict[str, Dict[str, float]] = {}
        self.best_models: Dict[str, BaseDeepModel] = {}
        self.training_history: Dict[str, List[float]] = {}
        self.models_dir = models_dir
        self.db_manager = DatabaseManager()

        # Ініціалізація компонентів
        self.processor = DataPreprocessor()

        # Налаштування логування
        self.logger = CryptoLogger('trainer')

        # Створення директорії для моделей
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def _validate_model_type(self, model_type: Union[str, List[str]]) -> str:
        """
           Валідація та нормалізація типу моделі.

           Args:
               model_type (Union[str, List[str]]): Тип моделі у вигляді рядка або списку рядків.

           Returns:
               str: Валідований тип моделі у нижньому регістрі.

           Raises:
               ValueError: Якщо переданий тип моделі некоректний або список порожній.

           Логіка:
               - Якщо переданий список, береться перший елемент із попередженням.
               - Підтверджується, що тип моделі є рядком.
               - Перевіряється належність до списку допустимих типів: 'lstm', 'gru', 'transformer'.
           """
        if isinstance(model_type, list):
            if len(model_type) == 0:
                raise ValueError("Список типів моделей не може бути порожнім")
            # Якщо передано список, беремо перший елемент
            model_type = model_type[0]
            self.logger.warning(f"Передано список типів моделей, використовується перший: {model_type}")

        if not isinstance(model_type, str):
            raise ValueError(f"model_type повинен бути рядком, отримано: {type(model_type)}")

        model_type = model_type.lower().strip()

        valid_types = ['lstm', 'gru', 'transformer']
        if model_type not in valid_types:
            raise ValueError(f"Невідомий тип моделі: {model_type}. Доступні: {valid_types}")

        return model_type

    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
    Валідація та очищення даних перед тренуванням моделі.

    Args:
        data (pd.DataFrame): Вхідний DataFrame з даними для моделі.

    Returns:
        pd.DataFrame: Очищений DataFrame з заповненими пропущеними значеннями
                      та обробленими нескінченностями.

    Опис:
        - Логування початкової форми та типів колонок.
        - Ігнорування нечислових колонок.
        - Заповнення пропущених значень числових колонок медіаною.
        - Замінювання безкінечних значень (Inf) на NaN та заповнення їх медіаною.
        - Логування фінальної форми і типів колонок після очищення.
    """
        self.logger.info(f"Початкова форма даних: {data.shape}")
        self.logger.info(f"Типи колонок:\n{data.dtypes}")

        # Identify non-numeric columns to exclude
        non_numeric_cols = data.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric_cols:
            self.logger.info(f"Ignoring non-numeric columns: {non_numeric_cols}")

        # Process only numeric columns
        numeric_cols = [col for col in data.columns if col not in non_numeric_cols]

        # Перевірка на відсутні значення
        missing_counts = data[numeric_cols].isnull().sum()
        if missing_counts.any():
            self.logger.warning(f"Знайдено відсутні значення:\n{missing_counts[missing_counts > 0]}")

            # Заповнення відсутніх значень
            for col in numeric_cols:
                if data[col].isnull().any():
                    if data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                        # Для числових колонок - заповнюємо медіаною
                        data[col] = data[col].fillna(data[col].median())
                    else:
                        # Для інших типів - заповнюємо 0
                        data[col] = data[col].fillna(0)

        # Перевірка на inf значення
        inf_mask = np.isinf(data[numeric_cols].select_dtypes(include=[np.number]))
        if inf_mask.any().any():
            self.logger.warning("Знайдено безкінечні значення, замінюємо на NaN та заповнюємо")
            data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

        # Перевірка фінальних типів
        self.logger.info(f"Типи після очищення:\n{data.dtypes}")
        self.logger.info(f"Фінальна форма даних: {data.shape}")

        return data

    def _safe_tensor_conversion(self, array: np.ndarray, name: str = "array") -> torch.Tensor:
        """
    Безпечна конвертація numpy array у PyTorch tensor.

    Args:
        array (np.ndarray): Вхідний numpy масив.
        name (str): Ім'я масиву для логування.

    Returns:
        torch.Tensor: Конвертований тензор типу float32.

    Опис:
        - Якщо dtype масиву — object, спробує конвертувати в float32,
          інакше замінює масив нулями.
        - Замінює NaN та Inf значення на 0.
        - Логує інформацію про процес та можливі помилки.
        - При критичній помилці викидає виключення з детальною інформацією.
    """
        try:
            # Перевірка типу масиву
            if array.dtype == 'object':
                self.logger.error(f"{name} має тип 'object'. Спроба конвертації...")

                # Спроба конвертації кожного елементу
                try:
                    array = array.astype(np.float32)
                except (ValueError, TypeError) as e:
                    self.logger.error(f"Не вдалося конвертувати {name} в float32: {str(e)}")
                    # Створюємо масив нулів з правильними розмірами
                    array = np.zeros(array.shape, dtype=np.float32)

            # Перевірка на NaN та inf
            if np.isnan(array).any():
                self.logger.warning(f"{name} містить NaN значення, замінюємо на 0")
                array = np.nan_to_num(array, nan=0.0)

            if np.isinf(array).any():
                self.logger.warning(f"{name} містить безкінечні значення, замінюємо на 0")
                array = np.nan_to_num(array, posinf=0.0, neginf=0.0)

            # Конвертація в tensor
            tensor = torch.tensor(array, dtype=torch.float32)

            self.logger.debug(f"{name} успішно сконвертовано в tensor. Shape: {tensor.shape}, dtype: {tensor.dtype}")
            return tensor

        except Exception as e:
            self.logger.error(f"Критична помилка при конвертації {name} в tensor: {str(e)}")
            self.logger.error(f"Тип масиву: {array.dtype}, форма: {array.shape}")
            self.logger.error(f"Перші 5 значень: {array.flat[:5] if array.size > 0 else 'Порожній масив'}")
            raise

    def _prepare_sequences_for_rnn(self, X: np.ndarray, y: np.ndarray,
                                   sequence_length: int, model_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
           Підготовка послідовностей для RNN моделей (LSTM/GRU) з правильними розмірами.

           Args:
               X (np.ndarray): Вхідні ознаки у вигляді масиву (samples, features).
               y (np.ndarray): Цільові значення.
               sequence_length (int): Довжина послідовності для RNN.
               model_type (str): Тип моделі ('lstm', 'gru').

           Returns:
               Tuple[torch.Tensor, torch.Tensor]: Тензори послідовностей та цілей.

           Опис:
               - Створює послідовності довжиною sequence_length з X та відповідні цілі з y.
               - Виконує безпечну конвертацію у PyTorch тензори.
               - Переконується, що формат тензорів відповідає вимогам RNN.
               - Логує розміри вхідних даних та результату.
               - Викидає ValueError, якщо недостатньо даних для послідовностей.
           """
        self.logger.info(f"Підготовка послідовностей для {model_type} моделі")
        self.logger.info(f"Вхідні розміри - X: {X.shape}, y: {y.shape}")

        if len(X) < sequence_length:
            raise ValueError(f"Недостатньо даних для створення послідовностей довжиною {sequence_length}")

        # Створення послідовностей
        sequences = []
        targets = []

        for i in range(len(X) - sequence_length):
            # Послідовність входів
            seq = X[i:i + sequence_length]
            # Цільове значення
            target = y[i + sequence_length]

            sequences.append(seq)
            targets.append(target)

        X_sequences = np.array(sequences)
        y_targets = np.array(targets)

        self.logger.info(f"Створено послідовності - X: {X_sequences.shape}, y: {y_targets.shape}")

        # Конвертація в тензори
        X_tensor = self._safe_tensor_conversion(X_sequences, f"X_sequences_{model_type}")
        y_tensor = self._safe_tensor_conversion(y_targets, f"y_targets_{model_type}")

        # Для RNN моделей потрібен формат (batch_size, sequence_length, input_size)
        if model_type.lower() in ['lstm', 'gru']:
            # X_tensor вже має правильний формат: (batch_size, sequence_length, input_size)
            self.logger.info(f"Фінальні розміри для {model_type} - X: {X_tensor.shape}, y: {y_tensor.shape}")

        return X_tensor, y_tensor

    def _calculate_actual_input_dim(self, data: pd.DataFrame, target_column: str = 'target') -> int:
        """
    Обчислює фактичну розмірність вхідних ознак після очищення та перевірки даних.

    Аргументи:
        data (pd.DataFrame): Вхідний датафрейм з ознаками і цільовою колонкою.
        target_column (str): Бажане ім'я цільової колонки (за замовчуванням 'target').

    Повертає:
        int: Кількість числових колонок, що використовуються як ознаки (вхідна розмірність).

    Логіка:
        - Автоматично визначає цільову колонку з поширених варіантів.
        - Перевіряє, що цільова колонка має числовий тип, або намагається конвертувати.
        - Визначає список числових колонок, виключаючи цільову.
        - Логує знайдені колонки та розмірність.
    """
        # Check for target column (try common variations)
        target_col = None
        for col in ['target', 'target_close_1', 'target_close']:
            if col in data.columns:
                target_col = col
                break

        if target_col is None:
            # If no target column found, assume it's the last column or use the provided name
            if target_column in data.columns:
                target_col = target_column
            else:
                # Use the last column as target
                target_col = data.columns[-1]
                self.logger.warning(f"Target column not found, using last column: {target_col}")

        # Get numeric columns only
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        # Ensure target column is numeric
        if target_col not in numeric_cols:
            try:
                data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
                if data[target_col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    numeric_cols.append(target_col)
            except:
                pass

        # Calculate input dimension (all numeric columns except target)
        feature_cols = [col for col in numeric_cols if col != target_col]
        actual_input_dim = len(feature_cols)

        self.logger.info(f"Calculated input dimension: {actual_input_dim}")
        self.logger.info(f"Feature columns: {feature_cols}")
        self.logger.info(f"Target column: {target_col}")

        return actual_input_dim

    def train_model(self, symbol: str, timeframe: str, model_type: str,
                    data: pd.DataFrame, input_dim: int = None,
                    config: Optional[ModelConfig] = None,
                    validation_split: float = 0.2,
                    patience: int = 10, model_data=None,
                    save_after_training: bool = True,
                    target_column: str = 'target',
                    **kwargs) -> Dict[str, Any]:
        """
    Навчання моделі з покращеною валідацією, очищенням і підготовкою даних.

    Аргументи:
        symbol (str): Тікер або назва активу.
        timeframe (str): Таймфрейм даних.
        model_type (str): Тип моделі ('lstm', 'gru', 'transformer', 'mlp' тощо).
        data (pd.DataFrame): Вхідний датасет з ознаками та цільовою змінною.
        input_dim (int, optional): Вхідна розмірність. Якщо None, обчислюється автоматично.
        config (ModelConfig, optional): Конфігурація моделі.
        validation_split (float): Частка даних для валідації.
        patience (int): Кількість епох ранньої зупинки.
        model_data (optional): Додаткові дані для тренування (якщо потрібно).
        save_after_training (bool): Чи зберігати модель автоматично після тренування.
        target_column (str): Назва цільової колонки.
        **kwargs: Додаткові параметри для конфігурації.

    Повертає:
        dict: Результати тренування з інформацією про конфіг, метрики, історію, ключ моделі та колонки.

    Основні кроки:
        - Валідація вхідних даних та перевірка DataFrame.
        - Пошук цільової колонки серед поширених варіантів.
        - Перевірка і конвертація цільової колонки у числову.
        - Відбір числових колонок для ознак.
        - Автоматичне або ручне визначення input_dim з логуванням.
        - Очищення NaN та Inf в даних з різними стратегіями для ознак і цілі.
        - Перевірка варіації ознак і цілі.
        - Підготовка даних залежно від типу моделі (RNN, Transformer, інші).
        - Безпечна конвертація у тензори та перенесення на пристрій.
        - Створення і тренування моделі з логуванням прогресу.
        - Оцінка моделі та логування метрик.
        - Збереження моделі (опціонально).
    """
        # Validate and normalize model_type
        model_type = self._validate_model_type(model_type)

        self.logger.info(f"Starting training for {symbol}_{timeframe}_{model_type}")

        # Validate input data
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Expected DataFrame as input data")

        if data.empty:
            raise ValueError("Input DataFrame is empty")

        # === ВАЛІДАЦІЯ ТА ОЧИЩЕННЯ ДАНИХ ===
        self.logger.info("Валідація та очищення вхідних даних...")
        data = self._validate_and_clean_data(data)

        # === ПОШУК ЦІЛЬОВОЇ КОЛОНКИ ===
        target_col = None
        possible_targets = ['target', 'target_close_1', 'target_close', target_column]

        for col in possible_targets:
            if col in data.columns:
                target_col = col
                break

        if target_col is None:
            raise ValueError(
                f"DataFrame must contain a target column. Tried: {possible_targets}. "
                f"Available columns: {list(data.columns)}"
            )

        self.logger.info(f"Using target column: '{target_col}'")

        # === ПІДГОТОВКА ЧИСЛОВИХ ДАНИХ ===
        self.logger.info("Відбір та валідація числових колонок...")

        # Отримання числових колонок (крім цільової)
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        # Перевірка та конвертація цільової колонки
        if target_col not in numeric_cols:
            try:
                data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
                if pd.api.types.is_numeric_dtype(data[target_col]):
                    self.logger.info(f"Successfully converted target column '{target_col}' to numeric")
                else:
                    raise ValueError(f"Target column '{target_col}' cannot be converted to numeric type")
            except Exception as e:
                raise ValueError(f"Target column '{target_col}' is not numeric and cannot be converted: {str(e)}")

        # Видалення цільової колонки зі списку ознак
        feature_cols = [col for col in numeric_cols if col != target_col]

        if not feature_cols:
            raise ValueError("No numeric feature columns found after filtering")

        self.logger.info(f"Using {len(feature_cols)} feature columns: {feature_cols}")
        self.logger.info(f"Target column: {target_col}")

        # === АВТОМАТИЧНЕ ОБЧИСЛЕННЯ INPUT_DIM ===
        actual_input_dim = len(feature_cols)

        if input_dim is None:
            input_dim = actual_input_dim
            self.logger.info(f"Автоматично обчислено input_dim: {input_dim}")
        else:
            if input_dim != actual_input_dim:
                self.logger.warning(
                    f"Переданий input_dim ({input_dim}) не відповідає фактичному ({actual_input_dim}). "
                    f"Використовується фактичний: {actual_input_dim}"
                )
                input_dim = actual_input_dim

        # === СТВОРЕННЯ КОНФІГУРАЦІЇ МОДЕЛІ ===
        if config is None:
            config = self.create_model_config(symbol, timeframe, model_type, input_dim, **kwargs)
        else:
            # Оновлення конфігурації з правильним input_dim
            config.input_dim = input_dim

        # Валідація мінімальних вимог до даних
        min_samples_required = max(config.sequence_length * 2, 100)  # Мінімум для навчання
        if len(data) < min_samples_required:
            raise ValueError(
                f"Insufficient data for training. Required: {min_samples_required}, "
                f"available: {len(data)}"
            )

        # === ПІДГОТОВКА ДАНИХ ===
        # Створення DataFrame тільки з потрібними колонками
        training_data = data[feature_cols + [target_col]].copy()

        # Перевірка та обробка NaN значень
        nan_counts = training_data.isnull().sum()
        if nan_counts.any():
            self.logger.warning(f"Виявлено NaN значення: {nan_counts[nan_counts > 0].to_dict()}")

            # Різні стратегії для різних типів колонок
            for col in training_data.columns:
                if training_data[col].isnull().any():
                    if col == target_col:
                        # Для цільової змінної використовуємо медіану
                        training_data[col] = training_data[col].fillna(training_data[col].median())
                    else:
                        # Для ознак використовуємо медіану або forward fill
                        training_data[col] = training_data[col].fillna(training_data[col].median())

        # Фінальна перевірка на NaN
        if training_data.isnull().any().any():
            self.logger.warning("Залишились NaN значення після заповнення, замінюємо на 0")
            training_data = training_data.fillna(0)

        # Підготовка X та y
        X = training_data[feature_cols].values.astype(np.float32)
        y = training_data[target_col].values.astype(np.float32)

        # === ДОДАТКОВА ВАЛІДАЦІЯ ДАНИХ ===
        self.logger.info(f"Фінальні розміри даних - X: {X.shape}, y: {y.shape}")

        # Перевірка на некоректні значення
        if np.isnan(X).any() or np.isinf(X).any():
            self.logger.warning("Виявлено NaN/Inf значення в X, очищаємо...")
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        if np.isnan(y).any() or np.isinf(y).any():
            self.logger.warning("Виявлено NaN/Inf значення в y, очищаємо...")
            y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

        # Перевірка варіації в даних
        if np.std(X) == 0:
            self.logger.warning("Відсутня варіація в ознаках X")
        if np.std(y) == 0:
            self.logger.warning("Відсутня варіація в цільовій змінній y")

        # === ПІДГОТОВКА ДАНИХ ДЛЯ РІЗНИХ ТИПІВ МОДЕЛЕЙ ===
        if model_type.lower() == 'transformer':
            try:
                # Для Transformer використовуємо спеціальну підготовку даних
                prepared_data = self.processor.prepare_data_with_config(
                    symbol=symbol,
                    timeframe=timeframe,
                    model_type=model_type,
                    validation_split=validation_split,
                    target_column=target_column
                )

                if len(prepared_data) == 5:
                    X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, transformer_config = prepared_data
                    # Оновлюємо конфігурацію якщо потрібно
                    if transformer_config and hasattr(transformer_config, 'input_dim'):
                        config.input_dim = transformer_config.input_dim
                else:
                    raise ValueError(f"Transformer data preparation returned {len(prepared_data)} values, expected 5")

            except Exception as e:
                self.logger.error(f"Transformer data preparation failed: {str(e)}")
                raise

        elif model_type.lower() in ['lstm', 'gru']:
            # === ПІДГОТОВКА ПОСЛІДОВНОСТЕЙ ДЛЯ RNN МОДЕЛЕЙ ===
            self.logger.info(f"Підготовка послідовностей для {model_type} моделі...")

            try:
                X_tensor, y_tensor = self._prepare_sequences_for_rnn(
                    X, y, config.sequence_length, model_type
                )

                if len(X_tensor) == 0:
                    raise ValueError("No sequences generated for RNN model")

                self.logger.info(f"Створено {len(X_tensor)} послідовностей")

                # Розділення на тренувальну та валідаційну вибірки
                split_idx = max(1, int(len(X_tensor) * (1 - validation_split)))

                X_train_tensor = X_tensor[:split_idx].to(self.device)
                y_train_tensor = y_tensor[:split_idx].to(self.device)
                X_val_tensor = X_tensor[split_idx:].to(self.device)
                y_val_tensor = y_tensor[split_idx:].to(self.device)

            except Exception as e:
                self.logger.error(f"RNN sequence preparation failed: {str(e)}")
                raise

        else:
            # === ДЛЯ ІНШИХ МОДЕЛЕЙ (MLP, CNN тощо) ===
            self.logger.info(f"Підготовка даних для {model_type} моделі...")

            # Розділення на тренувальну та валідаційну вибірки
            split_idx = max(1, int(len(X) * (1 - validation_split)))

            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Перевірка мінімальної кількості даних
            if len(X_train) < 10 or len(X_val) < 5:
                raise ValueError(f"Insufficient data after split. Train: {len(X_train)}, Val: {len(X_val)}")

            # === БЕЗПЕЧНА КОНВЕРТАЦІЯ В ТЕНЗОРИ ===
            self.logger.info("Конвертація даних в тензори...")
            try:
                X_train_tensor = self._safe_tensor_conversion(X_train, "X_train").to(self.device)
                y_train_tensor = self._safe_tensor_conversion(y_train, "y_train").to(self.device)
                X_val_tensor = self._safe_tensor_conversion(X_val, "X_val").to(self.device)
                y_val_tensor = self._safe_tensor_conversion(y_val, "y_val").to(self.device)
            except Exception as e:
                self.logger.error(f"Tensor conversion failed: {str(e)}")
                raise

        # === СТВОРЕННЯ ТА ТРЕНУВАННЯ МОДЕЛІ ===
        self.logger.info("Створення моделі...")
        try:
            model = self._build_model_from_config(model_type, config).to(self.device)
            self.logger.info(f"Model created successfully. Parameters: {sum(p.numel() for p in model.parameters())}")
        except Exception as e:
            self.logger.error(f"Model creation failed: {str(e)}")
            raise

        # Логування форм тензорів перед тренуванням
        self.logger.info(f"Training data shapes:")
        self.logger.info(f"  X_train: {X_train_tensor.shape}")
        self.logger.info(f"  y_train: {y_train_tensor.shape}")
        self.logger.info(f"  X_val: {X_val_tensor.shape}")
        self.logger.info(f"  y_val: {y_val_tensor.shape}")

        # Тренування моделі
        self.logger.info("Початок тренування...")
        try:
            history = self._train_loop(
                model=model,
                train_data=(X_train_tensor, y_train_tensor),
                val_data=(X_val_tensor, y_val_tensor),
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                patience=patience
            )
            self.logger.info("Тренування завершено успішно")
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

        # Оцінка моделі
        self.logger.info("Оцінка моделі...")
        try:
            metrics = self.evaluate(model, (X_val_tensor, y_val_tensor))
            self.logger.info(f"Model evaluation completed. Metrics: {metrics}")
        except Exception as e:
            self.logger.warning(f"Model evaluation failed: {str(e)}")
            metrics = {}

        # Збереження моделі
        model_key = self._create_model_key(symbol, timeframe, model_type)
        self.models[model_key] = model
        self.model_configs[model_key] = config
        self.model_metrics[model_key] = metrics
        self.training_history[model_key] = history

        # Автоматичне збереження моделі після тренування
        if save_after_training:
            try:
                model_path = self.save_model(symbol, timeframe, model_type)
                self.logger.info(f"Model automatically saved: {model_path}")
            except Exception as e:
                self.logger.error(f"Error during auto-saving model: {str(e)}")

        return {
            'config': config.__dict__,
            'metrics': metrics,
            'history': history,
            'model_key': model_key,
            'input_dim': input_dim,
            'feature_columns': feature_cols,
            'target_column': target_col
        }

    def create_model_config(self, symbol: str, timeframe: str, model_type: str,
                            input_dim: int, **kwargs) -> ModelConfig:
        """
            Створює конфігурацію моделі з параметрами, специфічними для типу моделі.

            Args:
                symbol (str): Символ фінансового інструменту (наприклад, 'BTCUSD').
                timeframe (str): Таймфрейм даних (наприклад, '1h', '15m').
                model_type (str): Тип моделі ('transformer', 'lstm', 'gru' тощо).
                input_dim (int): Розмірність вхідних даних (кількість фіч).
                **kwargs: Додаткові параметри конфігурації, наприклад:
                    - hidden_dim (int): Розмірність прихованого шару.
                    - num_heads (int): Кількість голів (для трансформера).
                    - dim_feedforward (int): Розмірність feedforward шару (для трансформера).
                    - num_layers (int): Кількість шарів.
                    - learning_rate (float): Крок навчання.

            Returns:
                ModelConfig: Об'єкт конфігурації моделі з заданими параметрами.
            """
        # Валідація та нормалізація model_type
        model_type = self._validate_model_type(model_type)

        # Базова конфігурація
        config = ModelConfig(
            input_dim=input_dim,
            sequence_length=ModelConfig.get_sequence_length_for_timeframe(timeframe)
        )

        # Специфічні налаштування для різних типів моделей
        if model_type == 'transformer':
            config.hidden_dim = kwargs.get('hidden_dim', 128)
            config.num_heads = kwargs.get('num_heads', 8)
            config.dim_feedforward = kwargs.get('dim_feedforward', 256)
            config.num_layers = kwargs.get('num_layers', 3)
            config.learning_rate = kwargs.get('learning_rate', 0.0005)
        elif model_type in ['lstm', 'gru']:
            config.hidden_dim = kwargs.get('hidden_dim', 64)
            config.num_layers = kwargs.get('num_layers', 2)
            config.learning_rate = kwargs.get('learning_rate', 0.001)

        # Перевизначення будь-яких параметрів
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def _prepare_sequence_data(self, data: pd.DataFrame, sequence_length: int,
                               batch_size: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
            Підготовлює послідовності ознак та цільових значень для тренування моделей послідовностей.

            Args:
                data (pd.DataFrame): Вхідний датафрейм із колонкою "target" та фічами.
                sequence_length (int): Довжина послідовності, яку формуємо для моделі.
                batch_size (int, optional): Розмір батчу для послідовної обробки (щоб уникнути перевантаження пам’яті). За замовчуванням 10000.

            Returns:
                Tuple[np.ndarray, np.ndarray]: Кортеж з двох numpy масивів:
                    - X: Масив послідовностей розміром (num_sequences, sequence_length, num_features).
                    - y: Масив цільових значень наступного кроку (num_sequences,).
                    Якщо вхідних даних недостатньо для формування хоча б однієї послідовності, повертає порожні масиви.
            """
        features = data.drop(columns=["target"]).values
        targets = data["target"].values

        total_sequences = len(features) - sequence_length
        if total_sequences <= 0:
            return np.array([]), np.array([])

        # Розраховуємо кількість батчів
        num_batches = (total_sequences + batch_size - 1) // batch_size

        X_batches = []
        y_batches = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_sequences)

            X_batch = []
            y_batch = []

            for i in range(start_idx, end_idx):
                seq_start = i
                seq_end = i + sequence_length
                X_batch.append(features[seq_start:seq_end])
                y_batch.append(targets[seq_end])

            X_batches.append(np.array(X_batch))
            y_batches.append(np.array(y_batch))

            # Логування прогресу
            if batch_idx % 10 == 0:
                self.logger.debug(f"Оброблено батч {batch_idx + 1}/{num_batches}")

        # Об'єднуємо всі батчі
        X = np.vstack(X_batches) if X_batches else np.array([])
        y = np.concatenate(y_batches) if y_batches else np.array([])

        self.logger.info(f"Створено {len(X)} послідовностей з {len(data)} рядків даних")
        return X, y

    def _build_model_from_config(self, model_type: str, config: ModelConfig) -> BaseDeepModel:
        """
    Створює екземпляр моделі на основі конфігурації та типу моделі.

    Args:
        model_type (str): Тип моделі ('lstm', 'gru', 'transformer').
        config (ModelConfig): Об'єкт конфігурації моделі з необхідними параметрами.

    Raises:
        ValueError: Якщо тип моделі невідомий.
        ImportError: Якщо не вдалося імпортувати потрібний клас моделі.

    Returns:
        BaseDeepModel: Ініціалізована модель відповідного типу.
    """
        # Валідація та нормалізація model_type
        model_type = self._validate_model_type(model_type)

        try:
            if model_type == 'lstm':
                from ML.LSTM import LSTMModel
                return LSTMModel(
                    input_dim=config.input_dim,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    output_dim=config.output_dim,
                    dropout=config.dropout
                )
            elif model_type == 'gru':
                from ML.GRU import GRUModel
                return GRUModel(
                    input_dim=config.input_dim,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    output_dim=config.output_dim,
                    dropout=config.dropout
                )
            elif model_type == 'transformer':
                from ML.transformer import TransformerModel
                return TransformerModel(
                    input_dim=config.input_dim,
                    hidden_dim=config.hidden_dim,
                    n_heads=config.num_heads,
                    num_layers=config.num_layers,
                    dim_feedforward=config.dim_feedforward,
                    output_dim=config.output_dim,
                    dropout=config.dropout,
                )
            else:
                raise ValueError(f"Невідомий тип моделі: {model_type}")
        except ImportError as e:
            self.logger.error(f"Помилка імпорту моделі {model_type}: {str(e)}")
            raise

    def _create_model_data_dict(self, symbol: str, timeframe: str, model_type: str,
                                config: ModelConfig) -> Dict[str, Any]:
        """
           Формує словник з метаданими та параметрами моделі для збереження або подальшої роботи.

           Args:
               symbol (str): Символ фінансового інструменту.
               timeframe (str): Таймфрейм даних.
               model_type (str): Тип моделі.
               config (ModelConfig): Конфігурація моделі.

           Returns:
               Dict[str, Any]: Словник з інформацією про модель, шляхом до файлу, параметрами і станом.
           """
        # Валідація та нормалізація model_type
        model_type = self._validate_model_type(model_type)

        model_path = os.path.join(self.models_dir, symbol, timeframe, f"{model_type}_model.pth")

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'model_type': model_type,
            'model_version': '1.0',
            'model_path': model_path,
            'input_features': list(range(config.input_dim)),
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers,
            'sequence_length': config.sequence_length,
            'num_heads': getattr(config, 'num_heads', None),
            'dim_feedforward': getattr(config, 'dim_feedforward', None),
            'active': True
        }

    def train_epoch(self, model: BaseDeepModel, train_loader: torch.utils.data.DataLoader,
                    optimizer, criterion) -> float:
        """
           Навчає модель на одному проході (епосі) по всіх батчах даталоадера.

           Args:
               model (BaseDeepModel): Модель для навчання.
               train_loader (torch.utils.data.DataLoader): Даталоадер для тренувальних даних.
               optimizer: Оптимізатор PyTorch.
               criterion: Функція втрат (loss function).

           Returns:
               float: Середнє значення втрат (loss) за епоху.
           """
        model.train()
        total_loss = 0
        batch_count = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            optimizer.zero_grad()
            output = model(X_batch).squeeze()
            loss = criterion(output, y_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

            # Очищення пам'яті після кожного батчу
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return total_loss / batch_count if batch_count > 0 else 0

    def validate_epoch(self, model: BaseDeepModel, val_loader: torch.utils.data.DataLoader,
                       criterion) -> float:
        """
    Виконує валідацію моделі на одному проході (епосі) по валідаційному даталоадеру.

    Args:
        model (BaseDeepModel): Модель для валідації.
        val_loader (torch.utils.data.DataLoader): Даталоадер з валідаційними даними.
        criterion: Функція втрат для обчислення помилки.

    Returns:
        float: Середнє значення втрат (loss) на валідації за епоху.
    """
        model.eval()
        total_loss = 0
        batch_count = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                output = model(X_batch).squeeze()
                loss = criterion(output, y_batch)
                total_loss += loss.item()
                batch_count += 1

                # Очищення пам'яті після кожного батчу
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return total_loss / batch_count if batch_count > 0 else 0

    def _train_loop(self, model: BaseDeepModel, train_data: Tuple[torch.Tensor, torch.Tensor],
                    val_data: Tuple[torch.Tensor, torch.Tensor], epochs: int,
                    batch_size: int, learning_rate: float, patience: int) -> Dict[str, List[float]]:
        """
    Основний цикл навчання моделі з підтримкою ранньої зупинки (early stopping).

    Args:
        model (BaseDeepModel): Модель для навчання.
        train_data (Tuple[torch.Tensor, torch.Tensor]): Тренувальні дані (X_train, y_train).
        val_data (Tuple[torch.Tensor, torch.Tensor]): Валідаційні дані (X_val, y_val).
        epochs (int): Максимальна кількість епох для тренування.
        batch_size (int): Розмір батчу.
        learning_rate (float): Швидкість навчання для оптимізатора.
        patience (int): Кількість епох без покращення валідаційної втрати для ранньої зупинки.

    Returns:
        Dict[str, List[float]]: Історія навчання з ключами 'train_loss' та 'val_loss'.
    """
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*train_data),
            batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*val_data),
            batch_size=batch_size
        )

        optimizer, criterion = self.setup_optimizer_and_loss(model, learning_rate)

        # Історія навчання
        history = {
            'train_loss': [],
            'val_loss': []
        }

        best_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # Тренування - передаємо train_loader
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)

            # Валідація - передаємо val_loader
            val_loss = self.validate_epoch(model, val_loader, criterion)

            # Додаємо до історії
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            # Early stopping logic
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.logger.info(f"Early stopping на епосі {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Епоха {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Завантаження найкращих ваг
        if best_model_state:
            model.load_state_dict(best_model_state)

        return history

    def evaluate(self, model: BaseDeepModel, test_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
           Оцінює модель на тестовому наборі даних.

           Args:
               model (BaseDeepModel): Навчена модель для оцінки.
               test_data (Tuple[torch.Tensor, torch.Tensor]): Кортеж з тестовими даними (вхідні дані, цільові значення).

           Returns:
               Dict[str, float]: Словник з обчисленими метриками якості моделі.
           """
        model.eval()
        X_test, y_true = test_data

        with torch.no_grad():
            y_pred = model(X_test).squeeze().cpu().numpy()
            y_true_np = y_true.cpu().numpy()

        return self.calculate_metrics(y_true_np, y_pred)

    def evaluate_batched(self, model: BaseDeepModel, test_data: Tuple[torch.Tensor, torch.Tensor],
                         batch_size: int = 1000) -> Dict[str, float]:
        """
           Оцінює модель на тестових даних з обробкою по батчах, що корисно при великих обсягах даних.

           Args:
               model (BaseDeepModel): Навчена модель для оцінки.
               test_data (Tuple[torch.Tensor, torch.Tensor]): Кортеж з тестовими даними (вхідні дані, цільові значення).
               batch_size (int, optional): Розмір батчу для оцінки. За замовчуванням 1000.

           Returns:
               Dict[str, float]: Словник з обчисленими метриками якості моделі.
           """
        model.eval()
        X_test, y_true = test_data

        # Якщо дані невеликі, використовуємо звичайну оцінку
        if len(X_test) <= batch_size:
            return self.evaluate(model, test_data)

        self.logger.info(f"Початок батчевої оцінки на {len(X_test)} зразках")

        all_predictions = []

        # Створюємо DataLoader для батчевої обробки
        test_dataset = torch.utils.data.TensorDataset(X_test, y_true)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        with torch.no_grad():
            for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
                X_batch = X_batch.to(self.device)

                # Отримання прогнозів для батчу
                y_pred_batch = model(X_batch).squeeze().cpu().numpy()
                all_predictions.append(y_pred_batch)

                # Очищення пам'яті GPU після кожного батчу
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if batch_idx % 50 == 0:
                    self.logger.debug(f"Оброблено батч {batch_idx + 1}/{len(test_loader)}")

        # Об'єднання всіх прогнозів
        y_pred = np.concatenate(all_predictions)
        y_true_np = y_true.cpu().numpy()

        metrics = self.calculate_metrics(y_true_np, y_pred)
        self.logger.info(f"Батчева оцінка завершена. RMSE: {metrics.get('RMSE', 'N/A'):.6f}")

        return metrics

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
            Обчислює основні метрики якості прогнозу.

            Args:
                y_true (np.ndarray): Масив істинних значень.
                y_pred (np.ndarray): Масив передбачених значень.

            Returns:
                Dict[str, float]: Словник з метриками:
                    - MSE (Mean Squared Error)
                    - RMSE (Root Mean Squared Error)
                    - MAE (Mean Absolute Error)
                    - MAPE (Mean Absolute Percentage Error)
                    - R2 (Coefficient of Determination)
            """
        # Обробка випадків з нульовими значеннями для MAPE
        epsilon = 1e-8
        y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)

        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

        # Додаткові метрики
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

        return {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MAPE': float(mape),
            'R2': float(r2)
        }

    def setup_optimizer_and_loss(self, model: BaseDeepModel, learning_rate: float):
        """
            Налаштовує оптимізатор та функцію втрат для моделі.

            Args:
                model (BaseDeepModel): Модель для навчання.
                learning_rate (float): Швидкість навчання.

            Returns:
                Tuple[torch.optim.Optimizer, torch.nn.Module]: Оптимізатор і функція втрат (MSELoss).
            """
        # Для Transformer моделей можна використовувати AdamW
        if hasattr(model, 'model_type') and model.model_type == 'transformer':
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        criterion = nn.MSELoss()
        return optimizer, criterion

    def online_learning(self, symbol: str, timeframe: str, model_type: str,
                        new_data: pd.DataFrame, input_dim: int = None,
                        epochs: int = 10, learning_rate: float = 0.0005,
                        model_data=None,
                        save_after_update: bool = True,
                        **kwargs) -> Dict[str, Any]:
        """
    Онлайн навчання існуючої моделі на нових даних.

    Args:
        symbol (str): Тікер або символ активу.
        timeframe (str): Таймфрейм даних (наприклад, '1h', '1d').
        model_type (str): Тип моделі ('lstm', 'gru', 'transformer' тощо).
        new_data (pd.DataFrame): Новий датасет для додаткового навчання, повинен містити колонку 'target'.
        input_dim (int, optional): Розмір вхідного вектора (необов’язковий).
        epochs (int, optional): Кількість епох для додаткового навчання. За замовчуванням 10.
        learning_rate (float, optional): Швидкість навчання. За замовчуванням 0.0005.
        model_data (optional): Додаткові параметри моделі (не використовується в коді).
        save_after_update (bool, optional): Чи зберігати модель після навчання. За замовчуванням True.
        **kwargs: Додаткові параметри.

    Returns:
        Dict[str, Any]: Словник з історією навчання і метриками якості моделі після оновлення.
    """
        # Валідація та нормалізація model_type
        model_type = self._validate_model_type(model_type)

        model_key = self._create_model_key(symbol, timeframe, model_type)

        # Завантаження моделі, якщо вона не в пам'яті
        if model_key not in self.models:
            if not self.load_model(symbol, timeframe, model_type):
                raise ValueError(f"Модель {model_key} не знайдена")

        model = self.models[model_key]
        model_config = self.model_configs[model_key]

        # Валідація вхідних даних
        if not isinstance(new_data, pd.DataFrame):
            raise ValueError("Очікується DataFrame як вхідні дані")

        if 'target' not in new_data.columns:
            raise ValueError("DataFrame повинен містити колонку 'target'")

        # Підготовка даних
        X = new_data.drop(columns=['target']).values
        y = new_data['target'].values

        # Конвертація в тензори
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        # Налаштування оптимізатора
        optimizer, criterion = self.setup_optimizer_and_loss(model, learning_rate)

        # Історія навчання
        history = {'train_loss': []}

        # Навчання
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor).squeeze()
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            history['train_loss'].append(loss.item())

            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Епоха {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

        # Оцінка моделі
        metrics = self.evaluate(model, (X_tensor, y_tensor))
        self.model_metrics[model_key] = metrics

        # Автоматичне збереження оновленої моделі
        if save_after_update:
            try:
                model_path = self.save_model(symbol, timeframe, model_type)
                self.logger.info(f"Оновлена модель автоматично збережена: {model_path}")
            except Exception as e:
                self.logger.error(f"Помилка при автоматичному збереженні оновленої моделі: {str(e)}")

        return {
            'metrics': metrics,
            'history': history
        }

    def _build_model(self, model_type: str, input_dim: int, hidden_dim: int,
                     num_layers: int) -> BaseDeepModel:
        """
            Створює модель на основі переданих параметрів.

            Args:
                model_type (str): Тип моделі ('lstm', 'gru', 'transformer' тощо).
                input_dim (int): Розмірність вхідних даних.
                hidden_dim (int): Розмірність прихованого шару.
                num_layers (int): Кількість шарів у моделі.

            Returns:
                BaseDeepModel: Ініціалізований об'єкт моделі відповідного типу.
            """
        # Валідація та нормалізація model_type
        model_type = self._validate_model_type(model_type)

        config = ModelConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        return self._build_model_from_config(model_type, config)

    def _create_model_key(self, symbol: str, timeframe: str, model_type: str) -> str:
        """
            Генерує унікальний ключ моделі для ідентифікації.

            Args:
                symbol (str): Тікер або символ активу.
                timeframe (str): Таймфрейм даних.
                model_type (str): Тип моделі.

            Returns:
                str: Унікальний ключ у форматі "{symbol}_{timeframe}_{model_type}".
            """
        # Валідація та нормалізація model_type
        model_type = self._validate_model_type(model_type)

        return f"{symbol}_{timeframe}_{model_type}"

    def save_model(self, symbol: str, timeframe: str, model_type: str) -> str:
        """
            Зберігає модель на диск та оновлює запис у базі даних.

            Args:
                symbol (str): Тікер або символ активу.
                timeframe (str): Таймфрейм даних.
                model_type (str): Тип моделі.

            Raises:
                ValueError: Якщо модель не знайдена в пам'яті.

            Returns:
                str: Шлях до збереженого файлу моделі.
            """
        # Валідація та нормалізація model_type
        model_type = self._validate_model_type(model_type)

        model_key = self._create_model_key(symbol, timeframe, model_type)
        if model_key not in self.models:
            raise ValueError(f"Модель {model_key} не знайдена")

        # Створюємо директорію для моделі
        model_dir = os.path.join(self.models_dir, symbol, timeframe)
        os.makedirs(model_dir, exist_ok=True)

        # Шлях до файлу моделі
        model_path = os.path.join(model_dir, f"{model_type}_model.pth")

        try:
            # Зберігаємо модель
            torch.save({
                'model_state_dict': self.models[model_key].state_dict(),
                'config': self.model_configs[model_key].__dict__,
                'metrics': self.model_metrics.get(model_key, {}),
                'training_history': self.training_history.get(model_key, [])
            }, model_path)

            # Підготовлюємо дані для збереження в БД
            config = self.model_configs[model_key]
            model_data = self._create_model_data_dict(symbol, timeframe, model_type, config)

            # Зберігаємо інформацію про модель в БД
            model_id = self.db_manager.save_ml_model(model_data)

            self.logger.info(f"Модель {model_key} збережена за шляхом: {model_path} з ID: {model_id}")
            return model_path

        except Exception as e:
            self.logger.error(f"Помилка при збереженні моделі {model_key}: {str(e)}")
            raise

    def load_model(self, symbol: str, timeframe: str, model_type: str) -> bool:
        """
           Завантажує модель з диску за заданими параметрами.

           Args:
               symbol (str): Тікер або символ активу.
               timeframe (str): Таймфрейм даних.
               model_type (str): Тип моделі.

           Returns:
               bool: True, якщо модель успішно завантажена, інакше False.
           """
        model_key = self._create_model_key(symbol, timeframe, model_type)

        # Шлях до файлу моделі
        model_dir = os.path.join(self.models_dir, symbol, timeframe)
        model_path = os.path.join(model_dir, f"{model_type.lower()}_model.pth")

        if not os.path.exists(model_path):
            self.logger.warning(f"Модель {model_key} не знайдена за шляхом {model_path}")
            return False

        try:
            # Завантаження моделі
            checkpoint = torch.load(model_path, map_location=self.device)

            # Відновлення конфігурації
            config_dict = checkpoint['config']
            config = ModelConfig(**config_dict)

            # Створення моделі
            model = self._build_model_from_config(model_type, config)

            # Завантаження ваг моделі
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)

            # Збереження моделі та її конфігурації
            self.models[model_key] = model
            self.model_configs[model_key] = config
            self.model_metrics[model_key] = checkpoint.get('metrics', {})
            self.training_history[model_key] = checkpoint.get('training_history', [])

            self.logger.info(f"Модель {model_key} успішно завантажена")
            return True

        except Exception as e:
            self.logger.error(f"Помилка при завантаженні моделі {model_key}: {str(e)}")
            return False

    # ==================== НАВЧАННЯ МОДЕЛЕЙ ====================

    def train_all_models(self, symbols: Optional[List[str]] = None,
                         timeframes: Optional[List[str]] = None,
                         model_types: Optional[List[str]] = None,
                         save_models: bool = True,
                         **training_params) -> Dict[str, Dict[str, Any]]:
        """
            Запускає послідовне навчання множини моделей по заданих символах, таймфреймах та типах моделей.

            Args:
                symbols (Optional[List[str]]): Список символів (за замовчуванням беруться з конфігурації).
                timeframes (Optional[List[str]]): Список таймфреймів (за замовчуванням беруться з конфігурації).
                model_types (Optional[List[str]]): Список типів моделей (за замовчуванням беруться з конфігурації).
                save_models (bool): Чи зберігати моделі після навчання. За замовчуванням True.
                **training_params: Додаткові параметри, які передаються в метод навчання окремої моделі.

            Returns:
                Dict[str, Dict[str, Any]]: Результати навчання кожної моделі у вигляді словника, ключ — унікальний ключ моделі,
                значення — словник з метриками, історією, шляхом збереження або помилками.
            """
        config = CryptoConfig()
        symbols = symbols or config.symbols
        timeframes = timeframes or config.timeframes
        model_types = model_types or config.model_types

        results = {}
        total_models = len(symbols) * len(timeframes) * len(model_types)
        current_model = 0
        saved_models = 0
        failed_saves = 0

        self.logger.info(f"Starting training of {total_models} models")

        for symbol in symbols:
            for timeframe in timeframes:
                for model_type in model_types:
                    current_model += 1
                    model_key = self._create_model_key(symbol, timeframe, model_type)

                    try:
                        self.logger.info(f"Training model {current_model}/{total_models}: {model_key}")

                        # Get data
                        data_loader = self.processor.get_data_loader(symbol, timeframe, model_type)
                        raw_data = data_loader()

                        # Convert list to DataFrame if needed
                        if isinstance(raw_data, list):
                            if not raw_data:
                                raise ValueError(f"Empty data list for {symbol}-{timeframe}")
                            data = pd.DataFrame(raw_data)
                        elif isinstance(raw_data, pd.DataFrame):
                            data = raw_data.copy()
                        else:
                            raise ValueError(f"Unsupported data type: {type(raw_data)}")

                        if data.empty:
                            raise ValueError(f"Empty DataFrame for {symbol}-{timeframe}")

                        # Determine input dimension
                        input_dim = data.shape[1] - 1  # Subtract target column

                        # Train model (without auto-saving)
                        result = self.train_model(
                            symbol=symbol,
                            timeframe=timeframe,
                            model_type=model_type,
                            data=data,
                            input_dim=input_dim,
                            save_after_training=False,
                            **training_params
                        )

                        results[model_key] = result
                        self.logger.info(
                            f"Model {model_key} trained successfully. RMSE: {result['metrics']['RMSE']:.6f}")

                        # Save model if needed
                        if save_models:
                            try:
                                model_path = self.save_model(symbol, timeframe, model_type)
                                saved_models += 1
                                results[model_key]['model_path'] = model_path
                                self.logger.info(f"Model {model_key} saved: {model_path}")
                            except Exception as save_error:
                                failed_saves += 1
                                self.logger.error(f"Error saving model {model_key}: {str(save_error)}")
                                results[model_key]['save_error'] = str(save_error)

                    except Exception as e:
                        self.logger.error(f"Error training model {model_key}: {str(e)}")
                        results[model_key] = {'error': str(e)}

        successful_models = len([r for r in results.values() if 'error' not in r])

        summary_msg = f"Training completed. Successful: {successful_models}/{total_models}"
        if save_models:
            summary_msg += f". Saved: {saved_models}, Save errors: {failed_saves}"

        self.logger.info(summary_msg)

        return results


    def cross_validate_model(self, symbol: str, timeframe: str, model_type: str,
                             k_folds: int = 5, batch_size: int = 1000,
                             **model_params) -> Dict[str, List[float]]:
        """
            Виконує батчеву крос-валідацію моделі з підтримкою великих обсягів даних.

            Args:
                symbol (str): Символ активу для моделювання.
                timeframe (str): Таймфрейм даних.
                model_type (str): Тип моделі.
                k_folds (int, optional): Кількість фолдів у крос-валідації. За замовчуванням 5.
                batch_size (int, optional): Розмір батчу для навчання і валідації. За замовчуванням 1000.
                **model_params: Додаткові параметри моделі (наприклад, learning_rate, epochs, patience тощо).

            Returns:
                Dict[str, List[float]]: Результати крос-валідації з ключами у вигляді метрик
                та значеннями - списками показників по фолдах, а також середні і стандартні відхилення.
            """
        self.logger.info(f"Початок батчевої крос-валідації для {symbol}_{timeframe}_{model_type}")

        # Завантаження та підготовка даних
        data = self.processor.get_data_loader(symbol, timeframe, model_type)
        if data is None:
            raise ValueError(f"Не вдалося завантажити дані для {symbol}_{timeframe}")

        features = self.processor.prepare_data_with_config(data, symbol, model_type)
        if not isinstance(features, dict) or 'X' not in features or 'y' not in features:
            raise ValueError("Очікувався словник з ключами 'X' та 'y' від prepare_features")

        X_data = features['X']
        y_data = features['y']

        input_dim = X_data.shape[1]
        config = self.create_model_config(symbol, timeframe, model_type, input_dim, **model_params)

        # Підготовка послідовностей з батчевою обробкою
        X, y = self._prepare_sequence_data(X_data, y_data, config.sequence_length, batch_size)

        if len(X) == 0:
            raise ValueError("Недостатньо даних для створення послідовностей")

        # Ініціалізація крос-валідації
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        # Результати для кожного фолда
        fold_results = {
            'MSE': [],
            'RMSE': [],
            'MAE': [],
            'MAPE': [],
            'R2': []
        }

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            self.logger.info(f"Обробка фолда {fold + 1}/{k_folds}")

            # Розділення даних
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Створення моделі
            model = self._build_model_from_config(model_type, config).to(self.device)

            # Підготовка тензорів
            train_tensor = (torch.tensor(X_train).float(), torch.tensor(y_train).float())
            val_tensor = (torch.tensor(X_val).float(), torch.tensor(y_val).float())

            # Навчання моделі з батчевою обробкою
            optimizer, criterion = self.setup_optimizer_and_loss(model, config.learning_rate)

            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(config.epochs):
                # Навчання
                model.train()
                train_loss = self.train_epoch(model, train_tensor, optimizer, criterion, batch_size)

                # Валідація
                model.eval()
                val_loss = self.validate_epoch(model, val_tensor, criterion, batch_size)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= model_params.get('patience', 10):
                    self.logger.debug(f"Early stopping на епосі {epoch + 1} для фолда {fold + 1}")
                    break

            # Оцінка
            metrics = self.evaluate_batched(model, val_tensor, batch_size)

            for metric_name, value in metrics.items():
                if metric_name in fold_results:
                    fold_results[metric_name].append(value)

            del model, train_tensor, val_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Обчислення середніх значень і відхилень
        summary = {}
        for metric_name, values in fold_results.items():
            if values:
                summary[f'{metric_name}_mean'] = np.mean(values)
                summary[f'{metric_name}_std'] = np.std(values)
                summary[f'{metric_name}_values'] = values

        self.logger.info(
            f"Батчева крос-валідація завершена. Середній RMSE: {summary.get('RMSE_mean', 'N/A'):.6f} ± {summary.get('RMSE_std', 'N/A'):.6f}")

        return summary

    def get_training_summary(self) -> Dict[str, Any]:
        """
            Формує зведену статистику по навчених моделях.

            Returns:
                Dict[str, Any]: Словник зі статистикою, включно з кількістю моделей, розподілом по символах, таймфреймах,
                типах моделей та описовою статистикою по основних метриках (RMSE, MAE, R2).
            """
        total_models = len(self.models)

        if total_models == 0:
            return {'total_models': 0, 'message': 'Немає навчених моделей'}

        # Статистика по символам
        symbol_stats = {}
        timeframe_stats = {}
        model_type_stats = {}

        for model_key in self.models.keys():
            parts = model_key.split('_')
            if len(parts) >= 3:
                symbol = parts[0]
                timeframe = parts[1]
                model_type = parts[2]

                symbol_stats[symbol] = symbol_stats.get(symbol, 0) + 1
                timeframe_stats[timeframe] = timeframe_stats.get(timeframe, 0) + 1
                model_type_stats[model_type] = model_type_stats.get(model_type, 0) + 1

        # Статистика по метриках
        rmse_values = []
        mae_values = []
        r2_values = []

        for metrics in self.model_metrics.values():
            if 'RMSE' in metrics:
                rmse_values.append(metrics['RMSE'])
            if 'MAE' in metrics:
                mae_values.append(metrics['MAE'])
            if 'R2' in metrics:
                r2_values.append(metrics['R2'])

        summary = {
            'total_models': total_models,
            'symbol_distribution': symbol_stats,
            'timeframe_distribution': timeframe_stats,
            'model_type_distribution': model_type_stats,
            'metrics_summary': {
                'rmse': {
                    'mean': np.mean(rmse_values) if rmse_values else None,
                    'std': np.std(rmse_values) if rmse_values else None,
                    'min': np.min(rmse_values) if rmse_values else None,
                    'max': np.max(rmse_values) if rmse_values else None
                },
                'mae': {
                    'mean': np.mean(mae_values) if mae_values else None,
                    'std': np.std(mae_values) if mae_values else None,
                    'min': np.min(mae_values) if mae_values else None,
                    'max': np.max(mae_values) if mae_values else None
                },
                'r2': {
                    'mean': np.mean(r2_values) if r2_values else None,
                    'std': np.std(r2_values) if r2_values else None,
                    'min': np.min(r2_values) if r2_values else None,
                    'max': np.max(r2_values) if r2_values else None
                }
            }
        }

        return summary