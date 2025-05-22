from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from ML.base import BaseDeepModel
from data.db import save_ml_model, save_ml_model_metrics, load_ml_model
from feature_engineering import prepare_features_pipeline

class ModelTrainer:
    """
    Клас для навчання, оцінки та онлайн-оновлення моделей глибокого навчання
    на основі часових рядів.
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Ініціалізація класу.

        Args:
            device (torch.device, optional): Пристрій (CPU або GPU) для виконання.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, BaseDeepModel] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {}
        self.model_metrics: Dict[str, Dict[str, float]] = {}
        self.best_models: Dict[str, BaseDeepModel] = {}
        self.training_history: Dict[str, List[float]] = {}

    def train_model(self, symbol: str, timeframe: str, model_type: str,
                    data: pd.DataFrame, input_dim: int,
                    epochs: int = 100, batch_size: int = 32,
                    learning_rate: float = 0.001, hidden_dim: int = 64,
                    num_layers: int = 2, validation_split: float = 0.2,
                    patience: int = 10) -> Dict[str, Any]:
        """
        Повний цикл навчання моделі: підготовка, тренування, збереження та оцінка.

        Args:
            symbol (str): Символ криптовалюти.
            timeframe (str): Таймфрейм (наприклад, '1h', '1d').
            model_type (str): Тип моделі ('lstm' або 'gru').
            data (pd.DataFrame): Підготовлені фічі та ціль.
            input_dim (int): Кількість вхідних ознак.
            epochs (int): Кількість епох.
            batch_size (int): Розмір батчу.
            learning_rate (float): Швидкість навчання.
            hidden_dim (int): Розмір прихованого шару.
            num_layers (int): Кількість шарів.
            validation_split (float): Частка валідаційної вибірки.
            patience (int): Патієнс для early stopping.

        Returns:
            dict: Конфігурація, метрики та історія втрат.
        """
        X, y = data.drop(columns=["target"]).values, data["target"].values
        split = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        train_tensor = (torch.tensor(X_train).float(), torch.tensor(y_train).float())
        val_tensor = (torch.tensor(X_val).float(), torch.tensor(y_val).float())

        model = self._build_model(model_type, input_dim, hidden_dim, num_layers).to(self.device)
        history = self._train_loop(model, train_tensor, val_tensor,
                                   epochs, batch_size, learning_rate, patience)

        model_key = self._create_model_key(symbol, timeframe, model_type)
        self.models[model_key] = model
        self.model_configs[model_key] = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'output_dim': 1,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }

        metrics = self.evaluate(model, val_tensor)
        self.model_metrics[model_key] = metrics

        save_ml_model(symbol, timeframe, model_type, self.model_configs[model_key])
        save_ml_model_metrics(symbol, timeframe, model_type, metrics)

        return {'config': self.model_configs[model_key], 'metrics': metrics, 'history': history}

    def _train_loop(self, model: BaseDeepModel, train_data: Tuple[torch.Tensor, torch.Tensor],
                    val_data: Tuple[torch.Tensor, torch.Tensor], epochs: int,
                    batch_size: int, learning_rate: float, patience: int) -> List[float]:
        """
        Навчальний цикл з підтримкою early stopping.

        Returns:
            list: Список валідаційних втрат по епохах.
        """
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*train_data),
                                                   batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*val_data),
                                                 batch_size=batch_size)

        optimizer, criterion = self.setup_optimizer_and_loss(model, learning_rate)
        val_losses = []
        best_loss = float('inf')
        best_model_state = None

        for epoch in range(epochs):
            model.train()
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)

            model.eval()
            val_loss = self.validate_epoch(model, val_loader, criterion)
            val_losses.append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = model.state_dict()

            if self.early_stopping_check(val_losses, patience):
                break

        if best_model_state:
            model.load_state_dict(best_model_state)

        return val_losses

    def train_epoch(self, model: BaseDeepModel, train_loader,
                    optimizer, criterion) -> float:
        """
        Одна епоха тренування.

        Returns:
            float: Середнє значення втрати.
        """
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            optimizer.zero_grad()
            output = model(X_batch).squeeze()
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate_epoch(self, model: BaseDeepModel, val_loader, criterion) -> float:
        """
        Оцінка моделі на валідації.

        Returns:
            float: Середнє значення втрати.
        """
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                output = model(X_batch).squeeze()
                loss = criterion(output, y_batch)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def evaluate(self, model: BaseDeepModel, test_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        Обчислення метрик моделі.

        Returns:
            dict: MSE, RMSE, MAE, MAPE.
        """
        model.eval()
        X_test, y_true = test_data
        X_test, y_true = X_test.to(self.device), y_true.to(self.device)
        with torch.no_grad():
            y_pred = model(X_test).squeeze().cpu().numpy()
        y_true = y_true.cpu().numpy()
        return self.calculate_metrics(y_true, y_pred)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Обчислення метрик регресії.

        Returns:
            dict: Метрики.
        """
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

    def early_stopping_check(self, val_losses: List[float], patience: int = 10) -> bool:
        """
        Перевірка умови early stopping.

        Returns:
            bool: Чи слід зупинити тренування.
        """
        if len(val_losses) < patience:
            return False
        return val_losses[-patience] <= min(val_losses[-patience:])

    def setup_optimizer_and_loss(self, model: BaseDeepModel, learning_rate: float):
        """
        Налаштування оптимізатора та функції втрат.

        Returns:
            Tuple: Оптимізатор, функція втрат.
        """
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        return optimizer, criterion

    def online_learning(self, symbol: str, timeframe: str, model_type: str,
                        new_data: pd.DataFrame, input_dim: int,
                        epochs: int = 10, learning_rate: float = 0.0005) -> Dict[str, Any]:
        """
        Онлайн-дообучення моделі на нових даних.

        Returns:
            dict: Оновлені метрики.
        """
        model_key = self._create_model_key(symbol, timeframe, model_type)
        if model_key not in self.models:
            if not self.load_model(symbol, timeframe, model_type):
                raise ValueError(f"Модель {model_key} не знайдена")

        model = self.models[model_key]

        X = new_data.drop(columns=["target"]).values
        y = new_data["target"].values
        train_tensor = (torch.tensor(X).float(), torch.tensor(y).float())

        self._train_loop(model, train_tensor, train_tensor, epochs, 32, learning_rate, patience=5)

        metrics = self.evaluate(model, train_tensor)
        self.model_metrics[model_key] = metrics

        save_ml_model(symbol, timeframe, model_type, self.model_configs[model_key])
        save_ml_model_metrics(symbol, timeframe, model_type, metrics)

        return {'metrics': metrics}

    def _build_model(self, model_type: str, input_dim: int, hidden_dim: int, num_layers: int) -> BaseDeepModel:
        """
        Побудова моделі за типом.

        Returns:
            BaseDeepModel: Ініціалізована модель.
        """
        from ML.models import LSTMModel, GRUModel
        if model_type == 'lstm':
            return LSTMModel(input_dim, hidden_dim, num_layers, output_dim=1)
        elif model_type == 'gru':
            return GRUModel(input_dim, hidden_dim, num_layers, output_dim=1)
        else:
            raise ValueError(f"Невідомий тип моделі: {model_type}")

    def _create_model_key(self, symbol: str, timeframe: str, model_type: str) -> str:
        """
        Генерує унікальний ключ для моделі.

        Returns:
            str: Ключ у форматі SYMBOL_TIMEFRAME_MODELTYPE.
        """
        return f"{symbol}_{timeframe}_{model_type}"

    def load_model(self, symbol: str, timeframe: str, model_type: str) -> bool:
        """
        Завантаження збереженої моделі з бази.

        Returns:
            bool: True, якщо модель успішно завантажена.
        """
        model_key = self._create_model_key(symbol, timeframe, model_type)
        model = load_ml_model(symbol, timeframe, model_type)
        if model:
            self.models[model_key] = model.to(self.device)
            return True
        return False
