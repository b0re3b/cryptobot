import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Union, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib


class EnsembleModels:
    """
    Клас для створення та використання ансамблів моделей для підвищення точності прогнозів.
    """

    def __init__(self, log_level=logging.INFO):
        """
        Ініціалізація класу ансамблів моделей.

        Args:
            log_level: Рівень логування
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.models = {}
        self.base_models = {}

    def add_base_model(self, model_key: str, model) -> None:
        """
        Додавання базової моделі до ансамблю.

        Args:
            model_key: Ключ для ідентифікації моделі
            model: Об'єкт моделі (повинен мати методи fit, predict)
        """
        self.base_models[model_key] = model

    def create_voting_ensemble(self, models_dict: Dict, weights: Optional[List[float]] = None) -> VotingRegressor:
        """
        Створення ансамблю на основі голосування.

        Args:
            models_dict: Словник з моделями {ключ: модель}
            weights: Список вагів для кожної моделі (опціонально)

        Returns:
            Об'єкт VotingRegressor
        """
        models_list = [(key, model) for key, model in models_dict.items()]
        return VotingRegressor(estimators=models_list, weights=weights)

    def create_stacking_ensemble(self, base_models_dict: Dict, meta_model=None) -> object:
        """
        Створення ансамблю на основі стекінгу.

        Args:
            base_models_dict: Словник з базовими моделями
            meta_model: Метамодель для стекінгу (за замовчуванням GradientBoostingRegressor)

        Returns:
            Об'єкт моделі стекінгу
        """
        from sklearn.ensemble import StackingRegressor

        if meta_model is None:
            meta_model = GradientBoostingRegressor(n_estimators=100)

        models_list = [(key, model) for key, model in base_models_dict.items()]

        return StackingRegressor(
            estimators=models_list,
            final_estimator=meta_model
        )

    def fit_ensemble(self, ensemble_key: str, model, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Навчання ансамблю моделей.

        Args:
            ensemble_key: Ключ для збереження навченої моделі
            model: Об'єкт ансамблевої моделі
            X_train, y_train: Тренувальні дані
        """
        try:
            model.fit(X_train, y_train)
            self.models[ensemble_key] = model
            self.logger.info(f"Ансамбль {ensemble_key} успішно навчено")
        except Exception as e:
            self.logger.error(f"Помилка при навчанні ансамблю {ensemble_key}: {e}")

    def create_and_fit_weighted_ensemble(self, ensemble_key: str,
                                         base_models_dict: Dict,
                                         X_train: pd.DataFrame,
                                         y_train: pd.Series,
                                         X_val: pd.DataFrame = None,
                                         y_val: pd.Series = None) -> Dict:
        """
        Створення та навчання зваженого ансамблю з оптимізацією вагів.

        Args:
            ensemble_key: Ключ для ансамблю
            base_models_dict: Словник базових моделей
            X_train, y_train: Тренувальні дані
            X_val, y_val: Валідаційні дані для оптимізації вагів

        Returns:
            Словник з результатами
        """
        # Якщо валідаційні дані не вказані, використовуємо частину тренувальних
        if X_val is None or y_val is None:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )

        # Отримання прогнозів від кожної базової моделі на валідаційних даних
        predictions = {}
        for key, model in base_models_dict.items():
            model.fit(X_train, y_train)
            predictions[key] = model.predict(X_val)

        # Оптимізація вагів за допомогою мінімізації MSE
        from scipy.optimize import minimize

        def objective(weights):
            # Нормалізація вагів для суми 1
            weights = np.array(weights)
            weights = weights / np.sum(weights)

            # Зважене поєднання прогнозів
            weighted_predictions = np.zeros_like(y_val)
            for i, key in enumerate(predictions.keys()):
                weighted_predictions += weights[i] * predictions[key]

            # Розрахунок MSE
            return mean_squared_error(y_val, weighted_predictions)

        # Початкові значення рівних вагів
        initial_weights = [1.0 / len(base_models_dict)] * len(base_models_dict)

        # Обмеження: всі ваги невід'ємні
        constraints = {'type': 'ineq', 'fun': lambda w: w}

        # Оптимізація
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            constraints=constraints
        )

        # Нормалізація оптимальних вагів
        optimal_weights = result['x'] / np.sum(result['x'])

        # Створення та навчання зваженого ансамблю
        ensemble = self.create_voting_ensemble(base_models_dict, weights=optimal_weights)
        self.fit_ensemble(ensemble_key, ensemble, X_train, y_train)

        return {
            'model': ensemble,
            'weights': dict(zip(base_models_dict.keys(), optimal_weights)),
            'validation_mse': result['fun']
        }

    def predict(self, ensemble_key: str, X: pd.DataFrame) -> np.ndarray:
        """
        Прогнозування з використанням ансамблю.

        Args:
            ensemble_key: Ключ ансамблю
            X: Дані для прогнозування

        Returns:
            Прогнозні значення
        """
        if ensemble_key not in self.models:
            self.logger.error(f"Ансамбль {ensemble_key} не знайдений")
            return None

        try:
            return self.models[ensemble_key].predict(X)
        except Exception as e:
            self.logger.error(f"Помилка прогнозування з ансамблем {ensemble_key}: {e}")
            return None

    def evaluate(self, ensemble_key: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Оцінка точності ансамблю.

        Args:
            ensemble_key: Ключ ансамблю
            X_test, y_test: Тестові дані

        Returns:
            Метрики точності
        """
        predictions = self.predict(ensemble_key, X_test)
        if predictions is None:
            return None

        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }

    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:
        """
        Оцінка точності всіх ансамблів і базових моделей.

        Args:
            X_test, y_test: Тестові дані

        Returns:
            Словник з метриками для всіх моделей
        """
        results = {}

        # Оцінка базових моделей
        for key, model in self.base_models.items():
            try:
                predictions = model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, predictions)

                results[f"base_{key}"] = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae
                }
            except Exception as e:
                self.logger.error(f"Помилка при оцінці базової моделі {key}: {e}")

        # Оцінка ансамблевих моделей
        for key in self.models.keys():
            result = self.evaluate(key, X_test, y_test)
            if result:
                results[key] = result

        return results

    def save_model(self, ensemble_key: str, filepath: str) -> bool:
        """
        Збереження ансамблю моделей на диск.

        Args:
            ensemble_key: Ключ ансамблю
            filepath: Шлях для збереження

        Returns:
            Успішність операції
        """
        if ensemble_key not in self.models:
            self.logger.error(f"Ансамбль {ensemble_key} не знайдений")
            return False

        try:
            joblib.dump(self.models[ensemble_key], filepath)
            self.logger.info(f"Ансамбль {ensemble_key} збережено у {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Помилка при збереженні ансамблю {ensemble_key}: {e}")
            return False

    def load_model(self, ensemble_key: str, filepath: str) -> bool:
        """
        Завантаження збереженого ансамблю моделей.

        Args:
            ensemble_key: Ключ для збереження ансамблю
            filepath: Шлях до файлу моделі

        Returns:
            Успішність операції
        """
        try:
            self.models[ensemble_key] = joblib.load(filepath)
            self.logger.info(f"Ансамбль {ensemble_key} завантажено з {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Помилка при завантаженні ансамблю з {filepath}: {e}")
            return False

    def get_feature_importance(self, ensemble_key: str, features: List[str] = None) -> pd.DataFrame:
        """
        Отримання важливості ознак для ансамблю.

        Args:
            ensemble_key: Ключ ансамблю
            features: Список назв ознак

        Returns:
            DataFrame з важливістю ознак
        """
        if ensemble_key not in self.models:
            self.logger.error(f"Ансамбль {ensemble_key} не знайдений")
            return None

        model = self.models[ensemble_key]

        try:
            # Для VotingRegressor отримуємо важливість ознак з базових моделей
            if hasattr(model, 'estimators_'):
                importances = np.zeros(len(features) if features else 1)

                for estimator in model.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        importances += estimator.feature_importances_

                importances /= len(model.estimators_)

                if features:
                    return pd.DataFrame({'feature': features, 'importance': importances}).sort_values(
                        'importance', ascending=False)
                else:
                    return pd.DataFrame({'importance': importances})

            # Для StackingRegressor отримуємо важливість ознак з метамоделі
            elif hasattr(model, 'final_estimator_') and hasattr(model.final_estimator_, 'feature_importances_'):
                importances = model.final_estimator_.feature_importances_

                if features:
                    return pd.DataFrame({'feature': features, 'importance': importances}).sort_values(
                        'importance', ascending=False)
                else:
                    return pd.DataFrame({'importance': importances})

            else:
                self.logger.warning(f"Неможливо отримати важливість ознак для ансамблю {ensemble_key}")
                return None

        except Exception as e:
            self.logger.error(f"Помилка при отриманні важливості ознак для ансамблю {ensemble_key}: {e}")
            return None

    def create_bagging_ensemble(self, base_estimator, n_estimators: int = 10,
                                max_samples: float = 0.8, max_features: float = 0.8,
                                bootstrap: bool = True, random_state: int = 42) -> object:
        """
        Створення ансамблю на основі бегінгу.

        Args:
            base_estimator: Базовий алгоритм
            n_estimators: Кількість естіматорів в ансамблі
            max_samples: Максимальна частка зразків для кожного естіматора
            max_features: Максимальна частка ознак для кожного естіматора
            bootstrap: Чи використовувати бутстреп
            random_state: Ініціалізація генератора випадкових чисел

        Returns:
            Об'єкт BaggingRegressor
        """
        from sklearn.ensemble import BaggingRegressor

        return BaggingRegressor(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state
        )

    def create_boosting_ensemble(self, base_estimator=None, n_estimators: int = 100,
                                 learning_rate: float = 0.1, loss: str = 'squared_error',
                                 random_state: int = 42) -> object:
        """
        Створення ансамблю на основі бустінгу.

        Args:
            base_estimator: Базовий алгоритм (за замовчуванням - дерево рішень)
            n_estimators: Кількість естіматорів в ансамблі
            learning_rate: Швидкість навчання
            loss: Функція втрат
            random_state: Ініціалізація генератора випадкових чисел

        Returns:
            Об'єкт AdaBoostRegressor або GradientBoostingRegressor
        """
        if base_estimator is None:
            # Використовуємо GradientBoostingRegressor за замовчуванням
            return GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                loss=loss,
                random_state=random_state
            )
        else:
            # Використовуємо AdaBoostRegressor з вказаним базовим естіматором
            from sklearn.ensemble import AdaBoostRegressor
            return AdaBoostRegressor(
                base_estimator=base_estimator,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state
            )

    def tune_ensemble_hyperparameters(self, ensemble_type: str, X_train: pd.DataFrame,
                                      y_train: pd.Series, param_grid: Dict = None) -> Dict:
        """
        Оптимізація гіперпараметрів ансамблевої моделі.

        Args:
            ensemble_type: Тип ансамблю ('voting', 'stacking', 'bagging', 'boosting')
            X_train, y_train: Тренувальні дані
            param_grid: Сітка гіперпараметрів для пошуку

        Returns:
            Словник з найкращими параметрами та моделлю
        """
        from sklearn.model_selection import GridSearchCV

        # Визначення базової моделі в залежності від типу ансамблю
        if ensemble_type == 'voting':
            base_model = VotingRegressor(estimators=[('rf', RandomForestRegressor()),
                                                     ('gb', GradientBoostingRegressor())])
            if param_grid is None:
                param_grid = {
                    'weights': [[1, 1], [1, 2], [2, 1], [1, 3], [3, 1]]
                }

        elif ensemble_type == 'stacking':
            base_model = self.create_stacking_ensemble(
                {'rf': RandomForestRegressor(), 'gb': GradientBoostingRegressor()}
            )
            if param_grid is None:
                param_grid = {
                    'final_estimator__n_estimators': [50, 100, 150],
                    'final_estimator__learning_rate': [0.05, 0.1, 0.2]
                }

        elif ensemble_type == 'bagging':
            base_model = self.create_bagging_ensemble(RandomForestRegressor())
            if param_grid is None:
                param_grid = {
                    'n_estimators': [10, 20, 30],
                    'max_samples': [0.7, 0.8, 0.9],
                    'max_features': [0.7, 0.8, 0.9]
                }

        elif ensemble_type == 'boosting':
            base_model = self.create_boosting_ensemble()
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 4, 5]
                }

        else:
            self.logger.error(f"Невідомий тип ансамблю: {ensemble_type}")
            return None

        # Пошук оптимальних параметрів
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        try:
            grid_search.fit(X_train, y_train)

            return {
                'best_params': grid_search.best_params_,
                'best_model': grid_search.best_estimator_,
                'best_score': -grid_search.best_score_  # перетворюємо назад у MSE
            }

        except Exception as e:
            self.logger.error(f"Помилка при оптимізації гіперпараметрів: {e}")
            return None