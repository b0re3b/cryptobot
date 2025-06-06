from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import silhouette_score

from utils.logger import CryptoLogger


class DimensionalityReducer:
    def __init__(self):
        self.logger = CryptoLogger('Dimensionality Reducer')

    def select_features(self, X: pd.DataFrame, y: pd.Series,
                        n_features: Optional[int] = None,
                        method: str = 'f_regression') -> Tuple[pd.DataFrame, List[str]]:
        """
        Відбір найбільш інформативних ознак з використанням різних методів.

        Args:
            X: DataFrame з вхідними ознаками
            y: Series з цільовою змінною
            n_features: Кількість ознак для відбору
            method: Метод відбору ознак ('f_regression', 'mutual_info', 'rfe')

        Returns:
            Кортеж (вибрані_ознаки, список_імен_ознак)
        """
        self.logger.info(f"Вибір ознак методом '{method}'")

        # Перевіряємо, що X і y мають однакову кількість рядків
        if len(X) != len(y):
            self.logger.error(f"Розмірності X ({len(X)}) і y ({len(y)}) не співпадають")
            raise ValueError(f"Розмірності X ({len(X)}) і y ({len(y)}) не співпадають")

        # ФІКС: Безпечна перевірка на NaN
        x_has_nan = X.isna().any().any() if not X.empty else False
        y_has_nan = y.isna().any() if not y.empty else False

        if x_has_nan or y_has_nan:
            self.logger.warning("Виявлено пропущені значення. Видаляємо рядки з NaN")
            # ФІКС: Використовуємо векторизований підхід без неоднозначних булевих операцій
            x_valid_mask = X.notna().all(axis=1)
            y_valid_mask = y.notna()
            valid_indices = x_valid_mask & y_valid_mask

            X = X.loc[valid_indices]
            y = y.loc[valid_indices]
            self.logger.info(f"Залишилось {len(X)} рядків після видалення NaN")

        # Перевіряємо, що залишились дані для аналізу
        if X.empty or y.empty:
            self.logger.error("Після обробки NaN не залишилось даних для аналізу")
            raise ValueError("Після обробки пропущених значень не залишилось даних для аналізу")

        # Визначаємо кількість ознак для вибору, якщо не вказано
        if n_features is None:
            n_features = min(X.shape[1] // 2, X.shape[1])
            self.logger.info(f"Автоматично визначено кількість ознак: {n_features}")

        # Обмежуємо кількість ознак доступним числом
        n_features = min(n_features, X.shape[1])
        self.logger.info(f"Буде відібрано {n_features} ознак з {X.shape[1]}")

        # Словник функцій відбору ознак
        selection_methods = {
            'f_regression': lambda: SelectKBest(score_func=f_regression, k=n_features),
            'mutual_info': lambda: SelectKBest(score_func=mutual_info_regression, k=n_features),
            'rfe': lambda: RFE(estimator=LinearRegression(), n_features_to_select=n_features, step=1)
        }

        # Перевіряємо наявність методу
        if method not in selection_methods:
            self.logger.error(f"Невідомий метод вибору ознак: {method}")
            raise ValueError(
                f"Невідомий метод вибору ознак: {method}. Допустимі значення: {list(selection_methods.keys())}"
            )

        # Вибираємо та застосовуємо метод селекції ознак
        try:
            selector = selection_methods[method]()
            selector.fit(X, y)

            # Отримуємо маску вибраних ознак та назви ознак
            if hasattr(selector, 'get_support'):
                selected_mask = selector.get_support()
            else:
                selected_mask = selector.support_

            selected_features = X.columns[selected_mask].tolist()

            # Виводимо інформацію про вибрані ознаки
            if method in ['f_regression', 'mutual_info'] and hasattr(selector, 'scores_'):
                # Створюємо DataFrame для зручного відображення
                scores_df = pd.DataFrame({
                    'feature': X.columns,
                    'score': selector.scores_
                }).sort_values('score', ascending=False)

                top_features = scores_df.head(5).values.tolist()
                self.logger.info(f"Топ-5 ознак за {method}: {top_features}")
            elif method == 'rfe' and hasattr(selector, 'ranking_'):
                # Також використовуємо DataFrame для рангів
                ranks_df = pd.DataFrame({
                    'feature': X.columns,
                    'rank': selector.ranking_
                }).sort_values('rank')

                top_features = ranks_df.head(5)['feature'].tolist()
                self.logger.info(f"Топ-5 ознак за RFE: {top_features}")

        except Exception as e:
            self.logger.error(f"Помилка при використанні {method}: {str(e)}. Переходимо до F-тесту.")
            # У випадку помилки використовуємо F-тест
            selector = SelectKBest(score_func=f_regression, k=n_features)
            selector.fit(X, y)
            selected_mask = selector.get_support()
            selected_features = X.columns[selected_mask].tolist()

        # Створюємо DataFrame з відібраними ознаками
        X_selected = X[selected_features]
        self.logger.info(f"Відібрано {len(selected_features)} ознак: {selected_features[:5]}...")

        return X_selected, selected_features

    def reduce_dimensions(self, data: pd.DataFrame,
                          n_components: Optional[int] = None,
                          method: str = 'pca') -> Tuple[pd.DataFrame, object]:
        """
        Зменшення розмірності даних за допомогою різних методів.

        Args:
            data: DataFrame з вхідними даними
            n_components: Кількість компонентів для зменшення розмірності
            method: Метод зменшення розмірності ('pca', 'kmeans')

        Returns:
            Кортеж (трансформовані_дані, об'єкт_трансформатора)
        """
        self.logger.info(f"Зменшення розмірності методом '{method}'")

        # Створюємо копію, щоб не модифікувати оригінальні дані
        X = data.copy()

        # ФІКС: Безпечна перевірка на NaN
        if not X.empty and X.isna().any().any():
            self.logger.warning("Виявлено пропущені значення. Заповнюємо їх медіаною.")
            X = X.fillna(X.median())

        # Визначаємо кількість компонентів, якщо не вказано
        if n_components is None:
            n_components = min(int(np.sqrt(X.shape[1])), 10)
            self.logger.info(f"Автоматично визначено кількість компонентів: {n_components}")

        # Обмежуємо кількість компонентів доступним числом ознак
        n_components = min(n_components, X.shape[1], X.shape[0])

        # Словник методів зменшення розмірності
        reduction_methods = {
            'pca': self._apply_pca,
            'kmeans': self._apply_kmeans
        }

        # Перевіряємо наявність методу
        if method not in reduction_methods:
            self.logger.error(f"Невідомий метод зменшення розмірності: {method}")
            raise ValueError(
                f"Невідомий метод зменшення розмірності: {method}. Допустимі значення: {list(reduction_methods.keys())}"
            )

        # Застосовуємо вибраний метод
        result_df, transformer = reduction_methods[method](X, n_components)
        self.logger.info(f"Розмірність зменшено з {X.shape[1]} до {result_df.shape[1]} ознак")

        return result_df, transformer

    def _apply_pca(self, X: pd.DataFrame, n_components: int) -> Tuple[pd.DataFrame, PCA]:
        """Допоміжний метод для застосування PCA"""
        self.logger.info(f"Застосовуємо PCA з {n_components} компонентами")

        # Створюємо і застосовуємо PCA
        transformer = PCA(n_components=n_components)
        X_transformed = transformer.fit_transform(X)

        # Логування пояснення дисперсії
        explained_variance_ratio = transformer.explained_variance_ratio_
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)
        self.logger.info(f"PCA пояснює {cumulative_explained_variance[-1] * 100:.2f}% загальної дисперсії")
        self.logger.info(f"Перші 3 компоненти пояснюють: {explained_variance_ratio[:min(3, n_components)] * 100}")

        # Створюємо назви компонентів
        component_names = [f'pca_component_{i + 1}' for i in range(n_components)]

        # Аналіз важливості ознак для компонент (векторизований підхід)
        feature_importance = transformer.components_
        for i in range(min(3, n_components)):
            # Створюємо DataFrame для аналізу важливості ознак
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': feature_importance[i]
            })
            importance_df['abs_importance'] = importance_df['importance'].abs()
            importance_df = importance_df.sort_values('abs_importance', ascending=False)

            top_features = importance_df.head(5)[['feature', 'importance']].values.tolist()
            self.logger.info(f"Компонента {i + 1} найбільше залежить від: {top_features}")

        # Створюємо DataFrame з трансформованими даними
        result_df = pd.DataFrame(X_transformed, index=X.index, columns=component_names)

        return result_df, transformer

    def _apply_kmeans(self, X: pd.DataFrame, n_clusters: int) -> Tuple[pd.DataFrame, KMeans]:
        """Допоміжний метод для застосування KMeans"""
        self.logger.info(f"Застосовуємо KMeans з {n_clusters} кластерами")

        # Створюємо і застосовуємо KMeans
        transformer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = transformer.fit_predict(X)

        # Більш ефективне створення матриці one-hot encoding
        X_transformed = np.zeros((X.shape[0], n_clusters))
        X_transformed[np.arange(X.shape[0]), cluster_labels] = 1

        # Створюємо назви компонентів (кластерів)
        component_names = [f'cluster_{i + 1}' for i in range(n_clusters)]

        # Аналіз розміру кластерів (векторизований підхід)
        cluster_sizes = np.bincount(cluster_labels, minlength=n_clusters)
        cluster_info = list(zip(range(1, n_clusters + 1), cluster_sizes))
        self.logger.info(f"Розмір кластерів: {cluster_info}")

        # Аналіз центроїдів кластерів
        centroids = transformer.cluster_centers_
        mean_values = X.mean().values

        for i in range(min(3, n_clusters)):
            # Обчислюємо різницю між центроїдом і глобальним середнім
            centroid_diff = centroids[i] - mean_values

            # Створюємо DataFrame для відображення результатів
            centroid_df = pd.DataFrame({
                'feature': X.columns,
                'diff_from_mean': centroid_diff
            })
            centroid_df['abs_diff'] = centroid_df['diff_from_mean'].abs()
            centroid_df = centroid_df.sort_values('abs_diff', ascending=False)

            top_features = centroid_df.head(5)[['feature', 'diff_from_mean']].values.tolist()
            self.logger.info(f"Кластер {i + 1} характеризується: {top_features}")

        # Створюємо DataFrame з трансформованими даними
        result_df = pd.DataFrame(X_transformed, index=X.index, columns=component_names)

        return result_df, transformer

    def create_polynomial_features(self, data: pd.DataFrame,
                                   columns: Optional[List[str]] = None,
                                   degree: int = 2,
                                   interaction_only: bool = False,
                                   max_features: int = 50) -> pd.DataFrame:
        """
        Створення поліноміальних ознак на основі вибраних стовпців з контролем пам'яті.

        Args:
            data: DataFrame з вхідними даними
            columns: Список стовпців для створення поліноміальних ознак
            degree: Степінь поліному
            interaction_only: Якщо True, включає тільки взаємодії без степенів
            max_features: Максимальна кількість вхідних ознак для обробки

        Returns:
            DataFrame з доданими поліноміальними ознаками
        """
        self.logger.info("Створення поліноміальних ознак...")

        # ФІКС: Безпечний вибір числових стовпців
        if data.empty:
            self.logger.warning("Переданий порожній DataFrame")
            return data

        # Вибираємо числові стовпці, якщо columns не вказано
        if columns is None:
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            columns = numeric_columns.tolist() if not numeric_columns.empty else []
            self.logger.info(f"Автоматично вибрано {len(columns)} числових стовпців")
        else:
            # Перевіряємо наявність вказаних стовпців у даних
            missing_cols = list(set(columns) - set(data.columns))
            if missing_cols:
                self.logger.warning(f"Стовпці {missing_cols} не знайдено в даних і будуть пропущені")
                columns = [col for col in columns if col in data.columns]

        # Перевіряємо, чи залишились стовпці після фільтрації
        if not columns:
            self.logger.error("Немає доступних стовпців для створення поліноміальних ознак")
            return data

        if len(columns) > max_features:
            self.logger.warning(f"Занадто багато ознак ({len(columns)}). Обмежуємо до {max_features} найважливіших.")

            # Вибираємо найбільш варіативні ознаки
            X_temp = data[columns].copy()
            # ФІКС: Безпечна перевірка на NaN
            if not X_temp.empty and X_temp.isna().any().any():
                X_temp = X_temp.fillna(X_temp.median())

            # Обчислюємо коефіцієнт варіації для кожної ознаки
            cv_scores = (X_temp.std() / (X_temp.mean().abs() + 1e-8)).fillna(0)
            top_features = cv_scores.nlargest(max_features).index.tolist()
            columns = top_features
            self.logger.info(f"Вибрано {len(columns)} найбільш варіативних ознак")

        from math import comb
        if not interaction_only:
            # Для повних поліноміальних ознак
            estimated_features = sum(comb(len(columns) + degree - k - 1, degree - k) for k in range(degree + 1))
        else:
            # Для взаємодій
            estimated_features = sum(comb(len(columns), k) for k in range(1, degree + 1))

        estimated_memory_gb = (data.shape[0] * estimated_features * 8) / (1024 ** 3)  # 8 bytes per float64

        self.logger.info(f"Очікувана кількість нових ознак: {estimated_features}")
        self.logger.info(f"Очікуване використання пам'яті: {estimated_memory_gb:.2f} GB")

        max_memory_gb = 4.0
        if estimated_memory_gb > max_memory_gb:
            self.logger.warning(
                f"Очікуване використання пам'яті ({estimated_memory_gb:.2f} GB) перевищує ліміт ({max_memory_gb} GB)")

            if degree > 2:
                self.logger.info("Зменшуємо степінь поліному до 2")
                degree = 2
            elif not interaction_only:
                self.logger.info("Переключаємось на режим тільки взаємодій")
                interaction_only = True
            else:
                # Зменшуємо кількість ознак ще більше
                new_max_features = max(5, max_features // 2)
                self.logger.info(f"Зменшуємо кількість ознак до {new_max_features}")

                X_temp = data[columns].copy()
                # ФІКС: Безпечна перевірка на NaN
                if not X_temp.empty and X_temp.isna().any().any():
                    X_temp = X_temp.fillna(X_temp.median())

                cv_scores = (X_temp.std() / (X_temp.mean().abs() + 1e-8)).fillna(0)
                top_features = cv_scores.nlargest(new_max_features).index.tolist()
                columns = top_features

            # Перераховуємо після змін
            if not interaction_only:
                estimated_features = sum(comb(len(columns) + degree - k - 1, degree - k) for k in range(degree + 1))
            else:
                estimated_features = sum(comb(len(columns), k) for k in range(1, degree + 1))

            estimated_memory_gb = (data.shape[0] * estimated_features * 8) / (1024 ** 3)
            self.logger.info(f"Після оптимізації: {estimated_features} ознак, {estimated_memory_gb:.2f} GB")

            # Якщо все ще занадто багато, відмовляємось
            if estimated_memory_gb > max_memory_gb:
                self.logger.error(f"Неможливо створити поліноміальні ознаки без перевищення ліміту пам'яті")
                return data

        # Створюємо копію DataFrame з вибраними стовпцями
        result_df = data.copy()
        X = result_df[columns].copy()

        # ФІКС: Безпечна перевірка на NaN
        if not X.empty and X.isna().any().any():
            self.logger.warning(f"Виявлено NaN значення у вхідних даних. Заповнюємо їх медіаною.")
            X = X.fillna(X.median())

        # Перевірка на нескінченні значення
        if not X.empty and np.isinf(X.values).any():
            self.logger.warning("Виявлено нескінченні значення у вхідних даних. Замінюємо їх великими числами.")
            X = X.replace([np.inf, -np.inf], [1e10, -1e10])

        # Стандартизація даних
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # Обрізаємо значення до розумного діапазону
        X_scaled = X_scaled.clip(-5, 5)  # Більш консервативне обрізання
        self.logger.info(f"Дані стандартизовано та обрізано до діапазону [-5, 5]")

        # Створюємо об'єкт для поліноміальних ознак
        poly = PolynomialFeatures(degree=degree,
                                  interaction_only=interaction_only,
                                  include_bias=False)

        # Застосовуємо трансформацію з обробкою помилок
        try:
            poly_features = poly.fit_transform(X_scaled)

            # Перевірка на переповнення
            if np.isinf(poly_features).any() or np.isnan(poly_features).any():
                self.logger.warning("Виявлено нескінченні або NaN значення після створення поліноміальних ознак")
                poly_features = np.where(np.isinf(poly_features), np.nan, poly_features)

                if np.isnan(poly_features).all():
                    self.logger.error("Всі поліноміальні ознаки стали NaN. Повертаємо оригінальні дані.")
                    return data

            # Отримуємо назви нових ознак
            feature_names = poly.get_feature_names_out(X_scaled.columns)

            # Створюємо DataFrame з новими ознаками
            poly_df = pd.DataFrame(poly_features,
                                   columns=feature_names,
                                   index=X_scaled.index)

            # Видаляємо оригінальні ознаки, якщо degree > 1
            if degree > 1:
                n_original_features = len(X_scaled.columns)
                poly_df = poly_df.iloc[:, n_original_features:]

            # Додаємо префікс до назв ознак
            poly_df.columns = [f'poly_{col}' for col in poly_df.columns]

            # Замінюємо нескінченні значення на NaN
            poly_df = poly_df.replace([np.inf, -np.inf], np.nan)

            # ФІКС: Безпечна обробка NaN значень по стовпцях
            if not poly_df.empty and poly_df.isna().any().any():
                # Заповнюємо NaN медіаною або нулем
                for col in poly_df.columns:
                    col_series = poly_df[col]

                    # ФІКС: Використовуємо .all() замість ambiguous boolean evaluation
                    if col_series.isna().all():
                        poly_df[col] = 0
                    else:
                        # Безпечне обчислення медіани
                        try:
                            median_val = col_series.median()
                            # Перевіряємо, чи є медіана скаляром та чи не є NaN
                            if pd.isna(median_val):
                                poly_df[col] = col_series.fillna(0)
                            else:
                                # Використовуємо медіану для заповнення NaN
                                poly_df[col] = col_series.fillna(median_val)
                        except Exception as e:
                            self.logger.warning(f"Помилка при обчисленні медіани для колонки {col}: {str(e)}")
                            poly_df[col] = col_series.fillna(0)

            # Використовуємо IQR для виявлення викидів
            for col in poly_df.columns:
                try:
                    col_series = poly_df[col]
                    Q1 = col_series.quantile(0.25)
                    Q3 = col_series.quantile(0.75)
                    IQR = Q3 - Q1

                    # Перевіряємо, чи IQR є скаляром та чи більше нуля
                    if pd.notna(IQR) and IQR > 0:
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        poly_df[col] = col_series.clip(lower_bound, upper_bound)
                except Exception as e:
                    self.logger.warning(f"Помилка при обробці викидів для колонки {col}: {str(e)}")
                    continue

            # ФІКС: Правильне видалення константних стовпців
            constant_cols = []
            for col in poly_df.columns:
                try:
                    col_series = poly_df[col]
                    # ФІКС: Використовуємо правільний спосіб перевірки константності
                    if col_series.dropna().empty:  # Вся колонка NaN
                        constant_cols.append(col)
                    elif col_series.nunique(dropna=True) <= 1:  # Константна колонка
                        constant_cols.append(col)
                    else:
                        # ФІКС: Безпечна перевірка стандартного відхилення
                        try:
                            std_val = col_series.std()
                            if pd.isna(std_val) or std_val == 0:
                                constant_cols.append(col)
                        except Exception:
                            constant_cols.append(col)
                except Exception as e:
                    self.logger.warning(f"Помилка при перевірці константності колонки {col}: {str(e)}")
                    constant_cols.append(col)

            if constant_cols:
                self.logger.info(f"Видалено {len(constant_cols)} константних стовпців")
                poly_df = poly_df.drop(columns=constant_cols)

            # Обмежуємо кількість фінальних ознак
            max_final_features = 100  # Максимум 100 нових ознак
            if len(poly_df.columns) > max_final_features:
                self.logger.info(f"Обмежуємо кількість поліноміальних ознак до {max_final_features}")

                # Вибираємо найбільш варіативні ознаки
                try:
                    cv_scores = (poly_df.std() / (poly_df.mean().abs() + 1e-8)).fillna(0)
                    top_features = cv_scores.nlargest(max_final_features).index.tolist()
                    poly_df = poly_df[top_features]
                except Exception as e:
                    self.logger.warning(f"Помилка при відборі топ ознак: {str(e)}")
                    # Просто беремо перші N колонок
                    poly_df = poly_df.iloc[:, :max_final_features]

            # Об'єднуємо з вихідним DataFrame
            if len(poly_df.columns) > 0:
                result_df = pd.concat([result_df, poly_df], axis=1)
                self.logger.info(f"Додано {len(poly_df.columns)} поліноміальних ознак степені {degree}")
            else:
                self.logger.warning("Не створено жодної валідної поліноміальної ознаки")
                return data

        except MemoryError as e:
            self.logger.error(f"Помилка пам'яті при створенні поліноміальних ознак: {str(e)}")
            return data
        except Exception as e:
            self.logger.error(f"Помилка при створенні поліноміальних ознак: {str(e)}")

            # Спроба з меншими параметрами
            if degree > 2:
                self.logger.info(f"Спроба створити поліноміальні ознаки зі степенем {degree - 1}")
                return self.create_polynomial_features(data, columns, degree - 1, interaction_only, max_features)
            elif not interaction_only:
                self.logger.info("Спроба створити тільки взаємодії")
                return self.create_polynomial_features(data, columns, degree, True, max_features)
            else:
                self.logger.error("Не вдалося створити поліноміальні ознаки")
                return data

        return result_df

    def create_cluster_features(self, data: pd.DataFrame,
                                n_clusters: int = 5,
                                method: str = 'kmeans') -> pd.DataFrame:
        """
        Створення ознак на основі кластеризації даних з покращеною обробкою численних помилок.

        Args:
            data: DataFrame з вхідними даними
            n_clusters: Кількість кластерів
            method: Метод кластеризації ('kmeans', 'dbscan')

        Returns:
            DataFrame з доданими ознаками кластеризації
        """
        self.logger.info(f"Створення ознак на основі кластеризації методом '{method}'...")

        # Створюємо копію DataFrame
        result_df = data.copy()

        # Вибираємо числові стовпці для кластеризації
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            self.logger.error("Немає числових стовпців для кластеризації")
            return result_df

        # Підготовка даних для кластеризації
        X = result_df[numeric_cols].copy()

        # Замінюємо NaN та нескінченні значення
        if X.isna().any().any():
            self.logger.warning("Виявлено NaN значення у вхідних даних. Заповнюємо їх медіаною.")
            X = X.fillna(X.median())

        # Замінюємо нескінченні значення
        if np.isinf(X.values).any():
            self.logger.warning("Виявлено нескінченні значення у вхідних даних. Замінюємо їх граничними значеннями.")
            # Використовуємо більш безпечний підхід для заміни нескінченних значень
            for col in X.columns:
                col_data = X[col]
                finite_mask = np.isfinite(col_data)
                finite_data = col_data[finite_mask]
                if len(finite_data) > 0:
                    max_val = finite_data.max()
                    min_val = finite_data.min()
                    X.loc[col_data == np.inf, col] = max_val * 10
                    X.loc[col_data == -np.inf, col] = min_val * 10
                else:
                    X.loc[col_data == np.inf, col] = 1e6
                    X.loc[col_data == -np.inf, col] = -1e6

            # Додаткова перевірка та обрізання
            if np.isinf(X.values).any():
                X = X.clip(-1e10, 1e10)

        # Видаляємо константні стовпці (нульова варіативність)
        constant_cols = []
        for col in X.columns:
            if X[col].nunique() <= 1 or X[col].std() == 0:
                constant_cols.append(col)

        if constant_cols:
            self.logger.info(f"Видалено {len(constant_cols)} константних стовпців: {constant_cols}")
            X = X.drop(columns=constant_cols)

        if X.empty:
            self.logger.error("Після видалення константних стовпців не залишилось даних для кластеризації")
            return result_df

        # Стандартизуємо дані з додатковими перевірками
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Перевіряємо результат стандартизації
            if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
                self.logger.warning(
                    "Стандартизація призвела до NaN або нескінченних значень. Використовуємо робастну стандартизацію.")

                # Робастна стандартизація
                from sklearn.preprocessing import RobustScaler
                robust_scaler = RobustScaler()
                X_scaled = robust_scaler.fit_transform(X)

                # Якщо все ще є проблеми, обрізаємо значення
                if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
                    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=5.0, neginf=-5.0)

            # Обрізаємо екстремальні значення для стабільності
            X_scaled = np.clip(X_scaled, -10, 10)

        except Exception as e:
            self.logger.error(f"Помилка при стандартизації даних: {str(e)}")
            return result_df

        # Словник методів кластеризації
        clustering_methods = {
            'kmeans': self._apply_kmeans_clustering_safe,
            'dbscan': self._apply_dbscan_clustering_safe,
        }

        # Перевіряємо наявність методу
        method = method.lower()
        if method not in clustering_methods:
            self.logger.error(
                f"Невідомий метод кластеризації: {method}. Підтримуються: {list(clustering_methods.keys())}"
            )
            return result_df

        # Застосовуємо вибраний метод кластеризації
        try:
            return clustering_methods[method](result_df, X, X_scaled, n_clusters)
        except Exception as e:
            self.logger.error(f"Помилка при кластеризації: {str(e)}")
            return result_df

    def _apply_dbscan_clustering_safe(self, result_df: pd.DataFrame, X: pd.DataFrame,
                                      X_scaled: np.ndarray, n_clusters: int) -> pd.DataFrame:
        """Безпечний допоміжний метод для кластеризації DBSCAN"""
        try:
            # Визначаємо параметри DBSCAN більш консервативно
            try:
                nbrs = NearestNeighbors(n_neighbors=min(len(X), 5)).fit(X_scaled)
                distances, _ = nbrs.kneighbors(X_scaled)

                # Безпечне обчислення eps
                knee_distances = distances[:, -1]
                knee_distances = knee_distances[np.isfinite(knee_distances)]

                if len(knee_distances) > 0:
                    eps = float(np.median(knee_distances))  # Явне перетворення в float
                    eps = np.clip(eps, 0.1, 10.0)  # Обмежуємо eps розумними межами
                else:
                    eps = 0.5  # Значення за замовчуванням

            except Exception as e:
                self.logger.warning(f"Помилка при визначенні eps: {str(e)}. Використовуємо значення за замовчуванням.")
                eps = 0.5

            min_samples = max(3, min(len(X) // 50, 10))  # Більш консервативний min_samples

            self.logger.debug(f"DBSCAN параметри: eps={eps}, min_samples={min_samples}")

            # Застосовуємо DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            cluster_labels = dbscan.fit_predict(X_scaled)

            # Додаємо мітки кластерів як нову ознаку
            result_df['dbscan_cluster'] = cluster_labels

            # Аналізуємо результати кластеризації
            unique_labels = np.unique(cluster_labels)
            n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_outliers = int(np.sum(cluster_labels == -1))  # Явне перетворення в int

            self.logger.info(f"DBSCAN знайдено {n_clusters_found} кластерів")
            self.logger.info(f"Кількість точок-викидів: {n_outliers}")

            # Обробляємо викиди, якщо вони є
            if -1 in cluster_labels and n_clusters_found > 0:
                try:
                    outliers_mask = cluster_labels == -1
                    non_outliers_mask = ~outliers_mask

                    # ФІКС: Використовуємо .any() для перевірки наявності True значень
                    if non_outliers_mask.any() and outliers_mask.any():
                        # Знаходимо найближчий кластер для викидів
                        n_neighbors = min(3, int(non_outliers_mask.sum()))
                        if n_neighbors > 0:
                            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
                            knn.fit(X_scaled[non_outliers_mask], cluster_labels[non_outliers_mask])

                            closest_clusters = knn.predict(X_scaled[outliers_mask])

                            # Створюємо нову колонку з виправленими мітками
                            fixed_labels = cluster_labels.copy()
                            fixed_labels[outliers_mask] = closest_clusters
                            result_df['dbscan_nearest_cluster'] = fixed_labels

                    # Додаємо бінарну ознаку "чи є точка викидом"
                    result_df['dbscan_is_outlier'] = outliers_mask.astype(int)

                except Exception as e:
                    self.logger.warning(f"Помилка при обробці викидів: {str(e)}")

            # Обчислюємо відстані до центроїдів кластерів (якщо є кластери)
            if n_clusters_found > 0:
                try:
                    for i in unique_labels:
                        if i != -1:  # Пропускаємо викиди
                            cluster_mask = cluster_labels == i
                            # ФІКС: Використовуємо .any() для перевірки наявності True значень
                            if cluster_mask.any():
                                # Безпечне обчислення центроїда
                                cluster_points = X_scaled[cluster_mask]
                                centroid = np.mean(cluster_points, axis=0)

                                # Безпечне обчислення відстаней
                                distances = np.zeros(X_scaled.shape[0])
                                for j in range(X_scaled.shape[0]):
                                    try:
                                        diff = X_scaled[j] - centroid
                                        dist_squared = np.sum(diff ** 2)

                                        if np.isfinite(dist_squared) and dist_squared >= 0:
                                            distances[j] = float(np.sqrt(dist_squared))
                                        else:
                                            distances[j] = float(np.sum(np.abs(diff)))

                                    except (OverflowError, FloatingPointError):
                                        distances[j] = 1e6

                                # Обрізаємо та очищуємо відстані
                                distances = np.nan_to_num(distances, nan=0.0, posinf=1e6, neginf=0.0)
                                distances = np.clip(distances, 0, 1e6)

                                result_df[f'distance_to_dbscan_cluster_{i}'] = distances

                except Exception as e:
                    self.logger.warning(f"Помилка при обчисленні відстаней до центроїдів DBSCAN: {str(e)}")

            self.logger.info(f"Успішно створено ознаки кластеризації DBSCAN")

        except Exception as e:
            self.logger.error(f"Критична помилка при кластеризації DBSCAN: {str(e)}")

        return result_df

    def _apply_kmeans_clustering_safe(self, result_df: pd.DataFrame, X: pd.DataFrame,
                                      X_scaled: np.ndarray, n_clusters: int) -> pd.DataFrame:
        """Безпечний допоміжний метод для кластеризації KMeans з обробкою численних помилок"""
        try:
            # Обмежуємо кількість кластерів розміром даних
            max_clusters = min(n_clusters, len(X) // 5, 20)  # Максимум 20 кластерів
            n_clusters = max(2, max_clusters)

            # Визначаємо оптимальну кількість кластерів, якщо потрібно
            if n_clusters > 10:
                scores = []
                range_clusters = range(2, min(11, len(X) // 10))

                for i in range_clusters:
                    try:
                        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10, max_iter=100)
                        cluster_labels = kmeans.fit_predict(X_scaled)

                        # Перевіряємо кількість унікальних міток
                        if len(np.unique(cluster_labels)) < i:
                            continue

                        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                        if not np.isnan(silhouette_avg) and not np.isinf(silhouette_avg):
                            scores.append(silhouette_avg)
                        else:
                            scores.append(-1)  # Поганий score для недійсних значень

                    except Exception as e:
                        self.logger.debug(f"Помилка для n_clusters = {i}: {str(e)}")
                        scores.append(-1)

                if scores and max(scores) > -1:
                    best_n_clusters = range_clusters[np.argmax(scores)]
                    self.logger.info(f"Оптимальна кількість кластерів за silhouette score: {best_n_clusters}")
                    n_clusters = best_n_clusters

            # Застосовуємо KMeans з визначеною кількістю кластерів
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
            cluster_labels = kmeans.fit_predict(X_scaled)
            centers = kmeans.cluster_centers_

            # Додаємо мітки кластерів як нову ознаку
            result_df['cluster_label'] = cluster_labels

            # Безпечне обчислення відстаней до центроїдів
            for i in range(n_clusters):
                try:
                    # Обчислюємо відстань з обробкою переповнення
                    distances = np.zeros(X_scaled.shape[0])

                    for j in range(X_scaled.shape[0]):
                        try:
                            # Обчислюємо евклідову відстань
                            diff = X_scaled[j] - centers[i]
                            dist_squared = np.sum(diff ** 2)

                            # Перевіряємо на переповнення
                            if np.isfinite(dist_squared) and dist_squared >= 0:
                                distances[j] = np.sqrt(dist_squared)
                            else:
                                # Використовуємо манхеттенську відстань як резерв
                                distances[j] = np.sum(np.abs(diff))

                        except (OverflowError, FloatingPointError):
                            # У випадку переповнення використовуємо максимальну відстань
                            distances[j] = np.finfo(np.float64).max / 1e6

                    # Обрізаємо екстремальні значення та заповнюємо NaN
                    distances = np.nan_to_num(distances, nan=0.0, posinf=1e6, neginf=0.0)
                    distances = np.clip(distances, 0, 1e6)

                    result_df[f'distance_to_cluster_{i}'] = distances

                except Exception as e:
                    self.logger.warning(f"Помилка при обчисленні відстані до кластера {i}: {str(e)}")
                    # Додаємо нульові відстані у випадку помилки
                    result_df[f'distance_to_cluster_{i}'] = 0.0

            # Додаємо додаткові корисні ознаки
            try:
                # Відстань до найближчого центроїда
                distance_columns = [col for col in result_df.columns if col.startswith('distance_to_cluster_')]
                if distance_columns:
                    result_df['min_cluster_distance'] = result_df[distance_columns].min(axis=1)
                    result_df['max_cluster_distance'] = result_df[distance_columns].max(axis=1)

                    # Коефіцієнт силуетності для кожної точки (якщо можливо)
                    try:
                        # ФІКС: silhouette_score повертає одне значення, не масив
                        silhouette_avg = silhouette_score(X_scaled, cluster_labels, metric='euclidean')
                        if np.isfinite(silhouette_avg):
                            # Якщо потрібні індивідуальні значення силуетності, використовуємо silhouette_samples
                            from sklearn.metrics import silhouette_samples
                            silhouette_vals = silhouette_samples(X_scaled, cluster_labels, metric='euclidean')
                            # Перевіряємо на валідність
                            silhouette_vals = np.nan_to_num(silhouette_vals, nan=0.0)
                            result_df['silhouette_score'] = silhouette_vals
                    except Exception as silhouette_error:
                        self.logger.debug(f"Не вдалося обчислити силуетність: {str(silhouette_error)}")

            except Exception as e:
                self.logger.warning(f"Помилка при створенні додаткових ознак: {str(e)}")

            self.logger.info(f"Успішно створено ознаки кластеризації KMeans з {n_clusters} кластерами")

        except Exception as e:
            self.logger.error(f"Критична помилка при кластеризації KMeans: {str(e)}")

        return result_df