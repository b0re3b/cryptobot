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


class DimensionalityReducer():
    def __init__(self):
        self.logger = CryptoLogger('INFO')
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

        # Векторизоване видалення рядків з NaN
        if X.isna().any().any() or y.isna().any():
            self.logger.warning("Виявлено пропущені значення. Видаляємо рядки з NaN")
            valid_indices = X.notna().all(axis=1) & y.notna()
            X = X.loc[valid_indices]
            y = y.loc[valid_indices]
            self.logger.info(f"Залишилось {len(X)} рядків після видалення NaN")

        # Перевіряємо, що залишились дані для аналізу
        if len(X) == 0 or len(y) == 0:
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

        # Векторизоване заповнення NaN
        if X.isna().any().any():
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
                                   interaction_only: bool = False) -> pd.DataFrame:
        """
        Створення поліноміальних ознак на основі вибраних стовпців.

        Args:
            data: DataFrame з вхідними даними
            columns: Список стовпців для створення поліноміальних ознак
            degree: Степінь поліному
            interaction_only: Якщо True, включає тільки взаємодії без степенів

        Returns:
            DataFrame з доданими поліноміальними ознаками
        """
        self.logger.info("Створення поліноміальних ознак...")

        # Вибираємо числові стовпці, якщо columns не вказано (векторизований підхід)
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
            self.logger.info(f"Автоматично вибрано {len(columns)} числових стовпців")
        else:
            # Перевіряємо наявність вказаних стовпців у даних (векторизований підхід)
            missing_cols = list(set(columns) - set(data.columns))
            if missing_cols:
                self.logger.warning(f"Стовпці {missing_cols} не знайдено в даних і будуть пропущені")
                columns = [col for col in columns if col in data.columns]

        # Перевіряємо, чи залишились стовпці після фільтрації
        if not columns:
            self.logger.error("Немає доступних стовпців для створення поліноміальних ознак")
            return data

        # Створюємо копію DataFrame з вибраними стовпцями
        result_df = data.copy()
        X = result_df[columns]

        # Перевіряємо на наявність NaN і замінюємо їх (векторизований підхід)
        if X.isna().any().any():
            self.logger.warning(f"Виявлено NaN значення у вхідних даних. Заповнюємо їх медіаною.")
            X = X.fillna(X.median())

        # Створюємо об'єкт для поліноміальних ознак
        poly = PolynomialFeatures(degree=degree,
                                  interaction_only=interaction_only,
                                  include_bias=False)

        # Застосовуємо трансформацію
        try:
            poly_features = poly.fit_transform(X)

            # Отримуємо назви нових ознак
            feature_names = poly.get_feature_names_out(X.columns)

            # Створюємо DataFrame з новими ознаками
            poly_df = pd.DataFrame(poly_features,
                                   columns=feature_names,
                                   index=X.index)

            # Видаляємо оригінальні ознаки, оскільки вони будуть дублюватись у вихідному DataFrame
            if degree > 1:
                poly_df = poly_df.iloc[:, len(columns):]

            # Додаємо префікс до назв ознак для уникнення конфліктів
            poly_df = poly_df.add_prefix('poly_')

            # Обробка нескінченних значень і великих чисел (векторизований підхід)
            # Замінюємо нескінченні значення на NaN одним викликом
            poly_df = poly_df.replace([np.inf, -np.inf], np.nan)

            # Знаходимо стовпці з NaN
            cols_with_na = poly_df.columns[poly_df.isna().any()]

            for col in cols_with_na:
                if poly_df[col].isna().all():
                    # Якщо всі значення NaN, заповнюємо нулями
                    poly_df[col] = 0
                else:
                    # Інакше використовуємо медіану для заповнення
                    poly_df[col] = poly_df[col].fillna(poly_df[col].median())

            # Вінсоризація (обрізання екстремальних значень) - векторизований підхід
            # Обчислюємо квантилі для всіх стовпців одночасно
            quantiles = poly_df.quantile([0.01, 0.99])
            q_low, q_high = quantiles.loc[0.01], quantiles.loc[0.99]

            # Застосовуємо .clip для всього DataFrame
            poly_df = poly_df.clip(q_low, q_high, axis=1)

            # Об'єднуємо з вихідним DataFrame
            result_df = pd.concat([result_df, poly_df], axis=1)

            self.logger.info(f"Додано {len(poly_df.columns)} поліноміальних ознак степені {degree}")

        except Exception as e:
            self.logger.error(f"Помилка при створенні поліноміальних ознак: {str(e)}")
            return data

        return result_df

    def create_cluster_features(self, data: pd.DataFrame,
                                n_clusters: int = 5,
                                method: str = 'kmeans') -> pd.DataFrame:
        """
        Створення ознак на основі кластеризації даних.

        Args:
            data: DataFrame з вхідними даними
            n_clusters: Кількість кластерів
            method: Метод кластеризації ('kmeans', 'dbscan', 'hierarchical')

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

        # Замінюємо NaN значення векторизованим способом
        if X.isna().any().any():
            self.logger.warning("Виявлено NaN значення у вхідних даних. Заповнюємо їх медіаною.")
            X = X.fillna(X.median())

        # Стандартизуємо дані
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Словник методів кластеризації
        clustering_methods = {
            'kmeans': self._apply_kmeans_clustering,
            'dbscan': self._apply_dbscan_clustering,
        }

        # Перевіряємо наявність методу
        method = method.lower()
        if method not in clustering_methods:
            self.logger.error(
                f"Невідомий метод кластеризації: {method}. Підтримуються: {list(clustering_methods.keys())}"
            )
            return result_df

        # Застосовуємо вибраний метод кластеризації
        return clustering_methods[method](result_df, X, X_scaled, n_clusters)

    def _apply_kmeans_clustering(self, result_df: pd.DataFrame, X: pd.DataFrame,
                                 X_scaled: np.ndarray, n_clusters: int) -> pd.DataFrame:
        """Допоміжний метод для кластеризації KMeans"""
        try:
            # Визначаємо оптимальну кількість кластерів, якщо n_clusters > 10
            if n_clusters > 10:
                # Векторизований підхід для оцінки кількості кластерів
                scores = []
                range_clusters = range(2, min(11, len(X) // 10))

                for i in range_clusters:
                    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(X_scaled)

                    # Перевіряємо кількість унікальних міток
                    if len(np.unique(cluster_labels)) < i:
                        self.logger.warning(f"Для {i} кластерів отримано менше унікальних міток.")
                        continue

                    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                    scores.append(silhouette_avg)
                    self.logger.debug(f"Для n_clusters = {i}, silhouette score: {silhouette_avg}")

                if scores:
                    best_n_clusters = range_clusters[np.argmax(scores)]
                    self.logger.info(f"Оптимальна кількість кластерів за silhouette score: {best_n_clusters}")
                    n_clusters = best_n_clusters

            # Застосовуємо KMeans з визначеною кількістю кластерів
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            centers = kmeans.cluster_centers_

            # Додаємо мітки кластерів як нову ознаку
            result_df['cluster_label'] = cluster_labels

            # Векторизований підхід для обчислення відстаней до центроїдів
            # Створюємо масив для зберігання всіх відстаней
            distances_array = np.zeros((X_scaled.shape[0], n_clusters))

            for i in range(n_clusters):
                # Обчислюємо відстань від кожної точки до центроїда одним викликом
                distances_array[:, i] = np.linalg.norm(X_scaled - centers[i], axis=1)

            # Додаємо всі відстані як нові ознаки
            for i in range(n_clusters):
                result_df[f'distance_to_cluster_{i}'] = distances_array[:, i]

            self.logger.info(f"Створено {n_clusters + 1} ознак на основі кластеризації KMeans")

        except Exception as e:
            self.logger.error(f"Помилка при кластеризації KMeans: {str(e)}")

        return result_df

    def _apply_dbscan_clustering(self, result_df: pd.DataFrame, X: pd.DataFrame,
                                 X_scaled: np.ndarray, n_clusters: int) -> pd.DataFrame:
        """Допоміжний метод для кластеризації DBSCAN"""
        try:
            # Визначаємо eps (максимальна відстань між сусідніми точками)
            # Використовуємо k-найближчих сусідів (векторизований підхід)
            nbrs = NearestNeighbors(n_neighbors=min(len(X), 5)).fit(X_scaled)
            distances, _ = nbrs.kneighbors(X_scaled)

            # Сортуємо відстані для визначення точки перегину
            knee_distances = np.sort(distances[:, -1])

            # Евристика для eps та min_samples
            eps = np.mean(knee_distances)
            min_samples = max(5, len(X) // 100)

            self.logger.debug(f"DBSCAN параметри: eps={eps}, min_samples={min_samples}")

            # Застосовуємо DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(X_scaled)

            # Додаємо мітки кластерів як нову ознаку
            result_df['dbscan_cluster'] = cluster_labels

            # Рахуємо кількість унікальних кластерів (векторизований підхід)
            unique_labels = np.unique(cluster_labels)
            n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)

            self.logger.info(f"DBSCAN знайдено {n_clusters_found} кластерів")
            self.logger.info(f"Кількість точок-викидів: {np.sum(cluster_labels == -1)}")

            # Для викидів знаходимо найближчий кластер (векторизований підхід)
            if -1 in cluster_labels:
                # Створюємо маски для точок-викидів та не-викидів
                outliers_mask = cluster_labels == -1
                non_outliers_mask = ~outliers_mask

                # Перевіряємо, що є не-викиди
                if np.any(non_outliers_mask):
                    # Навчаємо класифікатор на точках-не викидах
                    knn = KNeighborsClassifier(n_neighbors=3)
                    knn.fit(X_scaled[non_outliers_mask], cluster_labels[non_outliers_mask])

                    # Знаходимо найближчий кластер для викидів
                    if np.any(outliers_mask):
                        closest_clusters = knn.predict(X_scaled[outliers_mask])

                        # Створюємо нову колонку з виправленими мітками
                        fixed_labels = cluster_labels.copy()
                        fixed_labels[outliers_mask] = closest_clusters
                        result_df['dbscan_nearest_cluster'] = fixed_labels

                        # Додаємо бінарну ознаку "чи є точка викидом"
                        result_df['dbscan_is_outlier'] = outliers_mask.astype(int)

            # Якщо знайдено кластери, обчислюємо відстані до центроїдів (векторизований підхід)
            if n_clusters_found > 0:
                # Створюємо словник центроїдів
                centroids = {}
                for i in unique_labels:
                    if i != -1:  # Пропускаємо викиди
                        cluster_mask = cluster_labels == i
                        if np.any(cluster_mask):
                            centroids[i] = np.mean(X_scaled[cluster_mask], axis=0)

                # Обчислюємо відстані до всіх центроїдів одночасно
                for i, centroid in centroids.items():
                    result_df[f'distance_to_dbscan_cluster_{i}'] = np.sqrt(np.sum((X_scaled - centroid) ** 2, axis=1))

            self.logger.info(f"Створено ознаки на основі кластеризації DBSCAN")

        except Exception as e:
            self.logger.error(f"Помилка при кластеризації DBSCAN: {str(e)}")

        return result_df