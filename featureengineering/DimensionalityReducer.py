from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from featureengineering.feature_engineering import FeatureEngineering


class DimensionalityReducer(FeatureEngineering):

    def select_features(self, X: pd.DataFrame, y: pd.Series,
                        n_features: Optional[int] = None,
                        method: str = 'f_regression') -> Tuple[pd.DataFrame, List[str]]:

        self.logger.info(f"Вибір ознак методом '{method}'")

        # Перевіряємо, що X і y мають однакову кількість рядків
        if len(X) != len(y):
            self.logger.error(f"Розмірності X ({len(X)}) і y ({len(y)}) не співпадають")
            raise ValueError(f"Розмірності X ({len(X)}) і y ({len(y)}) не співпадають")

        # Обробляємо пропущені значення в ознаках та цільовій змінній
        if X.isna().any().any() or y.isna().any():
            self.logger.warning("Виявлено пропущені значення. Видаляємо рядки з NaN")
            # Знаходимо індекси рядків без NaN значень
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

        # Вибір методу селекції ознак
        selected_features = []

        if method == 'f_regression':
            self.logger.info("Використовуємо F-тест для відбору ознак")
            selector = SelectKBest(score_func=f_regression, k=n_features)
            selector.fit(X, y)
            # Отримуємо маску вибраних ознак
            selected_mask = selector.get_support()
            # Отримуємо назви вибраних ознак
            selected_features = X.columns[selected_mask].tolist()

            # Логуємо найкращі ознаки з їх оцінками
            scores = selector.scores_
            feature_scores = list(zip(X.columns, scores))
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            self.logger.info(f"Топ-5 ознак за F-тестом: {feature_scores[:5]}")

        elif method == 'mutual_info':
            self.logger.info("Використовуємо взаємну інформацію для відбору ознак")
            selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
            selector.fit(X, y)
            # Отримуємо маску вибраних ознак
            selected_mask = selector.get_support()
            # Отримуємо назви вибраних ознак
            selected_features = X.columns[selected_mask].tolist()

            # Логуємо найкращі ознаки з їх оцінками
            scores = selector.scores_
            feature_scores = list(zip(X.columns, scores))
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            self.logger.info(f"Топ-5 ознак за взаємною інформацією: {feature_scores[:5]}")

        elif method == 'rfe':
            self.logger.info("Використовуємо рекурсивне виключення ознак (RFE)")
            # Для RFE потрібна базова модель, використовуємо лінійну регресію
            model = LinearRegression()
            selector = RFE(estimator=model, n_features_to_select=n_features, step=1)

            try:
                selector.fit(X, y)
                # Отримуємо маску вибраних ознак
                selected_mask = selector.support_
                # Отримуємо назви вибраних ознак
                selected_features = X.columns[selected_mask].tolist()

                # Логуємо ранги ознак (менший ранг означає більшу важливість)
                ranks = selector.ranking_
                feature_ranks = list(zip(X.columns, ranks))
                feature_ranks.sort(key=lambda x: x[1])
                self.logger.info(f"Топ-5 ознак за RFE: {[f[0] for f in feature_ranks[:5]]}")
            except Exception as e:
                self.logger.error(f"Помилка при використанні RFE: {str(e)}. Переходимо до F-тесту.")
                # У випадку помилки використовуємо F-тест
                selector = SelectKBest(score_func=f_regression, k=n_features)
                selector.fit(X, y)
                selected_mask = selector.get_support()
                selected_features = X.columns[selected_mask].tolist()

        else:
            self.logger.error(f"Невідомий метод вибору ознак: {method}")
            raise ValueError(
                f"Невідомий метод вибору ознак: {method}. Допустимі значення: 'f_regression', 'mutual_info', 'rfe'")

        # Створюємо DataFrame з відібраними ознаками
        X_selected = X[selected_features]

        self.logger.info(f"Відібрано {len(selected_features)} ознак: {selected_features[:5]}...")

        return X_selected, selected_features

    def reduce_dimensions(self, data: pd.DataFrame,
                          n_components: Optional[int] = None,
                          method: str = 'pca') -> Tuple[pd.DataFrame, object]:

        self.logger.info(f"Зменшення розмірності методом '{method}'")

        # Створюємо копію, щоб не модифікувати оригінальні дані
        X = data.copy()

        # Перевіряємо наявність пропущених значень
        if X.isna().any().any():
            self.logger.warning("Виявлено пропущені значення. Заповнюємо їх медіаною.")
            X = X.fillna(X.median())

        # Визначаємо кількість компонентів, якщо не вказано
        if n_components is None:
            # За замовчуванням використовуємо sqrt від кількості ознак, але не більше 10
            n_components = min(int(np.sqrt(X.shape[1])), 10)
            self.logger.info(f"Автоматично визначено кількість компонентів: {n_components}")

        # Обмежуємо кількість компонентів доступним числом ознак
        n_components = min(n_components, X.shape[1], X.shape[0])

        # Вибір методу зменшення розмірності
        transformer = None
        X_transformed = None
        component_names = []

        if method == 'pca':
            self.logger.info(f"Застосовуємо PCA з {n_components} компонентами")

            # Створюємо і застосовуємо PCA
            transformer = PCA(n_components=n_components)
            X_transformed = transformer.fit_transform(X)

            # Логування пояснення дисперсії
            explained_variance_ratio = transformer.explained_variance_ratio_
            cumulative_explained_variance = np.cumsum(explained_variance_ratio)
            self.logger.info(f"PCA пояснює {cumulative_explained_variance[-1] * 100:.2f}% загальної дисперсії")
            self.logger.info(f"Перші 3 компоненти пояснюють: {explained_variance_ratio[:3] * 100}")

            # Створюємо назви компонентів
            component_names = [f'pca_component_{i + 1}' for i in range(n_components)]

            # Додаткова інформація: внесок ознак в компоненти
            feature_importance = transformer.components_
            for i in range(min(3, n_components)):
                # Отримуємо абсолютні значення важливості ознак для компоненти
                abs_importance = np.abs(feature_importance[i])
                # Сортуємо індекси за важливістю
                sorted_indices = np.argsort(abs_importance)[::-1]
                # Виводимо топ-5 ознак для компоненти
                top_features = [(X.columns[idx], feature_importance[i, idx]) for idx in sorted_indices[:5]]
                self.logger.info(f"Компонента {i + 1} найбільше залежить від: {top_features}")

        elif method == 'kmeans':
            self.logger.info(f"Застосовуємо KMeans з {n_components} кластерами")

            # Створюємо і застосовуємо KMeans
            transformer = KMeans(n_clusters=n_components, random_state=42)
            cluster_labels = transformer.fit_predict(X)

            # Створюємо двовимірний масив з мітками кластерів
            X_transformed = np.zeros((X.shape[0], n_components))
            for i in range(X.shape[0]):
                X_transformed[i, cluster_labels[i]] = 1

            # Створюємо назви компонентів (кластерів)
            component_names = [f'cluster_{i + 1}' for i in range(n_components)]

            # Додаткова інформація: розмір кластерів
            cluster_sizes = np.bincount(cluster_labels)
            cluster_info = list(zip(range(1, n_components + 1), cluster_sizes))
            self.logger.info(f"Розмір кластерів: {cluster_info}")

            # Додаткова інформація: центроїди кластерів
            centroids = transformer.cluster_centers_
            for i in range(min(3, n_components)):
                # Знаходимо ознаки, які найбільше відрізняються від глобального середнього
                mean_values = X.mean().values
                centroid_diff = centroids[i] - mean_values
                abs_diff = np.abs(centroid_diff)
                sorted_indices = np.argsort(abs_diff)[::-1]
                # Виводимо топ-5 відмінних ознак для кластера
                top_features = [(X.columns[idx], centroid_diff[idx]) for idx in sorted_indices[:5]]
                self.logger.info(f"Кластер {i + 1} характеризується: {top_features}")

        else:
            self.logger.error(f"Невідомий метод зменшення розмірності: {method}")
            raise ValueError(f"Невідомий метод зменшення розмірності: {method}. Допустимі значення: 'pca', 'kmeans'")

        # Створюємо DataFrame з трансформованими даними
        result_df = pd.DataFrame(X_transformed, index=X.index, columns=component_names)

        self.logger.info(f"Розмірність зменшено з {X.shape[1]} до {result_df.shape[1]} ознак")

        return result_df, transformer
    def create_polynomial_features(self, data: pd.DataFrame,
                                   columns: Optional[List[str]] = None,
                                   degree: int = 2,
                                   interaction_only: bool = False) -> pd.DataFrame:

        self.logger.info("Створення поліноміальних ознак...")

        # Вибираємо числові стовпці, якщо columns не вказано
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
            self.logger.info(f"Автоматично вибрано {len(columns)} числових стовпців")
        else:
            # Перевіряємо наявність вказаних стовпців у даних
            missing_cols = [col for col in columns if col not in data.columns]
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

        # Перевіряємо на наявність NaN і замінюємо їх
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
            # (перші n стовпців у poly_features відповідають оригінальним ознакам)
            if degree > 1:
                poly_df = poly_df.iloc[:, len(columns):]

            # Додаємо префікс до назв ознак для уникнення конфліктів
            poly_df = poly_df.add_prefix('poly_')

            # Об'єднуємо з вихідним DataFrame
            result_df = pd.concat([result_df, poly_df], axis=1)

            # Перевіряємо на нескінченні значення або великі числа
            for col in poly_df.columns:
                if result_df[col].isna().any() or np.isinf(result_df[col]).any():
                    self.logger.warning(
                        f"Виявлено NaN або нескінченні значення у стовпці {col}. Заповнюємо їх медіаною.")
                    result_df[col] = result_df[col].replace([np.inf, -np.inf], np.nan)
                    if result_df[col].isna().all():
                        # Якщо всі значення NaN, заповнюємо нулями
                        result_df[col] = 0
                    else:
                        # Інакше використовуємо медіану для заповнення
                        result_df[col] = result_df[col].fillna(result_df[col].median())

                # Опціонально можна обмежити великі значення (вінсоризація)
                q_low, q_high = result_df[col].quantile([0.01, 0.99])
                result_df[col] = result_df[col].clip(q_low, q_high)

            self.logger.info(f"Додано {len(poly_df.columns)} поліноміальних ознак степені {degree}")

        except Exception as e:
            self.logger.error(f"Помилка при створенні поліноміальних ознак: {str(e)}")
            return data

        return result_df
    def create_cluster_features(self, data: pd.DataFrame,
                                n_clusters: int = 5,
                                method: str = 'kmeans') -> pd.DataFrame:

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

        # Замінюємо NaN значення
        if X.isna().any().any():
            self.logger.warning("Виявлено NaN значення у вхідних даних. Заповнюємо їх медіаною.")
            X = X.fillna(X.median())

        # Стандартизуємо дані
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Вибір методу кластеризації
        if method.lower() == 'kmeans':
            # KMeans кластеризація
            try:
                # Визначаємо оптимальну кількість кластерів, якщо n_clusters > 10
                if n_clusters > 10:
                    from sklearn.metrics import silhouette_score
                    scores = []
                    range_clusters = range(2, min(11, len(X) // 10))  # Обмежуємо максимальну кількість кластерів

                    for i in range_clusters:
                        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
                        cluster_labels = kmeans.fit_predict(X_scaled)

                        # Перевіряємо, що кількість унікальних міток відповідає очікуваній
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

                # Обчислюємо відстань до кожного центроїда
                for i in range(n_clusters):
                    # Для кожного кластера обчислюємо відстань від кожної точки до центроїда
                    if hasattr(kmeans, 'feature_names_in_'):
                        # Для новіших версій scikit-learn
                        distances = np.linalg.norm(X_scaled - centers[i], axis=1)
                    else:
                        # Для старіших версій, обчислюємо вручну
                        distances = np.sqrt(np.sum((X_scaled - centers[i]) ** 2, axis=1))

                    result_df[f'distance_to_cluster_{i}'] = distances

                self.logger.info(f"Створено {n_clusters + 1} ознак на основі кластеризації KMeans")

            except Exception as e:
                self.logger.error(f"Помилка при кластеризації KMeans: {str(e)}")
                return result_df

        elif method.lower() == 'dbscan':
            # DBSCAN кластеризація
            try:
                from sklearn.cluster import DBSCAN
                from sklearn.neighbors import KNeighborsClassifier

                # Визначаємо eps (максимальна відстань між сусідніми точками)
                # Можна використати евристику на основі відстаней до k-го найближчого сусіда
                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=min(len(X), 5)).fit(X_scaled)
                distances, _ = nbrs.kneighbors(X_scaled)

                # Сортуємо відстані для визначення точки перегину
                knee_distances = np.sort(distances[:, -1])

                # Евристика для eps: точка перегину на графіку відсортованих відстаней або середня відстань
                eps = np.mean(knee_distances)
                min_samples = max(5, len(X) // 100)  # Евристика для min_samples

                self.logger.debug(f"DBSCAN параметри: eps={eps}, min_samples={min_samples}")

                # Застосовуємо DBSCAN
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                cluster_labels = dbscan.fit_predict(X_scaled)

                # Додаємо мітки кластерів як нову ознаку
                result_df['dbscan_cluster'] = cluster_labels

                # Рахуємо кількість унікальних кластерів (без врахування викидів з міткою -1)
                n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

                self.logger.info(f"DBSCAN знайдено {n_clusters_found} кластерів")
                self.logger.info(f"Кількість точок-викидів: {sum(cluster_labels == -1)}")

                # Для викидів (точок з міткою -1) знайдемо найближчий кластер
                if -1 in cluster_labels:
                    # Навчаємо класифікатор KNN на точках, які належать до кластерів
                    mask = cluster_labels != -1
                    if sum(mask) > 0:  # Перевіряємо, що є точки не викиди
                        knn = KNeighborsClassifier(n_neighbors=3)
                        knn.fit(X_scaled[mask], cluster_labels[mask])

                        # Для викидів знаходимо найближчий кластер і відстань до нього
                        outliers_mask = cluster_labels == -1
                        closest_clusters = knn.predict(X_scaled[outliers_mask])

                        # Замінюємо -1 на мітку найближчого кластера
                        cluster_labels_fixed = cluster_labels.copy()
                        cluster_labels_fixed[outliers_mask] = closest_clusters
                        result_df['dbscan_nearest_cluster'] = cluster_labels_fixed

                        # Додаємо ознаку, що вказує чи є точка викидом
                        result_df['dbscan_is_outlier'] = outliers_mask.astype(int)

                # Якщо знайдено кластери, додаємо відстані до центроїдів
                if n_clusters_found > 0:
                    # Обчислюємо центроїди кластерів (крім викидів)
                    centroids = {}
                    for i in range(n_clusters_found):
                        cluster_idx = np.where(cluster_labels == i)[0]
                        if len(cluster_idx) > 0:
                            centroids[i] = np.mean(X_scaled[cluster_idx], axis=0)

                    # Обчислюємо відстані до центроїдів
                    for i, centroid in centroids.items():
                        # Для кожного кластера обчислюємо відстань від кожної точки до центроїда
                        distances = np.sqrt(np.sum((X_scaled - centroid) ** 2, axis=1))
                        result_df[f'distance_to_dbscan_cluster_{i}'] = distances

                self.logger.info(f"Створено ознаки на основі кластеризації DBSCAN")

            except Exception as e:
                self.logger.error(f"Помилка при кластеризації DBSCAN: {str(e)}")
                return result_df

        elif method.lower() == 'hierarchical':
            # Агломеративна кластеризація
            try:
                from sklearn.cluster import AgglomerativeClustering

                # Застосовуємо агломеративну кластеризацію
                agg = AgglomerativeClustering(n_clusters=n_clusters)
                cluster_labels = agg.fit_predict(X_scaled)

                # Додаємо мітки кластерів як нову ознаку
                result_df['hierarchical_cluster'] = cluster_labels

                # Обчислюємо центроїди кластерів
                centroids = {}
                for i in range(n_clusters):
                    cluster_idx = np.where(cluster_labels == i)[0]
                    centroids[i] = np.mean(X_scaled[cluster_idx], axis=0)

                # Обчислюємо відстані до центроїдів
                for i, centroid in centroids.items():
                    # Для кожного кластера обчислюємо відстань від кожної точки до центроїда
                    distances = np.sqrt(np.sum((X_scaled - centroid) ** 2, axis=1))
                    result_df[f'distance_to_hier_cluster_{i}'] = distances

                self.logger.info(f"Створено {n_clusters + 1} ознак на основі ієрархічної кластеризації")

            except Exception as e:
                self.logger.error(f"Помилка при ієрархічній кластеризації: {str(e)}")
                return result_df

        else:
            self.logger.error(
                f"Невідомий метод кластеризації: {method}. Підтримуються: 'kmeans', 'dbscan', 'hierarchical'")
            return result_df

        return result_df
