import argparse
import os
from datetime import datetime
import numpy as np

from data.db import DatabaseManager
from data_collection.NewsCollector import NewsCollector
from models.NewsAnalyzer import BERTNewsAnalyzer
from data.NewsManager import NewsStorage


class CryptoNewsScraper:
    """
    Комплексний клас для збору, аналізу та зберігання новин про криптовалюти.
    Об'єднує функціональність NewsCollector, BERTNewsAnalyzer та NewsStorage.
    """

    def __init__(self, db_connection=None, bert_model_path=None, topic_models_path=None,
                 use_proxies=False, user_agents=None, feedly_api_key=None):
        """
        Ініціалізація скрапера новин про криптовалюти.

        Args:
            db_connection: З'єднання з базою даних для зберігання
            bert_model_path: Шлях до попередньо навченої BERT моделі
            topic_models_path: Шлях до попередньо навчених моделей для виділення тем
            use_proxies: Чи використовувати проксі для запитів
            user_agents: Список агентів користувача для ротації
            feedly_api_key: API ключ для доступу до Feedly
        """
        # Ініціалізація колектора новин
        self.collector = NewsCollector()

        # Ініціалізація бази даних
        self.db_manager = DatabaseManager()

        # Ініціалізація аналізатора новин
        self.analyzer = BERTNewsAnalyzer()

        # Ініціалізація сховища новин
        self.storage = NewsStorage()

        # Налаштування та кешування даних
        self._cached_news = []
        self._cached_analysis = {}

    def collect_news(self, sources=None, days_back=1, limit_per_source=50):
        """
        Збирає новини з усіх або вказаних джерел.

        Args:
            sources: Список джерел для скрапінгу (якщо None, використовуються всі)
            days_back: За скільки днів назад збирати новини
            limit_per_source: Максимальна кількість новин з кожного джерела

        Returns:
            Список зібраних новин
        """
        if sources is None:
            # Збір новин з усіх доступних джерел
            news = self.collector.scrape_all_sources(days_back=days_back, limit_per_source=limit_per_source)
        else:
            news = []
            # Збір новин з конкретних джерел
            for source in sources:
                if source == "coindesk":
                    news.extend(self.collector.scrape_coindesk(days_back=days_back, limit=limit_per_source))
                elif source == "cointelegraph":
                    news.extend(self.collector.scrape_cointelegraph(days_back=days_back, limit=limit_per_source))
                elif source == "decrypt":
                    news.extend(self.collector.scrape_decrypt(days_back=days_back, limit=limit_per_source))
                elif source == "cryptoslate":
                    news.extend(self.collector.scrape_cryptoslate(days_back=days_back, limit=limit_per_source))
                elif source == "theblock":
                    news.extend(self.collector.scrape_theblock(days_back=days_back, limit=limit_per_source))
                elif source == "cryptobriefing":
                    news.extend(self.collector.scrape_cryptobriefing(days_back=days_back, limit=limit_per_source))
                elif source == "cryptopanic":
                    news.extend(self.collector.scrape_cryptopanic(days_back=days_back, limit=limit_per_source))
                elif source == "coinmarketcal":
                    news.extend(self.collector.scrape_coinmarketcal(days_back=days_back, limit=limit_per_source))
                elif source == "feedly":
                    news.extend(self.collector.scrape_feedly(days_back=days_back, limit=limit_per_source))
                elif source == "newsnow":
                    news.extend(self.collector.scrape_newsnow(days_back=days_back, limit=limit_per_source))

        self._cached_news = news
        return news

    def analyze_news(self, news=None, keywords=None, coins=None, batch_size=16):
        """
        Аналізує зібрані новини, застосовуючи різні методи аналізу.

        Args:
            news: Список новин для аналізу (якщо None, використовується кеш)
            keywords: Ключові слова для фільтрації
            coins: Конкретні криптовалюти для фільтрації
            batch_size: Розмір партії для пакетної обробки

        Returns:
            Результати аналізу новин
        """
        if news is None:
            news = self._cached_news
            if not news:
                raise ValueError("Спочатку потрібно зібрати новини за допомогою методу collect_news()")

        # Фільтрація за ключовими словами, якщо вказано
        if keywords:
            news = self.analyzer.filter_by_keywords(news, keywords)

        # Виконання пакетного аналізу всіх новин
        analysis_results = self.analyzer.analyze_news_batch(news, batch_size=batch_size)

        # Якщо вказані конкретні монети, відфільтруємо результати
        if coins:
            filtered_results = []
            for result in analysis_results:
                mentioned_coins = result.get('mentioned_coins', [])
                if any(coin in mentioned_coins for coin in coins):
                    filtered_results.append(result)
            analysis_results = filtered_results

        # Збереження результатів аналізу в кеш
        self._cached_analysis = {
            'individual_results': analysis_results,
            'trending_topics': self.analyzer.get_trending_topics(analysis_results),
            'sentiment_trends': self.analyzer.analyze_sentiment_trends(analysis_results),
            'market_signals': self.analyzer.identify_market_signals(analysis_results)
        }

        return self._cached_analysis

    def store_data(self, news=None, analysis_results=None):
        """
        Зберігає зібрані новини та результати аналізу в базу даних.

        Args:
            news: Список новин для збереження (якщо None, використовується кеш)
            analysis_results: Результати аналізу для збереження (якщо None, використовується кеш)

        Returns:
            Кількість збережених записів
        """
        if news is None:
            news = self._cached_news

        if analysis_results is None:
            analysis_results = self._cached_analysis.get('individual_results', [])

        # Комбінуємо новини та результати аналізу
        news_with_analysis = []
        for i, news_item in enumerate(news):
            if i < len(analysis_results):
                # Об'єднуємо оригінальні дані новини з результатами аналізу
                news_item.update(analysis_results[i])
            news_with_analysis.append(news_item)

        # Зберігаємо новини пакетом
        stored_count = self.storage.store_news_batch(news_with_analysis)

        # Зберігаємо додаткові дані аналізу
        if self._cached_analysis:
            self.storage.store_news_collector_data({
                'trending_topics': self._cached_analysis.get('trending_topics', []),
                'sentiment_trends': self._cached_analysis.get('sentiment_trends', {}),
                'market_signals': self._cached_analysis.get('market_signals', {})
            })

        return stored_count

    def get_news_summary(self, news=None, max_items=5):
        """
        Отримує короткі підсумки найважливіших новин.

        Args:
            news: Список новин для підсумовування (якщо None, використовується кеш)
            max_items: Максимальна кількість новин у підсумку

        Returns:
            Список найважливіших новин з підсумками
        """
        if news is None:
            news = self._cached_news

        # Сортуємо новини за важливістю
        sorted_news = sorted(
            news,
            key=lambda item: self.analyzer.calculate_importance_score(item),
            reverse=True
        )[:max_items]

        # Отримуємо підсумки для кожної новини
        summaries = []
        for news_item in sorted_news:
            summary = self.analyzer.get_news_summary(news_item)
            summaries.append({
                'title': news_item.get('title', ''),
                'source': news_item.get('source', ''),
                'url': news_item.get('url', ''),
                'summary': summary,
                'sentiment': news_item.get('sentiment', 'neutral'),
                'mentioned_coins': news_item.get('mentioned_coins', [])
            })

        return summaries

    def analyze_news_clusters(self, threshold=0.7):
        """
        Групує та аналізує кластери пов'язаних новин.

        Args:
            threshold: Поріг схожості для кластеризації

        Returns:
            Список кластерів новин з їх аналізом
        """
        if not self._cached_news:
            raise ValueError("Спочатку потрібно зібрати новини за допомогою методу collect_news()")

        # Отримуємо ембеддінги для всіх новин
        news_texts = [news.get('content', news.get('title', '')) for news in self._cached_news]
        embeddings = [
            self.analyzer._get_bert_embeddings(text)
            for text in news_texts
        ]

        # Виконуємо кластеризацію на основі косинусної схожості
        clusters = []
        used_indices = set()

        for i in range(len(self._cached_news)):
            if i in used_indices:
                continue

            cluster = [i]
            used_indices.add(i)

            for j in range(len(self._cached_news)):
                if j in used_indices or i == j:
                    continue

                # Обчислюємо косинусну схожість між ембеддінгами
                similarity = self._calculate_similarity(embeddings[i], embeddings[j])
                if similarity >= threshold:
                    cluster.append(j)
                    used_indices.add(j)

            if len(cluster) > 1:  # Додаємо тільки справжні кластери (більше 1 новини)
                cluster_news = [self._cached_news[idx] for idx in cluster]
                analysis = self.analyzer.analyze_news_cluster(cluster_news)
                clusters.append({
                    'news': cluster_news,
                    'analysis': analysis
                })

        return clusters

    def run_full_pipeline(self, sources=None, days_back=1, keywords=None, coins=None,
                          store_results=True, get_summary=True, analyze_clusters=True):
        """
        Виконує повний цикл збору, аналізу та зберігання новин.

        Args:
            sources: Список джерел для скрапінгу
            days_back: За скільки днів назад збирати новини
            keywords: Ключові слова для фільтрації
            coins: Конкретні криптовалюти для фільтрації
            store_results: Чи зберігати результати в базу даних
            get_summary: Чи отримувати підсумки найважливіших новин
            analyze_clusters: Чи аналізувати кластери пов'язаних новин

        Returns:
            Словник з результатами роботи
        """
        # Збір новин
        news = self.collect_news(sources=sources, days_back=days_back)
        print(f"Зібрано {len(news)} новин з {len(set(item.get('source', '') for item in news))} джерел")

        # Аналіз новин
        analysis = self.analyze_news(news=news, keywords=keywords, coins=coins)
        print(f"Проаналізовано {len(analysis['individual_results'])} новин")

        results = {
            'news_count': len(news),
            'analysis': analysis
        }

        # Зберігання результатів, якщо потрібно
        if store_results:
            stored_count = self.store_data(news=news, analysis_results=analysis['individual_results'])
            results['stored_count'] = stored_count
            print(f"Збережено {stored_count} новин у базу даних")

        # Отримання підсумків, якщо потрібно
        if get_summary:
            summaries = self.get_news_summary(news=news)
            results['summaries'] = summaries
            print(f"Створено {len(summaries)} підсумків найважливіших новин")

        # Аналіз кластерів, якщо потрібно
        if analyze_clusters:
            clusters = self.analyze_news_clusters()
            results['clusters'] = clusters
            print(f"Виявлено {len(clusters)} кластерів пов'язаних новин")

        return results

    def _calculate_similarity(self, embedding1, embedding2):
        """
        Допоміжний метод для обчислення косинусної схожості між ембеддінгами.

        Args:
            embedding1: Перший ембеддінг
            embedding2: Другий ембеддінг

        Returns:
            Значення косинусної схожості
        """
        # Нормалізуємо вектори
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0

        # Обчислюємо косинусну схожість
        return np.dot(embedding1, embedding2) / (norm1 * norm2)


def main():
    """
    Головна функція для запуску скрапера криптовалютних новин.
    Аналізує аргументи командного рядка та запускає процес збору і аналізу новин.
    """
    # Налаштування парсера аргументів командного рядка
    parser = argparse.ArgumentParser(description="Скрапер криптовалютних новин")
    parser.add_argument("--sources", nargs="+", default=None,
                        help="Список джерел для скрапінгу (наприклад, coindesk cointelegraph)")
    parser.add_argument("--days-back", type=int, default=1,
                        help="За скільки днів назад збирати новини")
    parser.add_argument("--keywords", nargs="+", default=None,
                        help="Ключові слова для фільтрації новин")
    parser.add_argument("--coins", nargs="+", default=None,
                        help="Список криптовалют для фільтрації (наприклад, BTC ETH)")
    parser.add_argument("--no-store", action="store_true",
                        help="Не зберігати результати в базу даних")
    parser.add_argument("--no-summary", action="store_true",
                        help="Не створювати підсумки новин")
    parser.add_argument("--no-clusters", action="store_true",
                        help="Не аналізувати кластери новин")
    parser.add_argument("--use-proxies", action="store_true",
                        help="Використовувати проксі для запитів")
    parser.add_argument("--db-path", type=str, default="crypto_news.db",
                        help="Шлях до файлу бази даних SQLite")
    parser.add_argument("--bert-model", type=str, default=None,
                        help="Шлях до попередньо навченої BERT моделі")
    parser.add_argument("--topic-models", type=str, default=None,
                        help="Шлях до моделей для виділення тем")
    parser.add_argument("--feedly-api-key", type=str, default=None,
                        help="API ключ для доступу до Feedly")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Директорія для збереження результатів")

    args = parser.parse_args()

    # Створення директорії для результатів, якщо вона не існує
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Інформація про запуск
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Запуск скрапера криптовалютних новин")

    # Встановлення з'єднання з базою даних
    db_connection = DatabaseManager()
    print(f"З'єднання з базою даних встановлено: {args.db_path}")

    # Базовий набір user-agents для ротації
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:93.0) Gecko/20100101 Firefox/93.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1"
    ]

    # Ініціалізація скрапера
    scraper = CryptoNewsScraper(
        db_connection=db_connection,
        bert_model_path=args.bert_model,
        topic_models_path=args.topic_models,
        use_proxies=args.use_proxies,
        user_agents=user_agents,
    )
    print("Скрапер ініціалізовано")

    try:
        # Запуск повного циклу
        results = scraper.run_full_pipeline(
            sources=args.sources,
            days_back=args.days_back,
            keywords=args.keywords,
            coins=args.coins,
            store_results=not args.no_store,
            get_summary=not args.no_summary,
            analyze_clusters=not args.no_clusters
        )

        # Виведення статистики
        print("\n=== Статистика ===")
        print(f"Всього зібрано новин: {results['news_count']}")
        print(f"Виявлено трендових тем: {len(results['analysis']['trending_topics'])}")

        # Збереження результатів у JSON файл
        if 'summaries' in results and results['summaries']:
            import json
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = os.path.join(args.output_dir, f"news_summary_{timestamp}.json")

            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(results['summaries'], f, ensure_ascii=False, indent=2)
            print(f"Зведення новин збережено у файлі: {summary_file}")

            # Виведення стислого зведення важливих новин
            print("\n=== Найважливіші новини ===")
            for i, summary in enumerate(results['summaries'], 1):
                print(f"{i}. {summary['title']} ({summary['source']})")
                print(f"   {summary['summary']}")
                print(f"   Монети: {', '.join(summary['mentioned_coins'])}")
                print(f"   Тональність: {summary['sentiment']}")
                print(f"   URL: {summary['url']}")
                print()

        # Вивід інформації про кластери
        if 'clusters' in results and results['clusters']:
            print(f"\n=== Виявлено {len(results['clusters'])} груп пов'язаних новин ===")
            for i, cluster in enumerate(results['clusters'], 1):
                print(f"Група {i}: {len(cluster['news'])} новин")
                print(f"Загальна тема: {cluster['analysis'].get('common_topic', 'Не визначено')}")
                print(f"Основна тональність: {cluster['analysis'].get('overall_sentiment', 'нейтральна')}")
                print()

    except Exception as e:
        print(f"Помилка під час виконання: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Закриття з'єднання з базою даних
        if 'db_connection' in locals() and db_connection:
            db_connection.close()
            print("З'єднання з базою даних закрито")

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Завершено роботу скрапера")


if __name__ == "__main__":
    main()