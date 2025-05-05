import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import logging
import re
import time
import praw
from typing import List, Dict, Optional, Union, Tuple, Any
from random import randint
from data.db import DatabaseManager
from utils.logger import CryptoLogger
from scipy import stats


class NewsAnalyzer:
    """
    Клас для аналізу новин криптовалютного ринку, включаючи аналіз настроїв,
    виявлення тенденцій та кореляцію з ринковими даними.
    """

    # Словник з популярними криптовалютами та їх скороченнями/синонімами
    CRYPTO_KEYWORDS = {
        'bitcoin': ['btc', 'xbt', 'bitcoin', 'биткоин', 'біткоїн'],
        'ethereum': ['eth', 'ethereum', 'эфириум', 'етеріум', 'ether'],
        'ripple': ['xrp', 'ripple'],
        'litecoin': ['ltc', 'litecoin'],
        'cardano': ['ada', 'cardano'],
        'polkadot': ['dot', 'polkadot'],
        'binance coin': ['bnb', 'binance coin', 'binance'],
        'dogecoin': ['doge', 'dogecoin'],
        'solana': ['sol', 'solana'],
        'tron': ['trx', 'tron'],
        'tether': ['usdt', 'tether'],
        'usd coin': ['usdc', 'usd coin'],
        'avalanche': ['avax', 'avalanche'],
        'chainlink': ['link', 'chainlink'],
        'polygon': ['matic', 'polygon'],
        'stellar': ['xlm', 'stellar'],
        'cosmos': ['atom', 'cosmos'],
        'vechain': ['vet', 'vechain'],
        'algorand': ['algo', 'algorand'],
        'uniswap': ['uni', 'uniswap'],
        'shiba inu': ['shib', 'shiba inu', 'shiba'],
        'filecoin': ['fil', 'filecoin'],
        'monero': ['xmr', 'monero'],
        'aave': ['aave'],
        'maker': ['mkr', 'maker'],
        'compound': ['comp', 'compound'],
        'decentraland': ['mana', 'decentraland']
    }

    # Ключові слова, що вказують на потенційно важливі події
    CRITICAL_KEYWORDS = {
        'regulation': ['regulation', 'регуляція', 'закон', 'заборона', 'легалізація', 'SEC', 'CFTC'],
        'hack': ['hack', 'хакер', 'зламали', 'атака', 'викрадено', 'вкрадено', 'безпека'],
        'market_crash': ['crash', 'collapse', 'обвал', 'крах', 'падіння', 'bear market', 'ведмежий'],
        'market_boom': ['boom', 'rally', 'ріст', 'буйк', 'bull market', 'бичачий', 'ath', 'all-time high'],
        'merge': ['merge', 'злиття', 'acquisition', 'поглинання', 'buyout', 'викуп'],
        'fork': ['fork', 'форк', 'hard fork', 'soft fork', 'chain split', 'розділення'],
        'adoption': ['adoption', 'впровадження', 'integration', 'інтеграція', 'partnership', 'партнерство'],
        'scandal': ['scandal', 'скандал', 'controversy', 'контроверсія', 'fraud', 'шахрайство'],
        'lawsuit': ['lawsuit', 'позов', 'court', 'суд', 'legal action', 'legal', 'investigation'],
        'innovation': ['innovation', 'інновація', 'breakthrough', 'прорив', 'launch', 'запуск']
    }

    # Слова, які часто зустрічаються і не несуть специфічного значення
    COMMON_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
        'to', 'for', 'with', 'by', 'about', 'as', 'of', 'from',
        'that', 'this', 'these', 'those', 'is', 'are', 'was', 'were',
        'has', 'have', 'had', 'been', 'will', 'would', 'could', 'should'
    }

    # Основні компанії/проекти в криптосфері
    MAJOR_ENTITIES = [
        'bitcoin', 'ethereum', 'binance', 'coinbase', 'ripple', 'tether',
        'ftx', 'metamask', 'opensea', 'uniswap', 'solana', 'avalanche'
    ]

    def __init__(self, sentiment_analyzer=None, logger=None):
        """
        Ініціалізує аналізатор новин.

        Args:
            sentiment_analyzer: Аналізатор настроїв текстів
            logger: Об'єкт логера для запису подій
        """
        self.sentiment_analyzer = sentiment_analyzer
        self.logger = logger or CryptoLogger("news_analyzer").get_logger()

        # Попередня компіляція регулярних виразів для криптовалют
        self.coin_patterns = self._compile_coin_patterns()

        # Попередня компіляція регулярних виразів для критичних подій
        self.category_patterns = self._compile_category_patterns()

    def _compile_coin_patterns(self) -> Dict[str, re.Pattern]:
        """
        Компілює регулярні вирази для пошуку криптовалют.

        Returns:
            Dict[str, re.Pattern]: Словник із скомпільованими шаблонами
        """
        coin_patterns = {}
        for coin, aliases in self.CRYPTO_KEYWORDS.items():
            # Створюємо шаблон регулярного виразу для кожної монети та її аліасів
            pattern = r'\b(?i)(' + '|'.join(map(re.escape, aliases)) + r')\b'
            coin_patterns[coin] = re.compile(pattern)
        return coin_patterns

    def _compile_category_patterns(self) -> Dict[str, List[re.Pattern]]:
        """
        Компілює регулярні вирази для категорій важливих подій.

        Returns:
            Dict[str, List[re.Pattern]]: Словник із скомпільованими шаблонами
        """
        category_patterns = {}
        for category, keywords in self.CRITICAL_KEYWORDS.items():
            patterns = [re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE) for keyword in keywords]
            category_patterns[category] = patterns
        return category_patterns

    def _get_text_to_analyze(self, news: Dict[str, Any]) -> str:
        """
        Витягує текст для аналізу з новини.

        Args:
            news: Словник з даними новини

        Returns:
            str: Комбінований текст для аналізу
        """
        return f"{news.get('title', '')} {news.get('summary', '')}"

    def analyze_news_sentiment(self, news_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Аналізує настрої в текстах новин.

        Args:
            news_data: Список словників з даними новин

        Returns:
            List[Dict[str, Any]]: Список новин з доданим аналізом настроїв
        """
        self.logger.info(f"Початок аналізу настроїв для {len(news_data)} новин")

        # Перевірка наявності аналізатора настроїв
        if not self.sentiment_analyzer:
            self.logger.error("Аналізатор настроїв не ініціалізований")
            # Додаємо нейтральний настрій за замовчуванням
            for news in news_data:
                news['sentiment'] = {
                    'score': 0.0,  # Нейтральний настрій
                    'label': 'neutral',
                    'confidence': 0.0,
                    'analyzed': False
                }
            return news_data

        analyzed_news = []

        for idx, news in enumerate(news_data):
            try:
                # Текст для аналізу
                text_to_analyze = self._get_text_to_analyze(news)

                # Викликаємо аналізатор настроїв
                sentiment_result = self.sentiment_analyzer.analyze(text_to_analyze)

                # Копіюємо новину та додаємо результат аналізу
                news_with_sentiment = news.copy()

                # Форматуємо результат аналізу
                if isinstance(sentiment_result, dict):
                    news_with_sentiment['sentiment'] = sentiment_result
                else:
                    # Якщо результат не у вигляді словника, створюємо базову структуру
                    news_with_sentiment['sentiment'] = {
                        'score': getattr(sentiment_result, 'score', 0.0),
                        'label': getattr(sentiment_result, 'label', 'neutral'),
                        'confidence': getattr(sentiment_result, 'confidence', 0.0),
                        'analyzed': True
                    }

                analyzed_news.append(news_with_sentiment)

                # Логування прогресу (кожні 50 новин)
                if idx > 0 and idx % 50 == 0:
                    self.logger.info(f"Проаналізовано {idx}/{len(news_data)} новин")

            except Exception as e:
                self.logger.error(f"Помилка при аналізі настроїв для новини '{news.get('title', 'unknown')}': {e}")
                # Додаємо новину з нейтральним настроєм у випадку помилки
                news['sentiment'] = {
                    'score': 0.0,
                    'label': 'neutral',
                    'confidence': 0.0,
                    'analyzed': False,
                    'error': str(e)
                }
                analyzed_news.append(news)

        self.logger.info(f"Аналіз настроїв завершено для {len(analyzed_news)} новин")
        return analyzed_news

    def extract_mentioned_coins(self, news_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Виявляє згадки криптовалют у новинах.

        Args:
            news_data: Список словників з даними новин

        Returns:
            List[Dict[str, Any]]: Список новин з доданою інформацією про згадані криптовалюти
        """
        self.logger.info(f"Початок пошуку згаданих криптовалют у {len(news_data)} новинах")

        for news in news_data:
            try:
                # Текст для аналізу
                text_to_analyze = self._get_text_to_analyze(news)

                # Ініціалізуємо словник для згаданих монет та їх кількості
                mentioned_coins = {}

                # Пошук кожної монети в тексті
                for coin, pattern in self.coin_patterns.items():
                    matches = pattern.findall(text_to_analyze)
                    if matches:
                        # Записуємо кількість згадок
                        mentioned_coins[coin] = len(matches)

                # Сортуємо монети за кількістю згадок (в порядку спадання)
                sorted_mentions = sorted(
                    mentioned_coins.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                # Формуємо структурований результат
                news['mentioned_coins'] = {
                    'coins': {coin: count for coin, count in sorted_mentions},
                    'top_mentioned': sorted_mentions[0][0] if sorted_mentions else None,
                    'total_coins': len(sorted_mentions)
                }

            except Exception as e:
                self.logger.error(
                    f"Помилка при пошуку згаданих криптовалют для новини '{news.get('title', 'unknown')}': {e}")
                # Додаємо порожнє поле у випадку помилки
                news['mentioned_coins'] = {
                    'coins': {},
                    'top_mentioned': None,
                    'total_coins': 0,
                    'error': str(e)
                }

        self.logger.info("Пошук згаданих криптовалют завершено")
        return news_data

    def filter_by_keywords(self, news_data: List[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Фільтрує новини за наявністю ключових слів.

        Args:
            news_data: Список словників з даними новин
            keywords: Список ключових слів для пошуку

        Returns:
            List[Dict[str, Any]]: Список відфільтрованих новин
        """
        self.logger.info(f"Початок фільтрації {len(news_data)} новин за {len(keywords)} ключовими словами")

        if not keywords or not news_data:
            self.logger.warning("Порожній список ключових слів або новин для фільтрації")
            return news_data

        # Підготовка регулярних виразів для пошуку (нечутливість до регістру)
        keyword_patterns = [re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE) for keyword in keywords]

        filtered_news = []

        for news in news_data:
            try:
                # Текст для аналізу
                text_to_analyze = self._get_text_to_analyze(news)

                # Перевірка на наявність хоча б одного ключового слова
                matched_keywords = []

                for i, pattern in enumerate(keyword_patterns):
                    if pattern.search(text_to_analyze):
                        matched_keywords.append(keywords[i])

                if matched_keywords:
                    # Копіюємо новину та додаємо інформацію про знайдені ключові слова
                    matched_news = news.copy()
                    matched_news['matched_keywords'] = matched_keywords
                    filtered_news.append(matched_news)

            except Exception as e:
                self.logger.error(f"Помилка при фільтрації новини '{news.get('title', 'unknown')}': {e}")

        self.logger.info(f"Відфільтровано {len(filtered_news)} новин з {len(news_data)} за ключовими словами")
        return filtered_news

    def detect_major_events(self, news_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Виявляє важливі події на основі аналізу новин.

        Args:
            news_data: Список словників з даними новин

        Returns:
            List[Dict[str, Any]]: Список виявлених важливих подій
        """
        self.logger.info(f"Початок аналізу {len(news_data)} новин для виявлення важливих подій")

        major_events = []

        # Аналіз новин
        for news in news_data:
            try:
                # Текст для аналізу
                text_to_analyze = self._get_text_to_analyze(news)

                # Перевірка по категоріях
                event_categories = set()
                matched_keywords = {}

                for category, patterns in self.category_patterns.items():
                    for pattern in patterns:
                        if pattern.search(text_to_analyze):
                            event_categories.add(category)

                            # Збереження ключових слів, що співпали
                            if category not in matched_keywords:
                                matched_keywords[category] = []
                            keyword = pattern.pattern.replace(r'\b', '')
                            matched_keywords[category].append(re.escape(keyword))

                # Визначення важливості події
                importance_level = len(event_categories)

                # Додаткові фактори для визначення важливості:
                # 1. Перевірка наявності назв великих компаній/проектів
                entity_matches = []
                for entity in self.MAJOR_ENTITIES:
                    if re.search(rf'\b{re.escape(entity)}\b', text_to_analyze, re.IGNORECASE):
                        entity_matches.append(entity)
                        importance_level += 0.5  # Додаємо ваги до важливості

                # 2. Перевірка наявності цифр (суми грошей, відсотки тощо)
                if re.search(r'\$\d+(?:[,.]\d+)?(?:\s*(?:million|billion|m|b|млн|млрд))?|\d+%', text_to_analyze,
                             re.IGNORECASE):
                    importance_level += 1  # Наявність фінансових даних підвищує важливість

                # Якщо знайдено хоча б одну категорію або важливість висока - це важлива подія
                if event_categories or importance_level >= 2:
                    event_data = {
                        'title': news.get('title', ''),
                        'summary': news.get('summary', ''),
                        'source': news.get('source', ''),
                        'link': news.get('link', ''),
                        'published_at': news.get('published_at', datetime.now()),
                        'categories': list(event_categories),
                        'matched_keywords': matched_keywords,
                        'major_entities': entity_matches,
                        'importance_level': importance_level,
                        'original_news': news
                    }
                    major_events.append(event_data)

            except Exception as e:
                self.logger.error(f"Помилка при аналізі новини '{news.get('title', 'unknown')}' на важливі події: {e}")

        # Сортування за важливістю (в порядку спадання)
        major_events.sort(key=lambda x: x['importance_level'], reverse=True)

        self.logger.info(f"Виявлено {len(major_events)} важливих подій")
        return major_events

    def get_trending_topics(self, news_data: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Виявляє трендові теми на основі частоти слів у новинах.

        Args:
            news_data: Список словників з даними новин
            top_n: Кількість трендових тем для виведення

        Returns:
            List[Dict[str, Any]]: Список трендових тем
        """
        self.logger.info(f"Аналіз трендів серед {len(news_data)} новин")

        # Словник для підрахунку частоти ключових слів
        word_frequency = {}

        for news in news_data:
            # Текст для аналізу
            text = self._get_text_to_analyze(news)

            # Нормалізація тексту: нижній регістр і видалення пунктуації
            text = re.sub(r'[^\w\s]', '', text.lower())

            # Розбиття на слова
            words = text.split()

            # Підрахунок частоти слів (крім поширених)
            for word in words:
                if len(word) > 3 and word not in self.COMMON_WORDS:
                    word_frequency[word] = word_frequency.get(word, 0) + 1

        # Сортування за частотою
        sorted_words = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)

        # Формування результату
        trends = []
        for word, frequency in sorted_words[:top_n]:
            trends.append({
                'topic': word,
                'frequency': frequency,
                'weight': frequency / len(news_data)
            })

        self.logger.info(f"Знайдено {len(trends)} трендових тем")
        return trends

    def correlate_with_market(self, news_data: List[Dict[str, Any]], market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Аналізує кореляцію між настроями в новинах та ринковими даними.

        Args:
            news_data: Список словників з даними новин
            market_data: DataFrame з ринковими даними

        Returns:
            Dict[str, Any]: Результати аналізу кореляції
        """
        self.logger.info("Початок аналізу кореляції новин з ринком")

        if not news_data or market_data.empty:
            self.logger.warning("Недостатньо даних для аналізу кореляції")
            return {'correlation': 0, 'significance': 0, 'valid': False}

        try:
            # Створюємо DataFrame з даними настроїв по датам
            sentiment_data = []

            for news in news_data:
                if 'sentiment' in news and 'published_at' in news:
                    date = news['published_at'].date()
                    score = news['sentiment'].get('score', 0)
                    sentiment_data.append({
                        'date': date,
                        'sentiment_score': score
                    })

            if not sentiment_data:
                self.logger.warning("Відсутні дані про настрої для аналізу")
                return {'correlation': 0, 'significance': 0, 'valid': False}

            sentiment_df = pd.DataFrame(sentiment_data)

            # Агрегація за датою (середній настрій за день)
            daily_sentiment = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()

            # Підготовка ринкових даних
            market_df = market_data.copy()
            if 'date' not in market_df.columns:
                market_df['date'] = pd.to_datetime(market_df.index).date

            # Злиття даних по даті
            merged_data = pd.merge(daily_sentiment, market_df, on='date', how='inner')

            if len(merged_data) < 3:  # Мінімум для розрахунку кореляції
                self.logger.warning("Недостатньо даних для розрахунку кореляції")
                return {'correlation': 0, 'significance': 0, 'valid': False}

            # Розрахунок кореляції Пірсона з ціною
            price_column = next((col for col in merged_data.columns if 'price' in col.lower()), 'close')
            correlation = merged_data['sentiment_score'].corr(merged_data[price_column])

            # Розрахунок p-value для визначення статистичної значущості
            correlation_coefficient, correlation_significance = stats.pearsonr(
                merged_data['sentiment_score'],
                merged_data[price_column]
            )

            result = {
                'correlation': correlation,
                'significance': correlation_significance,
                'sample_size': len(merged_data),
                'valid': True,
                'period_start': merged_data['date'].min(),
                'period_end': merged_data['date'].max()
            }

            self.logger.info(f"Розрахована кореляція: {correlation:.4f} (p={correlation_significance:.4f})")
            return result

        except Exception as e:
            self.logger.error(f"Помилка при аналізі кореляції: {e}")
            return {'correlation': 0, 'significance': 0, 'valid': False, 'error': str(e)}