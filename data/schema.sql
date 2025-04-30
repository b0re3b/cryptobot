-- Таблиці для зберігання свічок (курсів) BTC
CREATE TABLE IF NOT EXISTS btc_klines (
    id SERIAL PRIMARY KEY,
    interval TEXT NOT NULL,
    open_time TIMESTAMP NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,
    close_time TIMESTAMP NOT NULL,
    quote_asset_volume NUMERIC NOT NULL,
    number_of_trades INTEGER NOT NULL,
    taker_buy_base_volume NUMERIC NOT NULL,
    taker_buy_quote_volume NUMERIC NOT NULL,
    is_closed BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (interval, open_time)
);

-- Індекс для швидкого пошуку свічок BTC за часом
CREATE INDEX IF NOT EXISTS idx_btc_klines_time ON btc_klines(interval, open_time);

-- Таблиці для зберігання свічок (курсів) ETH
CREATE TABLE IF NOT EXISTS eth_klines (
    id SERIAL PRIMARY KEY,
    interval TEXT NOT NULL,
    open_time TIMESTAMP NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,
    close_time TIMESTAMP NOT NULL,
    quote_asset_volume NUMERIC NOT NULL,
    number_of_trades INTEGER NOT NULL,
    taker_buy_base_volume NUMERIC NOT NULL,
    taker_buy_quote_volume NUMERIC NOT NULL,
    is_closed BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (interval, open_time)
);

-- Індекс для швидкого пошуку свічок ETH за часом
CREATE INDEX IF NOT EXISTS idx_eth_klines_time ON eth_klines(interval, open_time);

-- Таблиці для зберігання свічок (курсів) SOL
CREATE TABLE IF NOT EXISTS sol_klines (
    id SERIAL PRIMARY KEY,
    interval TEXT NOT NULL,
    open_time TIMESTAMP NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,
    close_time TIMESTAMP NOT NULL,
    quote_asset_volume NUMERIC NOT NULL,
    number_of_trades INTEGER NOT NULL,
    taker_buy_base_volume NUMERIC NOT NULL,
    taker_buy_quote_volume NUMERIC NOT NULL,
    is_closed BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (interval, open_time)
);

-- Індекс для швидкого пошуку свічок SOL за часом
CREATE INDEX IF NOT EXISTS idx_sol_klines_time ON sol_klines(interval, open_time);


-- Таблиця для логування подій
CREATE TABLE IF NOT EXISTS logs (
    id SERIAL PRIMARY KEY,
    log_level TEXT NOT NULL, -- 'INFO', 'WARNING', 'ERROR', 'DEBUG'
    message TEXT NOT NULL,
    component TEXT NOT NULL, -- 'BinanceClient', 'WebSocket', etc.
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Таблиця для оброблених свічок BTC
CREATE TABLE IF NOT EXISTS btc_klines_processed (
    id SERIAL PRIMARY KEY,
    interval TEXT NOT NULL,
    open_time TIMESTAMP NOT NULL,

    -- cleaned and normalized prices
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,

    -- engineered features
    price_zscore NUMERIC,
    volume_zscore NUMERIC,
    volatility NUMERIC,
    trend NUMERIC,

    -- time features
    hour INTEGER,
    day_of_week INTEGER,
    is_weekend BOOLEAN,
    session TEXT, -- 'Asia', 'Europe', 'US', etc.

    -- flags
    is_anomaly BOOLEAN DEFAULT FALSE,
    has_missing BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (interval, open_time)
);

CREATE INDEX IF NOT EXISTS idx_btc_klines_processed_time ON btc_klines_processed(interval, open_time);

-- Таблиця для профілю об'єму BTC
CREATE TABLE IF NOT EXISTS btc_volume_profile (
    id SERIAL PRIMARY KEY,
    interval TEXT NOT NULL,
    time_bucket TIMESTAMP NOT NULL, -- наприклад, кожна година або день
    price_bin_start NUMERIC NOT NULL,
    price_bin_end NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (interval, time_bucket, price_bin_start)
);

CREATE INDEX IF NOT EXISTS idx_btc_volume_profile ON btc_volume_profile(interval, time_bucket);

-- Таблиця для логування обробки даних
CREATE TABLE IF NOT EXISTS data_processing_log (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    data_type TEXT NOT NULL, -- 'klines', 'orderbook', 'volume_profile'
    interval TEXT,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    status TEXT NOT NULL, -- 'success', 'failed'
    steps TEXT, -- JSON список застосованих кроків: ["clean", "normalize", "fill", ...]
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);



-- Таблиця для профілю об'єму ETH
CREATE TABLE IF NOT EXISTS eth_volume_profile (
    id SERIAL PRIMARY KEY,
    interval TEXT NOT NULL,
    time_bucket TIMESTAMP NOT NULL, -- наприклад, кожна година або день
    price_bin_start NUMERIC NOT NULL,
    price_bin_end NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (interval, time_bucket, price_bin_start)
);

CREATE INDEX IF NOT EXISTS idx_eth_volume_profile ON eth_volume_profile(interval, time_bucket);

-- Таблиця для профілю об'єму SOL
CREATE TABLE IF NOT EXISTS sol_volume_profile (
    id SERIAL PRIMARY KEY,
    interval TEXT NOT NULL,
    time_bucket TIMESTAMP NOT NULL, -- наприклад, кожна година або день
    price_bin_start NUMERIC NOT NULL,
    price_bin_end NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (interval, time_bucket, price_bin_start)
);

CREATE INDEX IF NOT EXISTS idx_sol_volume_profile ON sol_volume_profile(interval, time_bucket);

-- Таблиця для оброблених свічок ETH
CREATE TABLE IF NOT EXISTS eth_klines_processed (
    id SERIAL PRIMARY KEY,
    interval TEXT NOT NULL,
    open_time TIMESTAMP NOT NULL,

    -- cleaned and normalized prices
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,

    -- engineered features
    price_zscore NUMERIC,
    volume_zscore NUMERIC,
    volatility NUMERIC,
    trend NUMERIC,

    -- time features
    hour INTEGER,
    day_of_week INTEGER,
    is_weekend BOOLEAN,
    session TEXT, -- 'Asia', 'Europe', 'US', etc.

    -- flags
    is_anomaly BOOLEAN DEFAULT FALSE,
    has_missing BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (interval, open_time)
);

CREATE INDEX IF NOT EXISTS idx_eth_klines_processed_time ON eth_klines_processed(interval, open_time);

-- Таблиця для оброблених свічок SOL
CREATE TABLE IF NOT EXISTS sol_klines_processed (
    id SERIAL PRIMARY KEY,
    interval TEXT NOT NULL,
    open_time TIMESTAMP NOT NULL,

    -- cleaned and normalized prices
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,

    -- engineered features
    price_zscore NUMERIC,
    volume_zscore NUMERIC,
    volatility NUMERIC,
    trend NUMERIC,

    -- time features
    hour INTEGER,
    day_of_week INTEGER,
    is_weekend BOOLEAN,
    session TEXT, -- 'Asia', 'Europe', 'US', etc.

    -- flags
    is_anomaly BOOLEAN DEFAULT FALSE,
    has_missing BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (interval, open_time)
);

CREATE INDEX IF NOT EXISTS idx_sol_klines_processed_time ON sol_klines_processed(interval, open_time);
-- Таблиця для зберігання сирих твітів
CREATE TABLE IF NOT EXISTS tweets_raw (
    id SERIAL PRIMARY KEY,
    tweet_id BIGINT UNIQUE NOT NULL,
    author_id TEXT NOT NULL,
    author_username TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    likes_count INTEGER NOT NULL,
    retweets_count INTEGER NOT NULL,
    quotes_count INTEGER,
    replies_count INTEGER,
    language TEXT NOT NULL,
    hashtags TEXT[], -- масив хештегів
    mentioned_cryptos TEXT[],  -- масив згаданих криптовалют
    tweet_url TEXT,
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Індекс для швидкого пошуку твітів за часом створення
CREATE INDEX IF NOT EXISTS idx_tweets_raw_created_at ON tweets_raw(created_at);
-- Індекс для швидкого пошуку за автором
CREATE INDEX IF NOT EXISTS idx_tweets_raw_author ON tweets_raw(author_username);
-- Індекс для швидкого пошуку за згаданими криптовалютами
CREATE INDEX IF NOT EXISTS idx_tweets_raw_cryptos ON tweets_raw USING GIN(mentioned_cryptos);

-- Таблиця для зберігання результатів аналізу настроїв твітів
CREATE TABLE IF NOT EXISTS tweet_sentiments (
    id SERIAL PRIMARY KEY,
    tweet_id BIGINT NOT NULL REFERENCES tweets_raw(tweet_id),
    sentiment TEXT NOT NULL, -- 'positive', 'negative', 'neutral'
    sentiment_score NUMERIC NOT NULL, -- числове значення настрою
    confidence NUMERIC NOT NULL, -- впевненість моделі
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_used TEXT NOT NULL, -- назва використаної моделі
    UNIQUE (tweet_id)
);

-- Індекс для швидкого пошуку аналізу настроїв за твітом
CREATE INDEX IF NOT EXISTS idx_tweet_sentiments_tweet_id ON tweet_sentiments(tweet_id);

-- Таблиця для кешування запитів Twitter
CREATE TABLE IF NOT EXISTS twitter_query_cache (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    search_params JSONB NOT NULL, -- параметри пошуку в форматі JSON
    cache_expires_at TIMESTAMP NOT NULL,
    results_count INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (query, search_params)
);

-- Індекс для швидкого пошуку в кеші за запитом
CREATE INDEX IF NOT EXISTS idx_twitter_query_cache_query ON twitter_query_cache(query);
-- Індекс для визначення застарілих кешів
CREATE INDEX IF NOT EXISTS idx_twitter_query_cache_expires ON twitter_query_cache(cache_expires_at);

-- Таблиця для зберігання інформації про крипто-інфлюенсерів
CREATE TABLE IF NOT EXISTS crypto_influencers (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    display_name TEXT,
    description TEXT,
    followers_count INTEGER,
    following_count INTEGER,
    tweet_count INTEGER,
    verified BOOLEAN,
    influence_score NUMERIC, -- обчислювана метрика впливовості
    crypto_topics TEXT[], -- основні криптовалютні теми
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Індекс для швидкого пошуку інфлюенсерів за іменем користувача
CREATE INDEX IF NOT EXISTS idx_crypto_influencers_username ON crypto_influencers(username);

-- Таблиця для відстеження активності інфлюенсерів
CREATE TABLE IF NOT EXISTS influencer_activity (
    id SERIAL PRIMARY KEY,
    influencer_id INTEGER NOT NULL REFERENCES crypto_influencers(id),
    tweet_id BIGINT NOT NULL REFERENCES tweets_raw(tweet_id),
    impact_score NUMERIC, -- оцінка впливу конкретного твіту
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (influencer_id, tweet_id)
);

-- Індекс для швидкого пошуку активності за інфлюенсером
CREATE INDEX IF NOT EXISTS idx_influencer_activity_influencer ON influencer_activity(influencer_id);

-- Таблиця для виявлених криптовалютних подій
CREATE TABLE IF NOT EXISTS crypto_events (
    id SERIAL PRIMARY KEY,
    event_type TEXT NOT NULL, -- 'pump', 'dump', 'announcement', 'regulation', etc.
    crypto_symbol TEXT NOT NULL, -- 'BTC', 'ETH', etc.
    description TEXT NOT NULL,
    confidence_score NUMERIC NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    related_tweets BIGINT[], -- масив ID твітів, пов'язаних з подією
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Індекс для швидкого пошуку подій за криптовалютою
CREATE INDEX IF NOT EXISTS idx_crypto_events_symbol ON crypto_events(crypto_symbol);
-- Індекс для швидкого пошуку подій за часом
CREATE INDEX IF NOT EXISTS idx_crypto_events_time ON crypto_events(start_time);

-- Таблиця для агрегованого аналізу настроїв за часовими інтервалами та криптовалютами
CREATE TABLE IF NOT EXISTS sentiment_time_series (
    id SERIAL PRIMARY KEY,
    crypto_symbol TEXT NOT NULL,
    time_bucket TIMESTAMP NOT NULL,
    interval TEXT NOT NULL, -- 'hour', 'day', 'week'
    positive_count INTEGER NOT NULL,
    negative_count INTEGER NOT NULL,
    neutral_count INTEGER NOT NULL,
    average_sentiment NUMERIC NOT NULL,
    sentiment_volatility NUMERIC, -- волатильність настрою
    tweet_volume INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (crypto_symbol, time_bucket, interval)
);

-- Індекс для швидкого пошуку часових рядів настроїв за криптовалютою та часом
CREATE INDEX IF NOT EXISTS idx_sentiment_time_series ON sentiment_time_series(crypto_symbol, time_bucket);

-- Таблиця для статистики помилок скрапінгу
CREATE TABLE IF NOT EXISTS scraping_errors (
    id SERIAL PRIMARY KEY,
    error_type TEXT NOT NULL, -- 'rate_limit', 'connection', 'parsing', etc.
    error_message TEXT NOT NULL,
    query TEXT,
    retry_count INTEGER,
    occurred_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Індекс для аналізу частоти помилок за типом
CREATE INDEX IF NOT EXISTS idx_scraping_errors_type ON scraping_errors(error_type);
-- Таблиця для зберігання джерел новин
CREATE TABLE IF NOT EXISTS news_sources (
    source_id SERIAL PRIMARY KEY,
    source_name VARCHAR(100) NOT NULL UNIQUE,
    base_url VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Таблиця для категорій новин
CREATE TABLE IF NOT EXISTS news_categories (
    category_id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES news_sources(source_id),
    category_name VARCHAR(100) NOT NULL,
    category_url_path VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(source_id, category_name)
);

-- Таблиця для зберігання новинних статей
CREATE TABLE IF NOT EXISTS news_articles (
    article_id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    summary TEXT,
    content TEXT,
    link VARCHAR(512) NOT NULL,
    source_id INTEGER REFERENCES news_sources(source_id),
    category_id INTEGER REFERENCES news_categories(category_id),
    published_at TIMESTAMP WITH TIME ZONE,
    scraped_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(link)
);

-- Індекси для новинних статей
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON news_articles(published_at);
CREATE INDEX IF NOT EXISTS idx_articles_source_id ON news_articles(source_id);

-- Таблиця для аналізу настроїв новин
CREATE TABLE IF NOT EXISTS news_sentiment_analysis (
    sentiment_id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES news_articles(article_id),
    sentiment_score NUMERIC(5,2),  -- Від -1.0 до 1.0
    sentiment_magnitude NUMERIC(5,2),
    sentiment_label VARCHAR(20),  -- 'positive', 'negative', 'neutral'
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(article_id)
);

-- Таблиця для згаданих криптовалют у новинах (зв'язок багато-до-багатьох)
CREATE TABLE IF NOT EXISTS article_mentioned_coins (
    mention_id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES news_articles(article_id),
    crypto_symbol TEXT NOT NULL,  -- Використовуємо існуючу символіку криптовалют
    mention_count INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(article_id, crypto_symbol)
);

-- Індекс для швидкого пошуку згадок криптовалют
CREATE INDEX IF NOT EXISTS idx_article_mentioned_coins ON article_mentioned_coins(crypto_symbol);

-- Таблиця для популярних тем у новинах
CREATE TABLE IF NOT EXISTS trending_news_topics (
    topic_id SERIAL PRIMARY KEY,
    topic_name VARCHAR(255) NOT NULL,
    start_date TIMESTAMP WITH TIME ZONE,
    end_date TIMESTAMP WITH TIME ZONE,
    importance_score NUMERIC(5,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Таблиця для зв'язку статей з темами (багато-до-багатьох)
CREATE TABLE IF NOT EXISTS article_topics (
    article_topic_id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES news_articles(article_id),
    topic_id INTEGER REFERENCES trending_news_topics(topic_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(article_id, topic_id)
);

-- Таблиця для кореляцій між новинами та ринковими даними
CREATE TABLE IF NOT EXISTS news_market_correlations (
    correlation_id SERIAL PRIMARY KEY,
    topic_id INTEGER REFERENCES trending_news_topics(topic_id),
    crypto_symbol TEXT NOT NULL,  -- Використовуємо існуючу символіку криптовалют
    time_period VARCHAR(50),  -- '1h', '4h', '1d'
    correlation_coefficient NUMERIC(5,4),  -- Від -1 до 1
    p_value NUMERIC(10,9),
    start_date TIMESTAMP WITH TIME ZONE,
    end_date TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Таблиця для відстеження важливих подій, виявлених із новин
CREATE TABLE IF NOT EXISTS news_detected_events (
    event_id SERIAL PRIMARY KEY,
    event_title TEXT NOT NULL,
    event_description TEXT,
    crypto_symbols TEXT[],  -- Масив пов'язаних криптовалют
    source_articles INTEGER[],  -- Масив ID статей-джерел події
    confidence_score NUMERIC(3,1),  -- 0-10 шкала
    detected_at TIMESTAMP WITH TIME ZONE,
    expected_impact TEXT,  -- 'positive', 'negative', 'neutral'
    event_category TEXT,  -- 'regulatory', 'technical', 'adoption', тощо
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Індекс для пошуку подій за криптовалютами
CREATE INDEX IF NOT EXISTS idx_news_detected_events_symbols ON news_detected_events USING GIN(crypto_symbols);

-- Таблиця для часових рядів настроїв у новинах
CREATE TABLE IF NOT EXISTS news_sentiment_time_series (
    id SERIAL PRIMARY KEY,
    crypto_symbol TEXT NOT NULL,
    time_bucket TIMESTAMP WITH TIME ZONE NOT NULL,
    interval TEXT NOT NULL,  -- 'hour', 'day', 'week'
    positive_count INTEGER NOT NULL,
    negative_count INTEGER NOT NULL,
    neutral_count INTEGER NOT NULL,
    average_sentiment NUMERIC NOT NULL,
    news_volume INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (crypto_symbol, time_bucket, interval)
);

-- Індекс для швидкого пошуку часових рядів настроїв за криптовалютою та часом
CREATE INDEX IF NOT EXISTS idx_news_sentiment_time_series ON news_sentiment_time_series(crypto_symbol, time_bucket);

-- Таблиця для логування роботи CryptoNewsScraper
CREATE TABLE IF NOT EXISTS news_scraping_log (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES news_sources(source_id),
    category_id INTEGER REFERENCES news_categories(category_id),
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    articles_found INTEGER NOT NULL,
    articles_processed INTEGER NOT NULL,
    status TEXT NOT NULL,  -- 'success', 'partial', 'failed'
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Функція для оновлення часових позначок
CREATE OR REPLACE FUNCTION update_news_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Тригери для оновлення позначок часу
CREATE TRIGGER update_news_sources_timestamp
BEFORE UPDATE ON news_sources
FOR EACH ROW EXECUTE FUNCTION update_news_timestamp();

CREATE TRIGGER update_news_categories_timestamp
BEFORE UPDATE ON news_categories
FOR EACH ROW EXECUTE FUNCTION update_news_timestamp();

CREATE TRIGGER update_trending_news_topics_timestamp
BEFORE UPDATE ON trending_news_topics
FOR EACH ROW EXECUTE FUNCTION update_news_timestamp();

-- Представлення для новин з настроями та згаданими криптовалютами
CREATE OR REPLACE VIEW news_analysis_view AS
SELECT
    na.article_id,
    na.title,
    na.summary,
    na.link,
    ns.source_name,
    nc.category_name,
    na.published_at,
    nsa.sentiment_score,
    nsa.sentiment_label,
    array_agg(DISTINCT amc.crypto_symbol) as mentioned_coins
FROM news_articles na
JOIN news_sources ns ON na.source_id = ns.source_id
JOIN news_categories nc ON na.category_id = nc.category_id
LEFT JOIN news_sentiment_analysis nsa ON na.article_id = nsa.article_id
LEFT JOIN article_mentioned_coins amc ON na.article_id = amc.article_id
GROUP BY
    na.article_id,
    na.title,
    na.summary,
    na.link,
    ns.source_name,
    nc.category_name,
    na.published_at,
    nsa.sentiment_score,
    nsa.sentiment_label;

-- Представлення для інтегрованого аналізу настроїв новин і твітів
CREATE OR REPLACE VIEW integrated_sentiment_view AS
SELECT
    crypto_symbol,
    time_bucket,
    interval,
    'news' as source_type,
    positive_count,
    negative_count,
    neutral_count,
    average_sentiment,
    news_volume as content_volume
FROM news_sentiment_time_series
UNION ALL
SELECT
    crypto_symbol,
    time_bucket,
    interval,
    'twitter' as source_type,
    positive_count,
    negative_count,
    neutral_count,
    average_sentiment,
    tweet_volume as content_volume
FROM sentiment_time_series;

-- Початкове заповнення таблиці news_sources
INSERT INTO news_sources (source_name, base_url) VALUES
('coindesk', 'https://www.coindesk.com'),
('cointelegraph', 'https://cointelegraph.com'),
('decrypt', 'https://decrypt.co'),
('cryptoslate', 'https://cryptoslate.com'),
('theblock', 'https://www.theblock.co'),
('cryptopanic', 'https://cryptopanic.com'),
('coinmarketcal', 'https://coinmarketcal.com'),
('feedly', 'https://feedly.com'),
('newsnow', 'https://www.newsnow.co.uk'),
('reddit', 'https://www.reddit.com')
ON CONFLICT (source_name) DO NOTHING;

-- Початкове заповнення таблиці news_categories
INSERT INTO news_categories (source_id, category_name, category_url_path) VALUES
-- CoinDesk категорії
((SELECT source_id FROM news_sources WHERE source_name = 'coindesk'), 'markets', 'markets'),
((SELECT source_id FROM news_sources WHERE source_name = 'coindesk'), 'business', 'business'),
((SELECT source_id FROM news_sources WHERE source_name = 'coindesk'), 'policy', 'policy'),
((SELECT source_id FROM news_sources WHERE source_name = 'coindesk'), 'tech', 'tech'),
-- Cointelegraph категорії
((SELECT source_id FROM news_sources WHERE source_name = 'cointelegraph'), 'news', 'news'),
((SELECT source_id FROM news_sources WHERE source_name = 'cointelegraph'), 'markets', 'markets'),
((SELECT source_id FROM news_sources WHERE source_name = 'cointelegraph'), 'features', 'features'),
((SELECT source_id FROM news_sources WHERE source_name = 'cointelegraph'), 'analysis', 'analysis'),
-- Decrypt категорії
((SELECT source_id FROM news_sources WHERE source_name = 'decrypt'), 'news', 'news'),
((SELECT source_id FROM news_sources WHERE source_name = 'decrypt'), 'analysis', 'analysis'),
((SELECT source_id FROM news_sources WHERE source_name = 'decrypt'), 'features', 'features'),
((SELECT source_id FROM news_sources WHERE source_name = 'decrypt'), 'learn', 'learn'),
-- CryptoSlate категорії
((SELECT source_id FROM news_sources WHERE source_name = 'cryptoslate'), 'news', 'news'),
((SELECT source_id FROM news_sources WHERE source_name = 'cryptoslate'), 'bitcoin', 'bitcoin'),
((SELECT source_id FROM news_sources WHERE source_name = 'cryptoslate'), 'ethereum', 'ethereum'),
((SELECT source_id FROM news_sources WHERE source_name = 'cryptoslate'), 'defi', 'defi'),
-- The Block категорії
((SELECT source_id FROM news_sources WHERE source_name = 'theblock'), 'latest', 'latest'),
((SELECT source_id FROM news_sources WHERE source_name = 'theblock'), 'policy', 'policy'),
((SELECT source_id FROM news_sources WHERE source_name = 'theblock'), 'business', 'business'),
((SELECT source_id FROM news_sources WHERE source_name = 'theblock'), 'markets', 'markets')
ON CONFLICT (source_id, category_name) DO NOTHING;
-- Схема для збереження даних про часові ряди та результати моделювання

-- Таблиця для зберігання метаданих моделей часових рядів
CREATE TABLE IF NOT EXISTS time_series_models (
    model_id SERIAL PRIMARY KEY,
    model_key VARCHAR(100) NOT NULL UNIQUE,
    symbol VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'arima', 'sarima', etc.
    interval VARCHAR(10) NOT NULL, -- '1m', '5m', '15m', '1h', '4h', '1d'
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE
);

-- Таблиця для зберігання параметрів моделей
CREATE TABLE IF NOT EXISTS model_parameters (
    parameter_id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES time_series_models(model_id) ON DELETE CASCADE,
    param_name VARCHAR(50) NOT NULL,
    param_value JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_id, param_name)
);

-- Таблиця для зберігання метрик ефективності моделей
CREATE TABLE IF NOT EXISTS model_metrics (
    metric_id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES time_series_models(model_id) ON DELETE CASCADE,
    metric_name VARCHAR(50) NOT NULL,
    metric_value FLOAT NOT NULL,
    test_date TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_id, metric_name, test_date)
);

-- Таблиця для зберігання прогнозів
CREATE TABLE IF NOT EXISTS model_forecasts (
    forecast_id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES time_series_models(model_id) ON DELETE CASCADE,
    forecast_date TIMESTAMP NOT NULL,
    forecast_value FLOAT NOT NULL,
    lower_bound FLOAT,
    upper_bound FLOAT,
    confidence_level FLOAT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_id, forecast_date)
);

-- Таблиця для зберігання повних результатів тренування моделі (серіалізована модель)
CREATE TABLE IF NOT EXISTS model_binary_data (
    binary_id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES time_series_models(model_id) ON DELETE CASCADE,
    model_binary BYTEA NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_id)
);

-- Таблиця для зберігання історії перетворень даних
CREATE TABLE IF NOT EXISTS data_transformations (
    transform_id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES time_series_models(model_id) ON DELETE CASCADE,
    transform_type VARCHAR(50) NOT NULL, -- 'log', 'diff', 'boxcox', etc.
    transform_params JSONB,
    transform_order INTEGER NOT NULL, -- порядок застосування трансформацій
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_id, transform_order)
);

-- Індекси для оптимізації запитів
CREATE INDEX IF NOT EXISTS idx_time_series_models_symbol ON time_series_models(symbol);
CREATE INDEX IF NOT EXISTS idx_time_series_models_model_type ON time_series_models(model_type);
CREATE INDEX IF NOT EXISTS idx_model_forecasts_date ON model_forecasts(forecast_date);
CREATE INDEX IF NOT EXISTS idx_model_forecasts_model_id_date ON model_forecasts(model_id, forecast_date);

-- Тригер для оновлення updated_at при зміні моделі
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_time_series_models_timestamp
BEFORE UPDATE ON time_series_models
FOR EACH ROW
EXECUTE FUNCTION update_modified_column();

-- Таблиця для збереження кореляційних матриць між криптовалютами
CREATE TABLE IF NOT EXISTS correlation_matrices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    correlation_type VARCHAR(20) NOT NULL, -- 'price', 'volume', 'returns', 'volatility'
    timeframe VARCHAR(10) NOT NULL, -- '1m', '5m', '15m', '1h', '4h', '1d', etc.
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    method VARCHAR(10) NOT NULL, -- 'pearson', 'kendall', 'spearman'
    matrix_json TEXT NOT NULL, -- JSON представлення кореляційної матриці
    symbols_list TEXT NOT NULL, -- JSON представлення списку символів
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(correlation_type, timeframe, start_time, end_time, method)
);

-- Індекс для швидкого пошуку кореляційних матриць
CREATE INDEX IF NOT EXISTS idx_correlation_matrices_lookup
ON correlation_matrices(correlation_type, timeframe, start_time, end_time, method);

-- Таблиця для збереження пар криптовалют з високою кореляцією
CREATE TABLE IF NOT EXISTS correlated_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol1 VARCHAR(20) NOT NULL,
    symbol2 VARCHAR(20) NOT NULL,
    correlation_value FLOAT NOT NULL,
    correlation_type VARCHAR(20) NOT NULL, -- 'price', 'volume', 'returns', 'volatility'
    timeframe VARCHAR(10) NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    method VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol1, symbol2, correlation_type, timeframe, start_time, end_time, method)
);

-- Індекс для швидкого пошуку корельованих пар за символом
CREATE INDEX IF NOT EXISTS idx_correlated_pairs_symbol1 ON correlated_pairs(symbol1, correlation_type);
CREATE INDEX IF NOT EXISTS idx_correlated_pairs_symbol2 ON correlated_pairs(symbol2, correlation_type);

-- Таблиця для збереження часових рядів кореляцій між парами криптовалют
CREATE TABLE IF NOT EXISTS correlation_time_series (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol1 VARCHAR(20) NOT NULL,
    symbol2 VARCHAR(20) NOT NULL,
    correlation_type VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    window_size INTEGER NOT NULL, -- розмір вікна для обчислення кореляції
    timestamp TIMESTAMP NOT NULL, -- час, для якого обчислена кореляція
    correlation_value FLOAT NOT NULL,
    method VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol1, symbol2, correlation_type, timeframe, window_size, timestamp, method)
);

-- Індекс для швидкого пошуку часових рядів кореляцій
CREATE INDEX IF NOT EXISTS idx_correlation_time_series_lookup
ON correlation_time_series(symbol1, symbol2, correlation_type, timeframe, window_size);

-- Таблиця для збереження кластерів криптовалют, які рухаються разом
CREATE TABLE IF NOT EXISTS market_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER NOT NULL,
    feature_type VARCHAR(20) NOT NULL, -- 'price', 'returns', 'volatility'
    timeframe VARCHAR(10) NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    symbols_list TEXT NOT NULL, -- JSON представлення списку символів у кластері
    clustering_method VARCHAR(30) NOT NULL, -- метод кластеризації
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(cluster_id, feature_type, timeframe, start_time, end_time, clustering_method)
);

-- Таблиця для збереження моментів зламу кореляцій між парами криптовалют
CREATE TABLE IF NOT EXISTS correlation_breakdowns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol1 VARCHAR(20) NOT NULL,
    symbol2 VARCHAR(20) NOT NULL,
    breakdown_time TIMESTAMP NOT NULL,
    correlation_before FLOAT NOT NULL,
    correlation_after FLOAT NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    window_size INTEGER NOT NULL,
    threshold FLOAT NOT NULL, -- поріг зміни кореляції для визначення зламу
    method VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol1, symbol2, breakdown_time, timeframe, window_size, method)
);

-- Таблиця для збереження бета-коефіцієнтів криптовалют відносно ринку
CREATE TABLE IF NOT EXISTS market_betas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) NOT NULL,
    market_symbol VARCHAR(20) NOT NULL, -- звичайно 'BTCUSDT'
    beta_value FLOAT NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, market_symbol, timeframe, start_time, end_time)
);

-- Таблиця для збереження часових рядів бета-коефіцієнтів
CREATE TABLE IF NOT EXISTS beta_time_series (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) NOT NULL,
    market_symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    beta_value FLOAT NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    window_size INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, market_symbol, timestamp, timeframe, window_size)
);

-- Таблиця для збереження кореляцій між секторами криптовалют
CREATE TABLE IF NOT EXISTS sector_correlations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sector1 VARCHAR(50) NOT NULL,
    sector2 VARCHAR(50) NOT NULL,
    correlation_value FLOAT NOT NULL,
    correlation_type VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    method VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(sector1, sector2, correlation_type, timeframe, start_time, end_time, method)
);

-- Таблиця для збереження провідних індикаторів
CREATE TABLE IF NOT EXISTS leading_indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_symbol VARCHAR(20) NOT NULL,
    indicator_symbol VARCHAR(20) NOT NULL,
    lag_period INTEGER NOT NULL,
    correlation_value FLOAT NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    method VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(target_symbol, indicator_symbol, lag_period, timeframe, start_time, end_time, method)
);

-- Таблиця для збереження кореляцій з зовнішніми активами
CREATE TABLE IF NOT EXISTS external_asset_correlations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    crypto_symbol VARCHAR(20) NOT NULL,
    external_asset VARCHAR(50) NOT NULL,
    correlation_value FLOAT NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    method VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(crypto_symbol, external_asset, timeframe, start_time, end_time, method)
);

-- Таблиця для збереження аналізу кореляцій у різних ринкових режимах
CREATE TABLE IF NOT EXISTS market_regime_correlations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    regime_name VARCHAR(50) NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    correlation_type VARCHAR(20) NOT NULL,
    matrix_json TEXT NOT NULL, -- JSON представлення кореляційної матриці
    symbols_list TEXT NOT NULL, -- JSON представлення списку символів
    method VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(regime_name, start_time, end_time, correlation_type, method)
);
CREATE TABLE halving_events (
    id SERIAL PRIMARY KEY,
    date TIMESTAMP NOT NULL,
    block_height INTEGER NOT NULL,
    block_reward_before NUMERIC(18,8) NOT NULL,
    block_reward_after NUMERIC(18,8) NOT NULL,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_halving_date ON halving_events(date);

CREATE TABLE market_cycles (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    cycle_start_date TIMESTAMP NOT NULL,
    cycle_end_date TIMESTAMP,
    peak_date TIMESTAMP,
    peak_price NUMERIC(18,8),
    trough_date TIMESTAMP,
    trough_price NUMERIC(18,8),
    cycle_type VARCHAR(50) NOT NULL,
    phase VARCHAR(30),
    duration_days INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_market_cycles_symbol ON market_cycles(symbol);
CREATE INDEX idx_market_cycles_dates ON market_cycles(cycle_start_date, cycle_end_date);
CREATE INDEX idx_market_cycles_type ON market_cycles(cycle_type);

CREATE TABLE cycle_patterns (
    id SERIAL PRIMARY KEY,
    pattern_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    window_size INTEGER NOT NULL,
    pattern_data JSONB NOT NULL,
    cluster_id INTEGER NOT NULL,
    occurrence_count INTEGER DEFAULT 0,
    average_duration NUMERIC(10,2),
    last_detected_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_cycle_patterns_symbol ON cycle_patterns(symbol);
CREATE INDEX idx_cycle_patterns_cluster ON cycle_patterns(cluster_id);

CREATE TABLE cycle_analysis_results (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    analysis_date TIMESTAMP NOT NULL,
    analysis_type VARCHAR(50) NOT NULL,
    results_json JSONB NOT NULL,
    summary TEXT,
    confidence_score NUMERIC(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_analysis_results_symbol ON cycle_analysis_results(symbol);
CREATE INDEX idx_analysis_results_date ON cycle_analysis_results(analysis_date);
CREATE INDEX idx_analysis_results_type ON cycle_analysis_results(analysis_type);

CREATE TABLE seasonality_patterns (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    period_days INTEGER NOT NULL,
    strength NUMERIC(5,2) NOT NULL,
    detection_date TIMESTAMP NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_seasonality_symbol ON seasonality_patterns(symbol);
CREATE INDEX idx_seasonality_period ON seasonality_patterns(period_days);

CREATE TABLE regime_changes (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    change_date TIMESTAMP NOT NULL,
    previous_regime VARCHAR(50),
    new_regime VARCHAR(50) NOT NULL,
    confidence_score NUMERIC(5,2),
    indicators_json JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_regime_changes_symbol ON regime_changes(symbol);
CREATE INDEX idx_regime_changes_date ON regime_changes(change_date);

CREATE TABLE cycle_predictions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    prediction_date TIMESTAMP NOT NULL,
    target_date TIMESTAMP NOT NULL,
    predicted_price NUMERIC(18,8),
    prediction_type VARCHAR(50) NOT NULL,
    confidence NUMERIC(5,2),
    methodology VARCHAR(100) NOT NULL,
    actual_price NUMERIC(18,8),
    accuracy_pct NUMERIC(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_predictions_symbol ON cycle_predictions(symbol);
CREATE INDEX idx_predictions_dates ON cycle_predictions(prediction_date, target_date);
CREATE INDEX idx_predictions_type ON cycle_predictions(prediction_type);

CREATE TABLE feature_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date TIMESTAMP NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    feature_value NUMERIC(18,8) NOT NULL,
    feature_category VARCHAR(50) NOT NULL, -- 'cycle', 'technical', 'fundamental', 'sentiment'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_feature_data_symbol ON feature_data(symbol);
CREATE INDEX idx_feature_data_date ON feature_data(date);
CREATE INDEX idx_feature_data_name ON feature_data(feature_name);
CREATE INDEX idx_feature_data_category ON feature_data(feature_category);

INSERT INTO halving_events (date, block_height, block_reward_before, block_reward_after, notes)
VALUES
    ('2012-11-28', 210000, 50, 25, 'First Bitcoin halving'),
    ('2016-07-09', 420000, 25, 12.5, 'Second Bitcoin halving'),
    ('2020-05-11', 630000, 12.5, 6.25, 'Third Bitcoin halving'),
    ('2024-04-20', 840000, 6.25, 3.125, 'Fourth Bitcoin halving (estimated date)');
