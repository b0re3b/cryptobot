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

-- Таблиця для зберігання книги ордерів BTC
CREATE TABLE IF NOT EXISTS btc_orderbook (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    last_update_id BIGINT NOT NULL,
    type TEXT NOT NULL, -- 'bid' або 'ask'
    price NUMERIC NOT NULL,
    quantity NUMERIC NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Індекс для швидкого пошуку в книзі ордерів BTC
CREATE INDEX IF NOT EXISTS idx_btc_orderbook_time ON btc_orderbook(timestamp);

-- Таблиця для зберігання книги ордерів ETH
CREATE TABLE IF NOT EXISTS eth_orderbook (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    last_update_id BIGINT NOT NULL,
    type TEXT NOT NULL, -- 'bid' або 'ask'
    price NUMERIC NOT NULL,
    quantity NUMERIC NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Індекс для швидкого пошуку в книзі ордерів ETH
CREATE INDEX IF NOT EXISTS idx_eth_orderbook_time ON eth_orderbook(timestamp);

-- Таблиця для зберігання книги ордерів SOL
CREATE TABLE IF NOT EXISTS sol_orderbook (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    last_update_id BIGINT NOT NULL,
    type TEXT NOT NULL, -- 'bid' або 'ask'
    price NUMERIC NOT NULL,
    quantity NUMERIC NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Індекс для швидкого пошуку в книзі ордерів SOL
CREATE INDEX IF NOT EXISTS idx_sol_orderbook_time ON sol_orderbook(timestamp);

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

-- Таблиця для оброблених книг ордерів BTC
CREATE TABLE IF NOT EXISTS btc_orderbook_processed (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,

    spread NUMERIC, -- різниця між найвищим bid і найнижчим ask
    imbalance NUMERIC, -- (bid_volume - ask_volume) / (bid_volume + ask_volume)
    bid_volume NUMERIC,
    ask_volume NUMERIC,
    average_bid_price NUMERIC,
    average_ask_price NUMERIC,

    volatility_estimate NUMERIC, -- можливо, з minutely рівня

    is_anomaly BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (timestamp)
);

CREATE INDEX IF NOT EXISTS idx_btc_orderbook_processed_time ON btc_orderbook_processed(timestamp);

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

-- Таблиця для оброблених книг ордерів ETH
CREATE TABLE IF NOT EXISTS eth_orderbook_processed (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,

    spread NUMERIC, -- різниця між найвищим bid і найнижчим ask
    imbalance NUMERIC, -- (bid_volume - ask_volume) / (bid_volume + ask_volume)
    bid_volume NUMERIC,
    ask_volume NUMERIC,
    average_bid_price NUMERIC,
    average_ask_price NUMERIC,

    volatility_estimate NUMERIC, -- можливо, з minutely рівня

    is_anomaly BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (timestamp)
);

CREATE INDEX IF NOT EXISTS idx_eth_orderbook_processed_time ON eth_orderbook_processed(timestamp);

-- Таблиця для оброблених книг ордерів SOL
CREATE TABLE IF NOT EXISTS sol_orderbook_processed (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,

    spread NUMERIC, -- різниця між найвищим bid і найнижчим ask
    imbalance NUMERIC, -- (bid_volume - ask_volume) / (bid_volume + ask_volume)
    bid_volume NUMERIC,
    ask_volume NUMERIC,
    average_bid_price NUMERIC,
    average_ask_price NUMERIC,

    volatility_estimate NUMERIC, -- можливо, з minutely рівня

    is_anomaly BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (timestamp)
);

CREATE INDEX IF NOT EXISTS idx_sol_orderbook_processed_time ON sol_orderbook_processed(timestamp);

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