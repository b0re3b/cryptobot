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