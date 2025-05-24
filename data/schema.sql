-- Таблиці для зберігання свічок (курсів) BTC
CREATE TABLE IF NOT EXISTS btc_klines (
    id SERIAL PRIMARY KEY,
    timeframe TEXT NOT NULL,
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
    UNIQUE (timeframe, open_time)
);

-- Індекс для швидкого пошуку свічок BTC за часом
CREATE INDEX IF NOT EXISTS idx_btc_klines_time ON btc_klines(timeframe, open_time);

-- Таблиці для зберігання свічок (курсів) ETH
CREATE TABLE IF NOT EXISTS eth_klines (
    id SERIAL PRIMARY KEY,
    timeframe TEXT NOT NULL,
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
    UNIQUE (timeframe, open_time)
);

-- Індекс для швидкого пошуку свічок ETH за часом
CREATE INDEX IF NOT EXISTS idx_eth_klines_time ON eth_klines(timeframe, open_time);

-- Таблиці для зберігання свічок (курсів) SOL
CREATE TABLE IF NOT EXISTS sol_klines (
    id SERIAL PRIMARY KEY,
    timeframe TEXT NOT NULL,
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
    UNIQUE (timeframe, open_time)
);

-- Індекс для швидкого пошуку свічок SOL за часом
CREATE INDEX IF NOT EXISTS idx_sol_klines_time ON sol_klines(timeframe, open_time);


-- Таблиця для логування подій
CREATE TABLE IF NOT EXISTS logs (
    id SERIAL PRIMARY KEY,
    log_level TEXT NOT NULL, -- 'INFO', 'WARNING', 'ERROR', 'DEBUG'
    message TEXT NOT NULL,
    component TEXT NOT NULL, -- 'BinanceClient', 'WebSocket', etc.
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
/*
-- Таблиця для профілю об'єму BTC
CREATE TABLE IF NOT EXISTS btc_volume_profile (
    id SERIAL PRIMARY KEY,
    timeframe TEXT NOT NULL,
    time_bucket TIMESTAMP NOT NULL, -- наприклад, кожна година або день
    price_bin_start NUMERIC NOT NULL,
    price_bin_end NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (timeframe, time_bucket, price_bin_start)
);

CREATE INDEX IF NOT EXISTS idx_btc_volume_profile ON btc_volume_profile(timeframe, time_bucket);

-- Таблиця для профілю об'єму ETH
CREATE TABLE IF NOT EXISTS eth_volume_profile (
    id SERIAL PRIMARY KEY,
    timeframe TEXT NOT NULL,
    time_bucket TIMESTAMP NOT NULL, -- наприклад, кожна година або день
    price_bin_start NUMERIC NOT NULL,
    price_bin_end NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (timeframe, time_bucket, price_bin_start)
);

CREATE INDEX IF NOT EXISTS idx_eth_volume_profile ON eth_volume_profile(timeframe, time_bucket);

-- Таблиця для профілю об'єму SOL
CREATE TABLE IF NOT EXISTS sol_volume_profile (
    id SERIAL PRIMARY KEY,
    timeframe TEXT NOT NULL,
    time_bucket TIMESTAMP NOT NULL,
    price_bin_start NUMERIC NOT NULL,
    price_bin_end NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (timeframe, time_bucket, price_bin_start)
);*/

-- CREATE INDEX IF NOT EXISTS idx_sol_volume_profile ON sol_volume_profile(timeframe, time_bucket);


/*-- Таблиця для зберігання джерел новин
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
    score INTEGER,                -- Додано для Reddit та інших соціальних джерел
    upvote_ratio NUMERIC(3,2),    -- Додано для Reddit
    num_comments INTEGER,         -- Додано для Reddit
    UNIQUE(link)
);

-- Додатковий індекс для пошуку за часом публікації та джерелом
CREATE INDEX IF NOT EXISTS idx_articles_pub_source ON news_articles(published_at, source_id);


-- Таблиця для аналізу настроїв новин
-- Розширена таблиця для аналізу настроїв новин
CREATE TABLE IF NOT EXISTS news_sentiment_analysis (
    sentiment_id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES news_articles(article_id),
    sentiment_score NUMERIC(5,2),  -- Від -1.0 до 1.0
    positive_score NUMERIC(5,2),   -- Додано деталізацію компонентів
    negative_score NUMERIC(5,2),   -- Додано деталізацію компонентів
    neutral_score NUMERIC(5,2),    -- Додано деталізацію компонентів
    sentiment_magnitude NUMERIC(5,2),
    sentiment_label VARCHAR(20),  -- 'positive', 'negative', 'neutral'
    confidence NUMERIC(5,2),      -- Додано впевненість аналізу
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    model_version VARCHAR(50),    -- Додано версію моделі аналізу
    UNIQUE(article_id)
);

-- Таблиця для згаданих криптовалют у новинах (зв'язок багато-до-багатьох)
CREATE TABLE IF NOT EXISTS article_mentioned_coins (
    mention_id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES news_articles(article_id),
    symbol TEXT NOT NULL,  -- Використовуємо існуючу символіку криптовалют
    mention_count INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(article_id, symbol)
);

-- Індекс для швидкого пошуку згадок криптовалют
CREATE INDEX IF NOT EXISTS idx_article_mentioned_coins ON article_mentioned_coins(symbol);

-- Об'єднана таблиця для тем новин
CREATE TABLE IF NOT EXISTS news_topics (
    topic_id SERIAL PRIMARY KEY,
    topic_name VARCHAR(255) NOT NULL,
    is_trending BOOLEAN DEFAULT FALSE,
    importance_score NUMERIC(5,2),
    first_observed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_observed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Оптимізована таблиця зв'язку статей з темами
CREATE TABLE IF NOT EXISTS article_topics (
    article_topic_id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES news_articles(article_id),
    topic_id INTEGER REFERENCES news_topics(topic_id),
    weight NUMERIC(5,4) NOT NULL,  -- Додано вагу теми в статті
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(article_id, topic_id)
);

-- Індекс для швидкого пошуку тем
CREATE INDEX IF NOT EXISTS idx_article_topics_topic ON article_topics(topic_id);
CREATE INDEX IF NOT EXISTS idx_article_topics_article ON article_topics(article_id);

-- Спрощена таблиця для часових рядів настроїв
CREATE TABLE IF NOT EXISTS sentiment_time_series (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    timeframe VARCHAR(20) NOT NULL,  -- 'hour', 'day', 'week'
    sentiment_avg NUMERIC(5,2) NOT NULL,
    news_count INTEGER NOT NULL,
    mentions_count INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (symbol, start_time, timeframe)
);

-- Індекс для часових запитів
CREATE INDEX IF NOT EXISTS idx_sentiment_time_period ON sentiment_time_series(symbol, start_time);


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



-- Початкове заповнення таблиці news_sources
INSERT INTO news_sources (source_name, base_url) VALUES
('coindesk', 'https://www.coindesk.com'),
('cointelegraph', 'https://cointelegraph.com'),
('decrypt', 'https://decrypt.co'),
('cryptoslate', 'https://cryptoslate.com'),
('theblock', 'https://www.theblock.co'),
('cryptopanic', 'https://cryptopanic.com'),
('coinmarketcal', 'https://coinmarketcal.com'),
('newsnow', 'https://www.newsnow.co.uk')
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
-- Схема для збереження даних про часові ряди та результати моделювання*/
-- Таблиця для зберігання метаданих моделей часових рядів
CREATE TABLE IF NOT EXISTS time_series_models (
    model_key VARCHAR(100) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'arima', 'sarima', etc.
    timeframe VARCHAR(10) NOT NULL, -- '1m', '5m', '15m', '1h', '4h', '1d'
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE
);

-- Таблиця для зберігання параметрів моделей
CREATE TABLE IF NOT EXISTS model_parameters (
    model_key VARCHAR(100) PRIMARY KEY,
    order_params VARCHAR(20),
    seasonal_order VARCHAR(20),
    seasonal_period INTEGER,
    FOREIGN KEY (model_key) REFERENCES time_series_models(model_key) ON DELETE CASCADE
);

-- Таблиця для зберігання метрик ефективності моделей
CREATE TABLE IF NOT EXISTS model_metrics (
    metric_id SERIAL PRIMARY KEY,
    model_key VARCHAR(100) NOT NULL REFERENCES time_series_models(model_key) ON DELETE CASCADE,
    mse DECIMAL(15,8),
    rmse DECIMAL(15,8),
    mae DECIMAL(15,8),
    mape DECIMAL(15,8),
    r2 DECIMAL(15,8),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metric_name VARCHAR(50) NOT NULL,
    test_date DATE NOT NULL,
    UNIQUE(model_key, metric_name, test_date)
);

-- Таблиця для зберігання прогнозів
CREATE TABLE IF NOT EXISTS model_forecasts (
    forecast_id SERIAL PRIMARY KEY,
    model_key VARCHAR(100) NOT NULL REFERENCES time_series_models(model_key) ON DELETE CASCADE,
    forecast_date TIMESTAMP NOT NULL,
    forecast_value FLOAT NOT NULL,
    lower_bound FLOAT,
    upper_bound FLOAT,
    confidence_level FLOAT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_key, forecast_date)
);

-- Таблиця для зберігання повних результатів тренування моделі
CREATE TABLE IF NOT EXISTS model_binary_data (
    model_key VARCHAR(100) PRIMARY KEY REFERENCES time_series_models(model_key) ON DELETE CASCADE,
    model_binary BYTEA NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Таблиця для зберігання історії перетворень даних
CREATE TABLE IF NOT EXISTS data_transformations (
    transform_id SERIAL PRIMARY KEY,
    model_key VARCHAR(100) NOT NULL REFERENCES time_series_models(model_key) ON DELETE CASCADE,
    transform_type VARCHAR(50) NOT NULL,
    transform_params JSONB,
    transform_order INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_key, transform_order)
);

-- Індекси для оптимізації запитів
CREATE INDEX IF NOT EXISTS idx_time_series_models_symbol ON time_series_models(symbol);
CREATE INDEX IF NOT EXISTS idx_time_series_models_model_type ON time_series_models(model_type);
CREATE INDEX IF NOT EXISTS idx_model_forecasts_date ON model_forecasts(forecast_date);
CREATE INDEX IF NOT EXISTS idx_model_forecasts_model_key_date ON model_forecasts(model_key, forecast_date);
CREATE INDEX IF NOT EXISTS idx_model_parameters_key ON model_parameters(model_key);
CREATE INDEX IF NOT EXISTS idx_model_metrics_key ON model_metrics(model_key);
CREATE INDEX IF NOT EXISTS idx_model_forecasts_key ON model_forecasts(model_key);
CREATE INDEX IF NOT EXISTS idx_model_binary_data_key ON model_binary_data(model_key);
CREATE INDEX IF NOT EXISTS idx_data_transformations_key ON data_transformations(model_key);
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
    id SERIAL PRIMARY KEY,
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
    id SERIAL PRIMARY KEY,
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
    id SERIAL PRIMARY KEY,
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
    id SERIAL PRIMARY KEY,
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
    id SERIAL PRIMARY KEY,
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
    id SERIAL PRIMARY KEY,
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
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    market_symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    beta_value FLOAT NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    window_size INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, market_symbol, timestamp, timeframe, window_size)
);



-- Таблиця для збереження провідних індикаторів
CREATE TABLE IF NOT EXISTS leading_indicators (
    id SERIAL PRIMARY KEY,
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
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    external_asset VARCHAR(50) NOT NULL,
    correlation_value FLOAT NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    method VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, external_asset, timeframe, start_time, end_time, method)
);

-- Таблиця для збереження аналізу кореляцій у різних ринкових режимах
CREATE TABLE IF NOT EXISTS market_regime_correlations (
    id SERIAL PRIMARY KEY,
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
CREATE TABLE IF NOT EXISTS market_cycles (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,                -- Cryptocurrency symbol
    cycle_type VARCHAR(50) NOT NULL,            -- Type of cycle (bull, bear, halving, etc.)
    start_date TIMESTAMP NOT NULL,              -- Start of the cycle
    end_date TIMESTAMP,                         -- End of the cycle (NULL if ongoing)
    peak_date TIMESTAMP,                        -- Date of cycle peak (if applicable)
    peak_price DECIMAL(24, 8),                  -- Price at peak
    bottom_date TIMESTAMP,                      -- Date of cycle bottom (if applicable)
    bottom_price DECIMAL(24, 8),                -- Price at bottom
    max_drawdown DECIMAL(10, 2),                -- Maximum drawdown percentage
    max_roi DECIMAL(10, 2),                     -- Maximum ROI percentage
    cycle_duration_days INTEGER,                -- Duration of the cycle in days
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for faster retrieval of cycles by symbol and type
CREATE INDEX IF NOT EXISTS idx_market_cycles_lookup ON market_cycles (symbol, cycle_type, start_date);

-- Table for storing cycle features for deep learning
CREATE TABLE IF NOT EXISTS cycle_features (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,                -- Cryptocurrency symbol
    timestamp TIMESTAMP NOT NULL,               -- Time of the features
    timeframe VARCHAR(5) NOT NULL,              -- Timeframe of the data
    days_since_last_halving INTEGER,            -- Days since last BTC halving
    days_to_next_halving INTEGER,               -- Days to next BTC halving
    halving_cycle_phase DECIMAL(10, 4),         -- Position in halving cycle (0-1)
    days_since_last_eth_upgrade INTEGER,        -- Days since last ETH upgrade
    days_to_next_eth_upgrade INTEGER,           -- Days to next ETH upgrade (if known)
    eth_upgrade_cycle_phase DECIMAL(10, 4),     -- Position in ETH upgrade cycle
    days_since_last_sol_event INTEGER,          -- Days since last SOL event
    sol_network_stability_score DECIMAL(10, 4), -- Derived from outage history
    weekly_cycle_position DECIMAL(10, 4),       -- Position in weekly cycle
    monthly_seasonality_factor DECIMAL(10, 4),  -- Monthly seasonality factor
    market_phase VARCHAR(20),                   -- Current market phase label
    optimal_cycle_length INTEGER,               -- Detected optimal cycle length
    btc_correlation DECIMAL(10, 4),             -- Correlation with BTC
    eth_correlation DECIMAL(10, 4),             -- Correlation with ETH
    sol_correlation DECIMAL(10, 4),             -- Correlation with SOL
    volatility_metric DECIMAL(10, 4),           -- Current volatility metric
    is_anomaly BOOLEAN DEFAULT FALSE,           -- Whether current pattern is anomalous
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (symbol, timeframe, timestamp)       -- Ensure no duplicate entries
);

-- Index for faster retrieval of features
CREATE INDEX IF NOT EXISTS idx_cycle_features_lookup ON cycle_features (symbol, timeframe, timestamp);

-- Table for storing similarity scores between current and historical cycles
CREATE TABLE IF NOT EXISTS cycle_similarity (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,                -- Cryptocurrency symbol
    reference_cycle_id INTEGER NOT NULL,        -- ID of the reference cycle
    compared_cycle_id INTEGER NOT NULL,         -- ID of the compared cycle
    similarity_score DECIMAL(10, 4) NOT NULL,   -- Similarity score (0-1)
    normalized BOOLEAN NOT NULL,                -- Whether comparison was normalized
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (reference_cycle_id) REFERENCES market_cycles(id),
    FOREIGN KEY (compared_cycle_id) REFERENCES market_cycles(id)
);

-- Table for storing predicted turning points
CREATE TABLE IF NOT EXISTS predicted_turning_points (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,                -- Cryptocurrency symbol
    prediction_date TIMESTAMP NOT NULL,         -- Date of prediction
    predicted_point_date TIMESTAMP NOT NULL,    -- Predicted turning point date
    point_type VARCHAR(20) NOT NULL,            -- Type of turning point (top, bottom)
    confidence DECIMAL(10, 4) NOT NULL,         -- Confidence level (0-1)
    price_prediction DECIMAL(24, 8),            -- Predicted price at turning point
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    actual_outcome VARCHAR(20),                 -- Actual outcome once known (confirmed, missed)
    actual_date TIMESTAMP,                      -- Actual date if confirmed
    actual_price DECIMAL(24, 8),                -- Actual price if confirmed
    updated_at TIMESTAMP                        -- When outcome was updated
);

-- Table for storing model performance metrics related to cycle features
CREATE TABLE IF NOT EXISTS cycle_feature_performance (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(100) NOT NULL,             -- ID of the deep learning model
    feature_name VARCHAR(100) NOT NULL,         -- Name of the cycle feature
    feature_importance DECIMAL(10, 4),          -- Importance score of the feature
    correlation_to_target DECIMAL(10, 4),       -- Correlation to prediction target
    symbol VARCHAR(20) NOT NULL,                -- Cryptocurrency symbol
    timeframe VARCHAR(5) NOT NULL,              -- Timeframe used
    training_period_start TIMESTAMP,            -- Start of training period
    training_period_end TIMESTAMP,              -- End of training period
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS btc_lstm_data (
    id SERIAL PRIMARY KEY,
    timeframe TEXT NOT NULL,
    sequence_id INTEGER NOT NULL, -- To group related sequences
    sequence_position INTEGER NOT NULL, -- Position within sequence
    open_time TIMESTAMP NOT NULL,

    -- Features (scaled data ready for neural networks)
    open_scaled NUMERIC NOT NULL,
    high_scaled NUMERIC NOT NULL,
    low_scaled NUMERIC NOT NULL,
    close_scaled NUMERIC NOT NULL,
    volume_scaled NUMERIC,

    -- Time features (cyclic encoding)
    hour_sin NUMERIC,
    hour_cos NUMERIC,
    day_of_week_sin NUMERIC,
    day_of_week_cos NUMERIC,
    month_sin NUMERIC,
    month_cos NUMERIC,
    day_of_month_sin NUMERIC,
    day_of_month_cos NUMERIC,
    volume_change_scaled NUMERIC,
    volume_rolling_mean_scaled NUMERIC,
    volume_rolling_std_scaled NUMERIC,
    volume_spike_scaled NUMERIC,


    -- Target values (future prices for different horizons)
    target_close_1 NUMERIC, -- Next timeframe close
    target_close_5 NUMERIC, -- 5 timeframes ahead
    target_close_10 NUMERIC, -- 10 timeframes ahead

    -- Metadata
    sequence_length INTEGER, -- Length of the sequence this row belongs to
    scaling_metadata TEXT, -- JSON with scaling parameters for reconstruction

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (timeframe, sequence_id, sequence_position)
);

-- Similar tables for ETH and SOL
CREATE TABLE IF NOT EXISTS eth_lstm_data (
    id SERIAL PRIMARY KEY,
    timeframe TEXT NOT NULL,
    sequence_id INTEGER NOT NULL, -- To group related sequences
    sequence_position INTEGER NOT NULL, -- Position within sequence
    open_time TIMESTAMP NOT NULL,

    -- Features (scaled data ready for neural networks)
    open_scaled NUMERIC NOT NULL,
    high_scaled NUMERIC NOT NULL,
    low_scaled NUMERIC NOT NULL,
    close_scaled NUMERIC NOT NULL,
    volume_scaled NUMERIC,

    -- Time features (cyclic encoding)
    hour_sin NUMERIC,
    hour_cos NUMERIC,
    day_of_week_sin NUMERIC,
    day_of_week_cos NUMERIC,
    month_sin NUMERIC,
    month_cos NUMERIC,
    day_of_month_sin NUMERIC,
    day_of_month_cos NUMERIC,
    volume_change_scaled NUMERIC,
    volume_rolling_mean_scaled NUMERIC,
    volume_rolling_std_scaled NUMERIC,
    volume_spike_scaled NUMERIC,


    -- Target values (future prices for different horizons)
    target_close_1 NUMERIC, -- Next timeframe close
    target_close_5 NUMERIC, -- 5 timeframes ahead
    target_close_10 NUMERIC, -- 10 timeframes ahead

    -- Metadata
    sequence_length INTEGER, -- Length of the sequence this row belongs to
    scaling_metadata TEXT, -- JSON with scaling parameters for reconstruction

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (timeframe, sequence_id, sequence_position)
);
CREATE TABLE IF NOT EXISTS sol_lstm_data (
    id SERIAL PRIMARY KEY,
    timeframe TEXT NOT NULL,
    sequence_id INTEGER NOT NULL, -- To group related sequences
    sequence_position INTEGER NOT NULL, -- Position within sequence
    open_time TIMESTAMP NOT NULL,

    -- Features (scaled data ready for neural networks)
    open_scaled NUMERIC NOT NULL,
    high_scaled NUMERIC NOT NULL,
    low_scaled NUMERIC NOT NULL,
    close_scaled NUMERIC NOT NULL,
    volume_scaled NUMERIC ,

    -- Time features (cyclic encoding)
    hour_sin NUMERIC,
    hour_cos NUMERIC,
    day_of_week_sin NUMERIC,
    day_of_week_cos NUMERIC,
    month_sin NUMERIC,
    month_cos NUMERIC,
    day_of_month_sin NUMERIC,
    day_of_month_cos NUMERIC,
    volume_change_scaled NUMERIC,
    volume_rolling_mean_scaled NUMERIC,
    volume_rolling_std_scaled NUMERIC,
    volume_spike_scaled NUMERIC,


    -- Target values (future prices for different horizons)
    target_close_1 NUMERIC, -- Next timeframe close
    target_close_5 NUMERIC, -- 5 timeframes ahead
    target_close_10 NUMERIC, -- 10 timeframes ahead

    -- Metadata
    sequence_length INTEGER, -- Length of the sequence this row belongs to
    scaling_metadata TEXT, -- JSON with scaling parameters for reconstruction

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (timeframe, sequence_id, sequence_position)
);
CREATE TABLE IF NOT EXISTS btc_arima_data (
    id SERIAL PRIMARY KEY,
    timeframe TEXT NOT NULL,
    open_time TIMESTAMP NOT NULL,

    -- Original data (for reference)
    original_close NUMERIC NOT NULL,

    -- Stationary transformations
    close_diff NUMERIC,
    close_diff2 NUMERIC,
    close_log NUMERIC,
    close_log_diff NUMERIC,
    close_pct_change NUMERIC,

    -- Seasonal differencing
    close_seasonal_diff NUMERIC,
    close_combo_diff NUMERIC,

    original_volume NUMERIC,
    volume_diff NUMERIC,
    volume_log NUMERIC,
    volume_pct_change NUMERIC,
    volume_seasonal_diff NUMERIC,

    -- Stationarity test results
    adf_pvalue NUMERIC,
    kpss_pvalue NUMERIC,
    is_stationary BOOLEAN,

    -- ACF/PACF info (useful for model configuration)
    significant_lags TEXT, -- Stored as JSON array of significant lags

    -- Additional ARIMA-specific metrics
    residual_variance NUMERIC,
    aic_score NUMERIC,
    bic_score NUMERIC,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (timeframe, open_time)
    );

-- Similar tables for ETH and SOL
CREATE TABLE IF NOT EXISTS eth_arima_data (
    id SERIAL PRIMARY KEY,
    timeframe TEXT NOT NULL,
    open_time TIMESTAMP NOT NULL,

    -- Original data (for reference)
    original_close NUMERIC NOT NULL,

    -- Stationary transformations
    close_diff NUMERIC,
    close_diff2 NUMERIC,
    close_log NUMERIC,
    close_log_diff NUMERIC,
    close_pct_change NUMERIC,

    -- Seasonal differencing
    close_seasonal_diff NUMERIC,
    close_combo_diff NUMERIC,

    original_volume NUMERIC,
    volume_diff NUMERIC,
    volume_log NUMERIC,
    volume_pct_change NUMERIC,
    volume_seasonal_diff NUMERIC,

    -- Stationarity test results
    adf_pvalue NUMERIC,
    kpss_pvalue NUMERIC,
    is_stationary BOOLEAN,

    -- ACF/PACF info (useful for model configuration)
    significant_lags TEXT, -- Stored as JSON array of significant lags

    -- Additional ARIMA-specific metrics
    residual_variance NUMERIC,
    aic_score NUMERIC,
    bic_score NUMERIC,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (timeframe, open_time)
);
CREATE TABLE IF NOT EXISTS sol_arima_data (
    id SERIAL PRIMARY KEY,
    timeframe TEXT NOT NULL,
    open_time TIMESTAMP NOT NULL,

    -- Original data (for reference)
    original_close NUMERIC NOT NULL,

    -- Stationary transformations
    close_diff NUMERIC,
    close_diff2 NUMERIC,
    close_log NUMERIC,
    close_log_diff NUMERIC,
    close_pct_change NUMERIC,

    -- Seasonal differencing
    close_seasonal_diff NUMERIC,
    close_combo_diff NUMERIC,

    original_volume NUMERIC,
    volume_diff NUMERIC,
    volume_log NUMERIC,
    volume_pct_change NUMERIC,
    volume_seasonal_diff NUMERIC,

    -- Stationarity test results
    adf_pvalue NUMERIC,
    kpss_pvalue NUMERIC,
    is_stationary BOOLEAN,

    -- ACF/PACF info (useful for model configuration)
    significant_lags TEXT, -- Stored as JSON array of significant lags

    -- Additional ARIMA-specific metrics
    residual_variance NUMERIC,
    aic_score NUMERIC,
    bic_score NUMERIC,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (timeframe, open_time)
);
CREATE INDEX IF NOT EXISTS idx_btc_arima_timeframe ON btc_arima_data(timeframe);
CREATE INDEX IF NOT EXISTS idx_btc_arima_open_time ON btc_arima_data(open_time);
CREATE INDEX IF NOT EXISTS idx_btc_arima_is_stationary ON btc_arima_data(is_stationary);

CREATE INDEX IF NOT EXISTS idx_eth_arima_timeframe ON eth_arima_data(timeframe);
CREATE INDEX IF NOT EXISTS idx_eth_arima_open_time ON eth_arima_data(open_time);
CREATE INDEX IF NOT EXISTS idx_eth_arima_is_stationary ON eth_arima_data(is_stationary);

CREATE INDEX IF NOT EXISTS idx_sol_arima_timeframe ON sol_arima_data(timeframe);
CREATE INDEX IF NOT EXISTS idx_sol_arima_open_time ON sol_arima_data(open_time);
CREATE INDEX IF NOT EXISTS idx_sol_arima_is_stationary ON sol_arima_data(is_stationary);

CREATE INDEX IF NOT EXISTS idx_btc_lstm_timeframe ON btc_lstm_data(timeframe);
CREATE INDEX IF NOT EXISTS idx_btc_lstm_sequence_id ON btc_lstm_data(sequence_id);
CREATE INDEX IF NOT EXISTS idx_btc_lstm_open_time ON btc_lstm_data(open_time);

CREATE INDEX IF NOT EXISTS idx_eth_lstm_timeframe ON eth_lstm_data(timeframe);
CREATE INDEX IF NOT EXISTS idx_eth_lstm_sequence_id ON eth_lstm_data(sequence_id);
CREATE INDEX IF NOT EXISTS idx_eth_lstm_open_time ON eth_lstm_data(open_time);

CREATE INDEX IF NOT EXISTS idx_sol_lstm_timeframe ON sol_lstm_data(timeframe);
CREATE INDEX IF NOT EXISTS idx_sol_lstm_sequence_id ON sol_lstm_data(sequence_id);
CREATE INDEX IF NOT EXISTS idx_sol_lstm_open_time ON sol_lstm_data(open_time);
-- Створення таблиці для зберігання розрахованих метрик волатильності
CREATE TABLE IF NOT EXISTS volatility_metrics (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,                  -- Символ криптовалюти
    timeframe VARCHAR(10) NOT NULL,               -- Часовий інтервал
    timestamp TIMESTAMP NOT NULL,                 -- Час вимірювання
    hist_vol_7d DECIMAL(10, 6),                   -- Історична волатильність за 7 днів
    hist_vol_14d DECIMAL(10, 6),                  -- Історична волатильність за 14 днів
    hist_vol_30d DECIMAL(10, 6),                  -- Історична волатильність за 30 днів
    hist_vol_60d DECIMAL(10, 6),                  -- Історична волатильність за 60 днів
    parkinson_vol DECIMAL(10, 6),                 -- Волатильність за методом Паркінсона
    garman_klass_vol DECIMAL(10, 6),              -- Волатильність за методом Гарман-Класс
    yang_zhang_vol DECIMAL(10, 6),                -- Волатильність за методом Янг-Чжан
    vol_of_vol DECIMAL(10, 6),                    -- Волатильність волатильності
    regime_id INTEGER,                            -- Ідентифікатор режиму волатильності
    is_breakout BOOLEAN,                          -- Флаг пробою волатильності
    CONSTRAINT unique_volatility_metrics UNIQUE (symbol, timeframe, timestamp)
);

-- Створення індексів для таблиці метрик волатильності
CREATE INDEX IF NOT EXISTS idx_vol_metrics_symbol ON volatility_metrics (symbol);
CREATE INDEX IF NOT EXISTS idx_vol_metrics_timestamp ON volatility_metrics (timestamp);
CREATE INDEX IF NOT EXISTS idx_vol_metrics_regime ON volatility_metrics (regime_id);

-- Створення таблиці для збереження GARCH моделей та їх параметрів
CREATE TABLE IF NOT EXISTS volatility_models (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,                  -- Символ криптовалюти
    timeframe VARCHAR(10) NOT NULL,               -- Часовий інтервал
    model_type VARCHAR(20) NOT NULL,              -- Тип моделі (GARCH, EGARCH, GJR-GARCH)
    p INTEGER NOT NULL,                           -- Параметр p моделі GARCH
    q INTEGER NOT NULL,                           -- Параметр q моделі GARCH
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),  -- Час створення моделі
    updated_at TIMESTAMP,                         -- Час останнього оновлення
    parameters JSONB,                             -- Параметри моделі у форматі JSON
    aic DECIMAL(15, 6),                           -- Критерій AIC
    bic DECIMAL(15, 6),                           -- Критерій BIC
    log_likelihood DECIMAL(15, 6),                -- Лог-правдоподібність
    serialized_model BYTEA,                       -- Серіалізована модель
    CONSTRAINT unique_volatility_model UNIQUE (symbol, timeframe, model_type, p, q)
);

-- Створення таблиці для зберігання інформації про режими волатильності
CREATE TABLE IF NOT EXISTS volatility_regimes (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,                  -- Символ криптовалюти
    timeframe VARCHAR(10) NOT NULL,               -- Часовий інтервал
    method VARCHAR(20) NOT NULL,                  -- Метод виявлення режимів (kmeans, hmm, threshold)
    n_regimes INTEGER NOT NULL,                   -- Кількість режимів
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),  -- Час створення
    regime_thresholds DECIMAL(10, 6)[],           -- Порогові значення для режимів
    regime_centroids DECIMAL(10, 6)[],            -- Центроїди кластерів (для kmeans)
    regime_labels VARCHAR(20)[],                  -- Назви режимів (наприклад, ['low', 'medium', 'high'])
    regime_parameters JSONB,                      -- Додаткові параметри режимів
    CONSTRAINT unique_volatility_regime UNIQUE (symbol, timeframe, method, n_regimes)
);

-- Створення таблиці для зберігання результатів аналізу волатильності для ML
CREATE TABLE IF NOT EXISTS volatility_features (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,                  -- Символ криптовалюти
    timeframe VARCHAR(10) NOT NULL,               -- Часовий інтервал
    timestamp TIMESTAMP NOT NULL,                 -- Час створення ознак
    features JSONB NOT NULL,                      -- Ознаки у форматі JSON
    CONSTRAINT unique_volatility_features UNIQUE (symbol, timeframe, timestamp)
);

-- Створення таблиці для зберігання аналізу кросс-активної волатильності
CREATE TABLE IF NOT EXISTS cross_asset_volatility (
    id SERIAL PRIMARY KEY,
    base_symbol VARCHAR(20) NOT NULL,             -- Базовий символ
    compared_symbol VARCHAR(20) NOT NULL,         -- Символ порівняння
    timeframe VARCHAR(10) NOT NULL,               -- Часовий інтервал
    timestamp TIMESTAMP NOT NULL,                 -- Час вимірювання
    correlation DECIMAL(5, 4),                    -- Кореляція волатильності
    lag INTEGER,                                  -- Лаг кореляції (якщо потрібно)
    CONSTRAINT unique_cross_vol UNIQUE (base_symbol, compared_symbol, timeframe, timestamp, lag)
);


CREATE TABLE IF NOT EXISTS trend_analysis (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    analysis_date TIMESTAMP NOT NULL,
    trend_type VARCHAR(20),
    trend_strength FLOAT,
    support_levels JSONB,
    resistance_levels JSONB,
    fibonacci_levels JSONB,
    swing_points JSONB,
    detected_patterns JSONB,
    market_regime VARCHAR(30),
    additional_metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (symbol, timeframe, analysis_date)
);

CREATE TABLE IF NOT EXISTS technical_indicators (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timeframe VARCHAR(5) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    rsi_14 DECIMAL(10, 4),
    macd DECIMAL(10, 4),
    macd_signal DECIMAL(10, 4),
    macd_histogram DECIMAL(10, 4),
    bollinger_upper DECIMAL(24, 8),
    bollinger_middle DECIMAL(24, 8),
    bollinger_lower DECIMAL(24, 8),
    sma_50 DECIMAL(24, 8),
    sma_200 DECIMAL(24, 8),
    ema_12 DECIMAL(24, 8),
    ema_26 DECIMAL(24, 8),
    atr_14 DECIMAL(10, 4),
    stoch_k DECIMAL(10, 4),
    stoch_d DECIMAL(10, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timeframe, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol_timeframe ON technical_indicators(symbol, timeframe);
CREATE TABLE IF NOT EXISTS ml_sequence_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timeframe VARCHAR(5) NOT NULL,
    sequence_start_time TIMESTAMP NOT NULL,
    sequence_end_time TIMESTAMP NOT NULL,
    data_json JSONB NOT NULL,  -- Store feature vectors as JSON
    target_json JSONB NOT NULL,  -- Store target values as JSON
    sequence_length INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timeframe, sequence_start_time)
);

CREATE INDEX IF NOT EXISTS idx_ml_sequence_data_symbol_timeframe ON ml_sequence_data(symbol, timeframe);

-- 7. ML Models Table
CREATE TABLE IF NOT EXISTS ml_models (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timeframe VARCHAR(5) NOT NULL,
    model_type VARCHAR(10) NOT NULL,  -- 'lstm', 'gru'
    model_version VARCHAR(20) NOT NULL,
    model_path TEXT NOT NULL,  -- Path to the saved model file
    input_features TEXT[] NOT NULL,  -- Array of feature names used
    hidden_dim INTEGER NOT NULL,
    num_layers INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    active BOOLEAN DEFAULT TRUE,
    UNIQUE(symbol, timeframe, model_type, model_version)
);

-- 8. ML Model Metrics Table
CREATE TABLE IF NOT EXISTS ml_model_metrics (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ml_models(id),
    mse DECIMAL(15, 8) NOT NULL,  -- Mean squared error
    rmse DECIMAL(15, 8) NOT NULL,  -- Root mean squared error
    mae DECIMAL(15, 8) NOT NULL,  -- Mean absolute error
    r2_score DECIMAL(5, 4) NOT NULL,  -- R-squared score
    test_date TIMESTAMP NOT NULL,  -- When the model was tested
    training_duration_seconds INTEGER,  -- How long training took
    epochs_completed INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ml_model_metrics_model_id ON ml_model_metrics(model_id);

-- 9. Predictions Table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ml_models(id),
    symbol VARCHAR(10) NOT NULL,
    timeframe VARCHAR(5) NOT NULL,
    prediction_timestamp TIMESTAMP NOT NULL,  -- When the prediction was made
    target_timestamp TIMESTAMP NOT NULL,  -- The time the prediction is for
    predicted_value DECIMAL(24, 8) NOT NULL,
    confidence_interval_low DECIMAL(24, 8),
    confidence_interval_high DECIMAL(24, 8),
    actual_value DECIMAL(24, 8),  -- To be filled when actual value becomes available
    prediction_error DECIMAL(24, 8),  -- To be calculated when actual value becomes available
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_id, symbol, timeframe, target_timestamp)
);

CREATE INDEX IF NOT EXISTS idx_predictions_model_id ON predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_predictions_symbol_timeframe ON predictions(symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_predictions_target_timestamp ON predictions(target_timestamp);
