-- Таблиці для зберігання свічок (курсів) BTC
CREATE TABLE btc_klines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interval TEXT NOT NULL,
    open_time TIMESTAMP NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    close_time TIMESTAMP NOT NULL,
    quote_asset_volume REAL NOT NULL,
    number_of_trades INTEGER NOT NULL,
    taker_buy_base_volume REAL NOT NULL,
    taker_buy_quote_volume REAL NOT NULL,
    is_closed BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (interval, open_time)
);

-- Індекс для швидкого пошуку свічок BTC за часом
CREATE INDEX idx_btc_klines_time ON btc_klines(interval, open_time);

-- Таблиці для зберігання свічок (курсів) ETH
CREATE TABLE eth_klines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interval TEXT NOT NULL,
    open_time TIMESTAMP NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    close_time TIMESTAMP NOT NULL,
    quote_asset_volume REAL NOT NULL,
    number_of_trades INTEGER NOT NULL,
    taker_buy_base_volume REAL NOT NULL,
    taker_buy_quote_volume REAL NOT NULL,
    is_closed BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (interval, open_time)
);

-- Індекс для швидкого пошуку свічок ETH за часом
CREATE INDEX idx_eth_klines_time ON eth_klines(interval, open_time);

-- Таблиці для зберігання свічок (курсів) SOL
CREATE TABLE sol_klines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interval TEXT NOT NULL,
    open_time TIMESTAMP NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    close_time TIMESTAMP NOT NULL,
    quote_asset_volume REAL NOT NULL,
    number_of_trades INTEGER NOT NULL,
    taker_buy_base_volume REAL NOT NULL,
    taker_buy_quote_volume REAL NOT NULL,
    is_closed BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (interval, open_time)
);

-- Індекс для швидкого пошуку свічок SOL за часом
CREATE INDEX idx_sol_klines_time ON sol_klines(interval, open_time);

-- Таблиця для зберігання книги ордерів BTC
CREATE TABLE btc_orderbook (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP NOT NULL,
    last_update_id INTEGER NOT NULL,
    type TEXT NOT NULL, -- 'bid' або 'ask'
    price REAL NOT NULL,
    quantity REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Індекс для швидкого пошуку в книзі ордерів BTC
CREATE INDEX idx_btc_orderbook_time ON btc_orderbook(timestamp);

-- Таблиця для зберігання книги ордерів ETH
CREATE TABLE eth_orderbook (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP NOT NULL,
    last_update_id INTEGER NOT NULL,
    type TEXT NOT NULL, -- 'bid' або 'ask'
    price REAL NOT NULL,
    quantity REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Індекс для швидкого пошуку в книзі ордерів ETH
CREATE INDEX idx_eth_orderbook_time ON eth_orderbook(timestamp);

-- Таблиця для зберігання книги ордерів SOL
CREATE TABLE sol_orderbook (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP NOT NULL,
    last_update_id INTEGER NOT NULL,
    type TEXT NOT NULL, -- 'bid' або 'ask'
    price REAL NOT NULL,
    quantity REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Індекс для швидкого пошуку в книзі ордерів SOL
CREATE INDEX idx_sol_orderbook_time ON sol_orderbook(timestamp);

-- Таблиця для логування подій
CREATE TABLE logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_level TEXT NOT NULL, -- 'INFO', 'WARNING', 'ERROR', 'DEBUG'
    message TEXT NOT NULL,
    component TEXT NOT NULL, -- 'BinanceClient', 'WebSocket', etc.
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);