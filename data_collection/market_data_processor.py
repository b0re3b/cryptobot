import pandas as pd
import numpy as np
import datetime as dt
import ta as t
import scikit-learn
import scipy as sp

class MarketDataProcessor:
    def __init__(self, cache_dir=None)
    def load_data(self, data_source, symbol, interval, start_date=None, end_date=None)
    def clean_data(self, data, remove_outliers=True, fill_missing=True)
    def resample_data(self, data, target_interval)
    def calculate_indicators(self, data, indicators=None)
    def prepare_features(self, data, target='close', shift=1, window_sizes=[5, 10, 20])
    def normalize_data(self, data, method='z-score')
    def create_train_test_split(self, data, test_size=0.2, validation_size=0.1)
    def save_processed_data(self, data, filename)
    def create_lagged_features(self, data, lag_periods=[1, 3, 5, 7, 14])
    def add_time_features(self, data)  # день тижня, година дня, тощо
    def detect_outliers(self, data, method='zscore', threshold=3)
    def visualize_features(self, data, features=None)