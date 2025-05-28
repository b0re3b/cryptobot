import numpy as np
import pandas as pd
import pandas_ta as ta
from utils.config import *
from cyclefeatures.BitcoinCycleFeatureExtractor import BitcoinCycleFeatureExtractor
from utils.logger import CryptoLogger
from decimal import Decimal


class MarketPhaseFeatureExtractor:
    def __init__(self):
        self.eth_significant_events = eth_significant_events
        self.sol_significant_events = sol_significant_events
        self.bitcoin_cycle_feature = BitcoinCycleFeatureExtractor()
        self.logger = CryptoLogger('MarketPhaseFeatureExtractor')

    def detect_market_phase(self, processed_data: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """
            Визначає ринкову фазу (наприклад, накопичення, зростання, розподіл, спад) на основі технічних індикаторів.

            Метод обчислює ключові технічні індикатори, такі як ковзні середні (SMA), волатильність, імпульс (momentum),
            індекс відносної сили (RSI), а також тренд об'єму. На основі поєднання цих показників здійснюється класифікація
            кожної дати у відповідну ринкову фазу.

            Параметри
            ----------
            processed_data : pd.DataFrame
                Вхідний DataFrame, який містить часовий ряд з щонайменше стовпцями 'close' та 'volume'. Індекс повинен бути datetime.

            window : int, optional (default=30)
                Базовий розмір вікна (кількість періодів) для розрахунку ковзних середніх і волатильності. Інші індикатори
                використовують похідні значення від цього вікна (наприклад, window//2, window//4 тощо).

            Повертає
            ---------
            pd.DataFrame
                DataFrame з додатковою колонкою `market_phase`, яка позначає ринкову фазу для кожного моменту часу:
                    - 'accumulation': період стабілізації після падіння
                    - 'uptrend': період стійкого зростання
                    - 'distribution': період уповільнення зростання або флатового ринку
                    - 'downtrend': період зниження
                    - 'undefined': не вдалося класифікувати

            Винятки
            --------
            ValueError:
                Генерується, якщо вхідний DataFrame не містить необхідних стовпців ('close' або 'volume').

            Інше
            ------
            - Для очищення результату всі проміжні технічні індикатори видаляються перед поверненням результату.
            - Усі значення у колонці `market_phase` заповнюються методом forward-fill (ffill), щоб уникнути пропусків.
            - Записуються логування ключових етапів та результатів розпізнавання фаз.
            """
        self.logger.info(f"Starting detect_market_phase with window size: {window}")

        try:
            # Create a copy of the input DataFrame to avoid modifying the original
            result_df = processed_data.copy()

            # Ensure we have the required columns
            required_columns = ['close', 'volume']
            for col in required_columns:
                if col not in result_df.columns:
                    error_msg = f"Required column '{col}' not found in processed_data"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)

            # Ensure all numeric data is of type float to prevent type conflicts
            result_df['close'] = result_df['close'].astype(float)
            result_df['volume'] = result_df['volume'].astype(float)

            # Calculate additional technical indicators
            self.logger.debug("Calculating technical indicators")

            # 1. SMA for trend direction
            result_df['sma_short'] = result_df['close'].rolling(window=window // 3).mean()
            result_df['sma_long'] = result_df['close'].rolling(window=window).mean()

            # 2. Volume trend
            result_df['volume_sma'] = result_df['volume'].rolling(window=window // 2).mean()

            # 3. Price volatility
            result_df['price_std'] = result_df['close'].rolling(window=window // 2).std()
            result_df['volatility'] = result_df['price_std'] / result_df['close']

            # 4. Price momentum
            result_df['momentum'] = result_df['close'].pct_change(periods=window // 4)

            # 5. RSI for overbought/oversold conditions
            delta = result_df['close'].diff()
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)

            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()

            rs = avg_gain / avg_loss.replace(0, 0.00001)  # Avoid division by zero
            result_df['rsi'] = 100 - (100 / (1 + rs))

            # Initialize market phase column
            result_df['market_phase'] = 'undefined'

            self.logger.debug("Identifying market phases")

            # Define market phases based on indicators
            # Phase 1: Accumulation
            accum_mask = (
                    (result_df['sma_short'] < result_df['sma_long']) &
                    (result_df['momentum'] > -0.05) &
                    (result_df['momentum'] < 0.05) &
                    (result_df['volatility'] < result_df['volatility'].rolling(window=window * 2).mean()) &
                    (result_df['rsi'] < 50)
            )
            result_df.loc[accum_mask, 'market_phase'] = 'accumulation'

            # Phase 2: Uptrend
            uptrend_mask = (
                    (result_df['sma_short'] > result_df['sma_long']) &
                    (result_df['momentum'] > 0) &
                    (result_df['volume'] > result_df['volume_sma'])
            )
            result_df.loc[uptrend_mask, 'market_phase'] = 'uptrend'

            # Phase 3: Distribution
            distrib_mask = (
                    (result_df['sma_short'] > result_df['sma_long']) &
                    (result_df['momentum'] < 0.05) &
                    (result_df['momentum'] > -0.05) &
                    (result_df['volatility'] < result_df['volatility'].rolling(window=window * 2).mean()) &
                    (result_df['rsi'] > 50)
            )
            result_df.loc[distrib_mask, 'market_phase'] = 'distribution'

            # Phase 4: Downtrend
            downtrend_mask = (
                    (result_df['sma_short'] < result_df['sma_long']) &
                    (result_df['momentum'] < 0)
            )
            result_df.loc[downtrend_mask, 'market_phase'] = 'downtrend'

            # Remove intermediate calculation columns to keep the DataFrame clean
            # Comment this out if you want to keep these columns for debugging
            columns_to_drop = ['sma_short', 'sma_long', 'volume_sma', 'price_std',
                               'volatility', 'momentum', 'rsi']
            result_df = result_df.drop(columns=columns_to_drop, errors='ignore')

            # Forward fill any NaN values in the market_phase column
            result_df['market_phase'] = result_df['market_phase'].ffill()

            phase_counts = result_df['market_phase'].value_counts()
            self.logger.info(f"Market phase distribution: {phase_counts.to_dict()}")

            return result_df

        except Exception as e:
            self.logger.error(f"Error in detect_market_phase: {str(e)}", exc_info=True)
            raise

    def identify_bull_bear_cycles(self, processed_data: pd.DataFrame,
                                  threshold_bull: float = 0.2,
                                  threshold_bear: float = -0.2,
                                  min_duration_days: int = 20) -> pd.DataFrame:
        """
            Визначає цикли бичачого (bull) та ведмежого (bear) ринку на основі відновлення (recovery)
            та просадки (drawdown) ціни.

            Метод аналізує часовий ряд цін закриття (close) активу, визначає фази ринку (bull/bear),
            розраховує ключові показники для кожного циклу та повертає розширений DataFrame із цією інформацією.

            Параметри
            ----------
            processed_data : pd.DataFrame
                Часовий ряд, який містить щонайменше колонку `'close'` та має `DatetimeIndex` як індекс.

            threshold_bull : float, optional, default=0.2
                Порогове значення для виявлення початку бичачого циклу (відновлення ціни на 20%+).

            threshold_bear : float, optional, default=-0.2
                Порогове значення для виявлення початку ведмежого циклу (просадка ціни на 20%-).

            min_duration_days : int, optional, default=20
                Мінімальна кількість днів, яку повинен тривати цикл, щоб вважатися валідним.

            Повертає
            -------
            pd.DataFrame
                Розширений DataFrame, що містить:
                - оригінальні дані;
                - метадані по кожному дню: стан циклу, ROI, тривалість, початкову/макс/мін ціну тощо;
                - атрибут `.cycles_summary` з окремим DataFrame з агрегацією по кожному валідному циклу.

            Атрибути результату
            -------------------
            result.cycles_summary : pd.DataFrame
                DataFrame з інформацією про кожен валідний цикл:
                ['cycle_id', 'cycle_type', 'start_date', 'end_date',
                 'duration_days', 'start_price', 'end_price',
                 'max_price', 'min_price', 'total_roi',
                 'max_roi', 'max_drawdown'].

            """

        self.logger.info(
            f"Starting identify_bull_bear_cycles with thresholds: bull={threshold_bull}, bear={threshold_bear}, min_duration={min_duration_days} days")

        try:
            # Create a copy of the input DataFrame
            df = processed_data.copy()

            # Ensure we have the required columns
            if 'close' not in df.columns:
                error_msg = "Required column 'close' not found in processed_data"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Ensure the DataFrame has a datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                error_msg = "DataFrame index must be a DatetimeIndex"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Convert close price to float to prevent decimal/float type conflicts
            df['close'] = df['close'].astype(float)

            # Calculate rolling maximum and minimum prices
            self.logger.debug("Calculating drawdowns and recoveries")
            df['rolling_max'] = df['close'].cummax()
            df['drawdown'] = (df['close'] / df['rolling_max']) - 1

            df['rolling_min'] = df['close'].cummin()
            df['recovery'] = (df['close'] / df['rolling_min']) - 1

            # Initialize cycle state variables
            cycle_states = pd.DataFrame(index=df.index)
            cycle_states['cycle_state'] = 'undefined'
            cycle_states['cycle_id'] = 0
            cycle_states['days_in_cycle'] = 0
            cycle_states['cycle_start_price'] = np.nan
            cycle_states['cycle_max_price'] = np.nan
            cycle_states['cycle_min_price'] = np.nan
            cycle_states['cycle_current_roi'] = 0.0
            cycle_states['cycle_max_roi'] = 0.0
            cycle_states['cycle_max_drawdown'] = 0.0

            # Identify cycle states
            self.logger.debug("Identifying bull/bear cycles")
            current_state = 'undefined'
            current_cycle_id = 0
            cycle_start_idx = 0
            cycle_start_price = df['close'].iloc[0]
            max_price_in_current_cycle = df['close'].iloc[0]
            min_price_in_current_cycle = df['close'].iloc[0]

            for i in range(1, len(df)):
                current_price = df['close'].iloc[i]
                current_date = df.index[i]

                # Check if we need to change state
                if current_state == 'undefined' or current_state == 'bear':
                    # Check for bull market signal
                    if df['recovery'].iloc[i] >= threshold_bull:
                        if current_state != 'bull':
                            # Start a new bull cycle
                            current_state = 'bull'
                            current_cycle_id += 1
                            cycle_start_idx = i
                            cycle_start_price = current_price
                            max_price_in_current_cycle = current_price
                            min_price_in_current_cycle = current_price
                            self.logger.info(f"New bull cycle detected at {current_date}, price: {current_price}")

                elif current_state == 'bull':
                    # Check for bear market signal
                    if df['drawdown'].iloc[i] <= threshold_bear:
                        # Switch to bear cycle
                        current_state = 'bear'
                        current_cycle_id += 1
                        cycle_start_idx = i
                        cycle_start_price = current_price
                        max_price_in_current_cycle = current_price
                        min_price_in_current_cycle = current_price
                        self.logger.info(f"New bear cycle detected at {current_date}, price: {current_price}")

                # Update cycle statistics
                days_in_cycle = (df.index[i] - df.index[cycle_start_idx]).days

                # Update max and min prices in the current cycle
                max_price_in_current_cycle = max(max_price_in_current_cycle, current_price)
                min_price_in_current_cycle = min(min_price_in_current_cycle, current_price)

                # Calculate ROI and drawdown for the current cycle
                cycle_current_roi = (current_price / cycle_start_price) - 1
                cycle_max_roi = (max_price_in_current_cycle / cycle_start_price) - 1
                cycle_max_drawdown = (min_price_in_current_cycle / max_price_in_current_cycle) - 1

                # Update the cycle states DataFrame
                cycle_states.loc[current_date, 'cycle_state'] = current_state
                cycle_states.loc[current_date, 'cycle_id'] = current_cycle_id
                cycle_states.loc[current_date, 'days_in_cycle'] = days_in_cycle
                cycle_states.loc[current_date, 'cycle_start_price'] = cycle_start_price
                cycle_states.loc[current_date, 'cycle_max_price'] = max_price_in_current_cycle
                cycle_states.loc[current_date, 'cycle_min_price'] = min_price_in_current_cycle
                cycle_states.loc[current_date, 'cycle_current_roi'] = cycle_current_roi
                cycle_states.loc[current_date, 'cycle_max_roi'] = cycle_max_roi
                cycle_states.loc[current_date, 'cycle_max_drawdown'] = cycle_max_drawdown

            # Filter out cycles that don't meet the minimum duration
            self.logger.debug("Filtering cycles by minimum duration")
            valid_cycles = []
            for cycle_id in cycle_states['cycle_id'].unique():
                cycle_data = cycle_states[cycle_states['cycle_id'] == cycle_id]
                if cycle_data['days_in_cycle'].max() >= min_duration_days:
                    valid_cycles.append(cycle_id)
                else:
                    self.logger.info(f"Filtering out cycle {cycle_id} with duration < {min_duration_days} days")

            # Mark invalid cycles as undefined
            cycle_states.loc[~cycle_states['cycle_id'].isin(valid_cycles), 'cycle_state'] = 'undefined'

            # Create a summary DataFrame with cycle statistics
            cycles_summary = []

            for cycle_id in valid_cycles:
                cycle_data = cycle_states[cycle_states['cycle_id'] == cycle_id]
                if len(cycle_data) > 0:
                    cycle_type = cycle_data['cycle_state'].iloc[0]
                    start_date = cycle_data.index[0]
                    end_date = cycle_data.index[-1]
                    duration_days = (end_date - start_date).days
                    start_price = cycle_data['cycle_start_price'].iloc[0]
                    end_price = df.loc[end_date, 'close']
                    max_price = cycle_data['cycle_max_price'].max()
                    min_price = cycle_data['cycle_min_price'].min()
                    total_roi = (end_price / start_price) - 1
                    max_roi = cycle_data['cycle_max_roi'].max()
                    max_drawdown = cycle_data['cycle_max_drawdown'].min()

                    cycle_summary = {
                        'cycle_id': cycle_id,
                        'cycle_type': cycle_type,
                        'start_date': start_date,
                        'end_date': end_date,
                        'duration_days': duration_days,
                        'start_price': start_price,
                        'end_price': end_price,
                        'max_price': max_price,
                        'min_price': min_price,
                        'total_roi': total_roi,
                        'max_roi': max_roi,
                        'max_drawdown': max_drawdown
                    }
                    cycles_summary.append(cycle_summary)
                    self.logger.info(
                        f"Cycle {cycle_id} summary: {cycle_type}, duration: {duration_days} days, ROI: {total_roi:.2%}")

            # Create the cycles summary DataFrame
            cycles_summary_df = pd.DataFrame(cycles_summary)

            # Merge the original data with cycle information
            result = pd.concat([df, cycle_states], axis=1)

            # Remove intermediate calculation columns
            result = result.drop(['rolling_max', 'drawdown', 'rolling_min', 'recovery'], axis=1, errors='ignore')

            # Add cycles_summary as an attribute
            result.cycles_summary = cycles_summary_df if len(cycles_summary) > 0 else pd.DataFrame()

            self.logger.info(f"Found {len(valid_cycles)} valid cycles")

            return result

        except Exception as e:
            self.logger.error(f"Error in identify_bull_bear_cycles: {str(e)}", exc_info=True)
            raise

    def calculate_volatility_by_cycle_phase(self, processed_data: pd.DataFrame,
                                            symbol: str,
                                            cycle_type: str = 'auto',
                                            window: int = 14) -> pd.DataFrame:
        """
           Обчислює волатильність цін та визначає фази циклу для заданого криптоактиву залежно від типу циклу.

           Цей метод розраховує показники волатильності, зокрема ковзне стандартне відхилення і нормалізовану волатильність,
           присвоює кожній точці даних фазу циклу (наприклад, фази халвінгу для BTC, фази оновлень мережі для ETH),
           а також обчислює зведену статистику волатильності по кожній фазі.

           Параметри
           ---------
           processed_data : pd.DataFrame
               DataFrame з індексом типу DatetimeIndex, який містить щонайменше колонку 'close' з цінами.
           symbol : str
               Символ криптовалюти (наприклад, 'BTCUSDT', 'ETHUSD').
           cycle_type : str, optional
               Тип циклу для визначення фаз. Можливі варіанти:
               - 'auto': автоматичне визначення на основі символу
               - 'halving': цикл халвінгу Bitcoin
               - 'network_upgrade': оновлення мережі Ethereum
               - 'ecosystem_event': події в екосистемі Solana (зокрема збої)
               - 'bull_bear': загальні цикли бичачого/ведмежого ринку
               За замовчуванням — 'auto'.
           window : int, optional
               Розмір ковзного вікна в днях для обчислення волатильності. За замовчуванням — 14.

           Повертає
           --------
           pd.DataFrame
               DataFrame з додатковими колонками:
               - 'volatility': ковзне стандартне відхилення відсоткових змін
               - 'volatility_normalized': волатильність, поділена на ковзне середнє ціни
               - 'cycle_phase': призначена фаза циклу для кожної дати
               - додаткові колонки, специфічні для обраного типу циклу (наприклад, 'cycle_number', 'cycle_state')

               Також додається атрибут `phase_volatility_summary`, що містить зведену статистику по кожній фазі:
               - середнє, медіанне, мінімальне, максимальне значення волатильності
               - стандартне відхилення волатильності (мета-волатильність)
               - 25-й та 75-й процентилі
               - кількість спостережень у фазі

           Виключення
           ----------
           ValueError
               Якщо у вхідних даних відсутня колонка 'close' або індекс не є типом DatetimeIndex.
           Exception
               У разі інших неочікуваних помилок під час обробки.

           Примітки
           --------
           - Метод підтримує кілька стратегій класифікації циклів залежно від типу активу.
           - Атрибут `phase_volatility_summary` не є колонкою DataFrame, а окремим доданим об'єктом.
           - Проміжні колонки, такі як 'log_return' та 'volatility_sq', видаляються перед поверненням результату.


           """
        self.logger.info(
            f"Starting calculate_volatility_by_cycle_phase for {symbol}, cycle_type: {cycle_type}, window: {window}")

        try:
            # Create a copy of the input DataFrame
            result_df = processed_data.copy()

            # Ensure we have the required columns
            if 'close' not in result_df.columns:
                error_msg = "Required column 'close' not found in processed_data"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Ensure the DataFrame has a datetime index
            if not isinstance(result_df.index, pd.DatetimeIndex):
                error_msg = "DataFrame index must be a DatetimeIndex"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Convert close price to float to prevent decimal/float type conflicts
            result_df['close'] = result_df['close'].astype(float)

            # Clean symbol format
            clean_symbol = symbol.upper().replace('USDT', '').replace('USD', '')
            self.logger.debug(f"Cleaned symbol: {clean_symbol}")

            # Determine the cycle type based on the symbol if set to 'auto'
            if cycle_type == 'auto':
                if clean_symbol == 'BTC':
                    cycle_type = 'halving'
                elif clean_symbol == 'ETH':
                    cycle_type = 'network_upgrade'
                elif clean_symbol == 'SOL':
                    cycle_type = 'ecosystem_event'
                else:
                    cycle_type = 'bull_bear'  # Default for other cryptocurrencies
                self.logger.info(f"Auto-detected cycle type: {cycle_type} for {clean_symbol}")

            # Calculate price volatility (rolling standard deviation)
            result_df['volatility'] = result_df['close'].pct_change().rolling(window=window).std()

            # Calculate log returns for additional volatility metrics
            result_df['log_return'] = np.log(result_df['close'] / result_df['close'].shift(1))

            # Calculate additional volatility metrics
            result_df['volatility_normalized'] = result_df['volatility'] / result_df['close'].rolling(
                window=window).mean()
            result_df['volatility_sq'] = result_df['log_return'] ** 2  # Squared returns (for GARCH-like analysis)

            # Determine cycle phases based on the cycle type
            self.logger.debug(f"Determining cycle phases based on {cycle_type}")
            if cycle_type == 'halving':
                # For BTC halving cycles
                self.logger.debug("Calculating BTC halving cycle features")
                halving_data = self.bitcoin_cycle_feature.calculate_btc_halving_cycle_features(result_df)

                # Define cycle phases based on halving cycle phase
                # Four phases: Phase 1 (0-0.25), Phase 2 (0.25-0.5), Phase 3 (0.5-0.75), Phase 4 (0.75-1.0)
                phase_conditions = [
                    (halving_data['halving_cycle_phase'] <= 0.25),
                    (halving_data['halving_cycle_phase'] > 0.25) & (halving_data['halving_cycle_phase'] <= 0.5),
                    (halving_data['halving_cycle_phase'] > 0.5) & (halving_data['halving_cycle_phase'] <= 0.75),
                    (halving_data['halving_cycle_phase'] > 0.75)
                ]
                phase_values = ['Phase 1 (Early Post-Halving)', 'Phase 2 (Mid Cycle)',
                                'Phase 3 (Late Cycle)', 'Phase 4 (Pre-Halving)']

                halving_data['cycle_phase'] = np.select(phase_conditions, phase_values, default='Unknown')

                # Add halving cycle columns to the result
                for col in ['cycle_number', 'halving_cycle_phase', 'cycle_phase']:
                    if col in halving_data.columns:
                        result_df[col] = halving_data[col]

            elif cycle_type == 'network_upgrade':
                # For ETH network upgrades
                self.logger.debug("Processing ETH network upgrade events")
                eth_events = [pd.Timestamp(event['date']) for event in self.eth_significant_events]
                self.logger.debug(f"Found {len(eth_events)} ETH significant events")
                result_df['cycle_phase'] = 'Between Upgrades'

                # Mark dates near events
                for event in self.eth_significant_events:
                    event_date = pd.Timestamp(event['date'])
                    event_name = event['name']

                    # Mark 30 days before event as "Pre-Upgrade" phase
                    pre_upgrade_mask = (result_df.index >= event_date - pd.Timedelta(days=30)) & (
                            result_df.index < event_date)
                    result_df.loc[pre_upgrade_mask, 'cycle_phase'] = f'Pre-{event_name}'

                    # Mark 30 days after event as "Post-Upgrade" phase
                    post_upgrade_mask = (result_df.index >= event_date) & (
                            result_df.index < event_date + pd.Timedelta(days=30))
                    result_df.loc[post_upgrade_mask, 'cycle_phase'] = f'Post-{event_name}'

                    self.logger.debug(f"Identified phase for {event_name} at {event_date}")

            elif cycle_type == 'ecosystem_event':
                # For SOL ecosystem events
                self.logger.debug("Processing SOL ecosystem events")
                sol_events = [pd.Timestamp(event['date']) for event in self.sol_significant_events]
                self.logger.debug(f"Found {len(sol_events)} SOL significant events")
                result_df['cycle_phase'] = 'Normal Operation'

                # Mark dates near events
                for event in self.sol_significant_events:
                    event_date = pd.Timestamp(event['date'])
                    event_name = event['name']

                    # Mark 15 days after events (especially outages) for analysis
                    if 'Outage' in event_name:
                        outage_mask = (result_df.index >= event_date) & (
                                result_df.index < event_date + pd.Timedelta(days=15))
                        result_df.loc[outage_mask, 'cycle_phase'] = f'Post-{event_name}'
                        self.logger.debug(f"Identified phase for outage event {event_name} at {event_date}")

            elif cycle_type == 'bull_bear':
                # For general bull/bear cycles
                self.logger.debug("Identifying bull/bear cycles")
                bull_bear_data = self.identify_bull_bear_cycles(result_df)

                # Add bull/bear cycle columns to the result
                for col in ['cycle_state', 'cycle_id']:
                    if col in bull_bear_data.columns:
                        result_df[col] = bull_bear_data[col]

                # Use cycle_state as cycle_phase for consistency
                if 'cycle_state' in result_df.columns:
                    result_df['cycle_phase'] = result_df['cycle_state']
                    self.logger.debug("Using cycle_state as cycle_phase")

            # Group by cycle phase and calculate summary statistics
            if 'cycle_phase' in result_df.columns:
                self.logger.debug("Calculating volatility statistics by cycle phase")
                phase_stats = []

                for phase in result_df['cycle_phase'].unique():
                    phase_data = result_df[result_df['cycle_phase'] == phase]

                    # Skip if not enough data points
                    if len(phase_data) < window:
                        self.logger.warning(
                            f"Skipping phase '{phase}' due to insufficient data points ({len(phase_data)} < {window})")
                        continue

                    stats = {
                        'cycle_phase': phase,
                        'avg_volatility': phase_data['volatility'].mean(),
                        'median_volatility': phase_data['volatility'].median(),
                        'max_volatility': phase_data['volatility'].max(),
                        'min_volatility': phase_data['volatility'].min(),
                        'volatility_of_volatility': phase_data['volatility'].std(),  # Meta-volatility
                        'count': len(phase_data),
                        # Add percentiles
                        'volatility_25pct': phase_data['volatility'].quantile(0.25),
                        'volatility_75pct': phase_data['volatility'].quantile(0.75),
                    }
                    phase_stats.append(stats)
                    self.logger.debug(
                        f"Phase '{phase}': avg_volatility={stats['avg_volatility']:.4f}, count={stats['count']}")

                # Create a summary DataFrame
                phase_stats_df = pd.DataFrame(phase_stats)

                # Attach the summary as an attribute of the result DataFrame
                result_df.phase_volatility_summary = phase_stats_df
                self.logger.info(f"Calculated volatility statistics for {len(phase_stats)} phases")
            else:
                self.logger.warning("No 'cycle_phase' column found, skipping phase statistics")

            # Clean up intermediate columns
            result_df = result_df.drop(['log_return', 'volatility_sq'], axis=1, errors='ignore')

            return result_df

        except Exception as e:
            self.logger.error(f"Error in calculate_volatility_by_cycle_phase: {str(e)}", exc_info=True)
            raise