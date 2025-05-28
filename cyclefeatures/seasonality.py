from typing import Dict
import numpy as np
import pandas as pd
import decimal

from utils.logger import CryptoLogger


class TemporalSeasonalityAnalyzer:
    def __init__(self):
        self.logger = CryptoLogger('TemporalSeasonalityAnalyzer')

    def _ensure_float(self, value):
        """Конвертирует decimal.Decimal в float для безопасной работы с числами."""
        if isinstance(value, decimal.Decimal):
            return float(value)
        return value

    def analyze_weekly_cycle(self, processed_data: pd.DataFrame) -> Dict:
        """
           Аналізує тижневі цикли ринку на основі цін закриття активу.

           Метод виконує повний аналіз поведінки ціни по днях тижня:
           - Обчислює середню дохідність, медіану, волатильність, максимальні підйоми/падіння,
             частки позитивних і негативних днів для кожного дня тижня.
           - Аналізує інерційні та зворотні зв’язки між днями (модель імпульсу).
           - Визначає найкращий, найгірший і найволатильніший день тижня.
           - Визначає середній шаблон тижня.
           - Оцінює ефект вихідних (відношення доходності на вихідних до буднів).

           Аргументи:
               processed_data (pd.DataFrame): DataFrame з часовим індексом (`DatetimeIndex`) та колонкою `close` — ціною закриття активу.

           Повертає:
               Dict: Словник зі статистикою та шаблонами, який містить:
                   - 'day_of_week_stats': словник із детальною статистикою по кожному дню тижня.
                   - 'weekly_momentum_patterns': шаблони імпульсного руху між суміжними днями.
                   - 'average_week_pattern': середнє значення ціни для кожного дня тижня.
                   - 'best_day': день тижня з найвищою середньою дохідністю.
                   - 'worst_day': день тижня з найгіршою середньою дохідністю.
                   - 'most_volatile_day': день тижня з найвищою стандартною відхиленням дохідності.
                   - 'weekend_effect': словник з:
                       - 'fri_to_mon_correlation': кореляція між п’ятницею та понеділком.
                       - 'weekend_to_weekday_return_ratio': співвідношення середньої доходності на вихідних до буднів.

           Винятки:
               ValueError: Якщо DataFrame не має `DatetimeIndex` або не містить колонки `close`.
               Інші винятки логуються, а потім проброшуються.

           Примітки:
               - Дані повинні мати щоденну частоту без великих пропусків.
               - Підтримується лише один часовий ряд одночасно.
               - Використовується метод `_ensure_float` для безпечної роботи з типами `decimal.Decimal`.
           """
        try:
            self.logger.info("Starting weekly cycle analysis")
            # Ensure we have the required data
            if not isinstance(processed_data.index, pd.DatetimeIndex):
                self.logger.error("DataFrame index must be a DatetimeIndex")
                raise ValueError("DataFrame index must be a DatetimeIndex")

            if 'close' not in processed_data.columns:
                self.logger.error("DataFrame must have a 'close' column")
                raise ValueError("DataFrame must have a 'close' column")

            # Create a copy to avoid modifying the original data
            df = processed_data.copy()
            self.logger.debug(f"Working with data from {df.index.min()} to {df.index.max()}")

            # Convert any Decimal values to float
            if 'close' in df.columns:
                df['close'] = df['close'].apply(self._ensure_float)
                self.logger.debug("Converted close prices to float type")

            # Extract day of week
            df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6
            self.logger.debug("Extracted day of week information")

            # Calculate daily returns vectorized
            df['daily_return'] = df['close'].pct_change().fillna(0)
            self.logger.debug("Calculated daily returns")

            # Calculate statistics by day of week using groupby operations
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_stats = {}

            # Group by day of week and compute all statistics at once
            self.logger.debug("Computing day of week statistics")
            grouped_stats = df.groupby('day_of_week')['daily_return'].agg([
                ('mean_return', 'mean'),
                ('median_return', 'median'),
                ('volatility', 'std'),
                ('max_gain', 'max'),
                ('max_loss', 'min'),
                ('sample_size', 'count')
            ])

            # Add positive/negative day rates
            grouped_positive = df.groupby('day_of_week')['daily_return'].apply(lambda x: (x > 0).mean())
            grouped_negative = df.groupby('day_of_week')['daily_return'].apply(lambda x: (x < 0).mean())

            # Combine all stats into the day_stats dictionary
            for day_num in range(7):
                if day_num in grouped_stats.index:
                    day_name = day_names[day_num]
                    stats = grouped_stats.loc[day_num].to_dict()
                    stats['positive_days'] = grouped_positive.get(day_num, 0)
                    stats['negative_days'] = grouped_negative.get(day_num, 0)
                    day_stats[day_name] = stats

            self.logger.debug(f"Collected statistics for {len(day_stats)} days of the week")

            # Calculate weekly momentum patterns
            self.logger.debug("Calculating weekly momentum patterns")
            week_momentum = {}

            # Create a shifted dataframe for previous day returns
            df_shifted = df.copy()
            df_shifted['prev_day_return'] = df_shifted['daily_return'].shift(1)
            df_shifted['prev_day'] = (df_shifted.index - pd.Timedelta(days=1))

            # Loop through day pairs (still needed for proper date handling)
            for i in range(1, 7):  # Tuesday through Sunday
                prev_day = i - 1
                current_day = i
                self.logger.debug(f"Analyzing momentum from {day_names[prev_day]} to {day_names[current_day]}")

                # Create a merged dataset of adjacent days
                prev_day_df = df[df['day_of_week'] == prev_day][['daily_return']]
                prev_day_df.columns = ['prev_return']

                # Get next day dates
                prev_dates = prev_day_df.index
                pairs_data = []

                # This part is harder to fully vectorize due to calendar gaps
                for date in prev_dates:
                    next_date = date + pd.Timedelta(days=1)
                    if next_date in df.index:
                        pairs_data.append({
                            'prev_date': date,
                            'next_date': next_date,
                            'prev_return': df.loc[date, 'daily_return'],
                            'next_return': df.loc[next_date, 'daily_return']
                        })

                if pairs_data:
                    pairs_df = pd.DataFrame(pairs_data)

                    # Calculate all metrics at once
                    total_pairs = len(pairs_df)
                    self.logger.debug(
                        f"Found {total_pairs} day pairs for {day_names[prev_day]}-{day_names[current_day]}")

                    # Calculate correlation vectorized
                    correlation = np.corrcoef(
                        pairs_df['prev_return'].values,
                        pairs_df['next_return'].values
                    )[0, 1] if total_pairs > 1 else 0

                    # Calculate continuation and reversal rates vectorized
                    same_direction = ((pairs_df['prev_return'] > 0) & (pairs_df['next_return'] > 0)) | \
                                     ((pairs_df['prev_return'] < 0) & (pairs_df['next_return'] < 0))

                    continuation_rate = same_direction.mean() if total_pairs > 0 else 0
                    reversal_rate = 1 - continuation_rate if total_pairs > 0 else 0

                    week_momentum[f"{day_names[prev_day]} to {day_names[current_day]}"] = {
                        'correlation': correlation,
                        'continuation_rate': continuation_rate,
                        'reversal_rate': reversal_rate,
                        'sample_size': total_pairs
                    }

            # Calculate average week pattern vectorized
            self.logger.debug("Calculating average week pattern")
            avg_week_pattern = df.groupby('day_of_week')['close'].mean().to_dict()
            avg_week_pattern = {day_names[day_num]: value for day_num, value in avg_week_pattern.items()
                                if day_num in range(7)}

            # Calculate best/worst days vectorized
            best_day = None
            worst_day = None
            most_volatile_day = None
            weekend_to_weekday_return_ratio = 0

            if day_stats:
                self.logger.debug("Computing best and worst days")
                best_day = max(day_stats.items(), key=lambda x: x[1]['mean_return'])[0]
                worst_day = min(day_stats.items(), key=lambda x: x[1]['mean_return'])[0]
                most_volatile_day = max(day_stats.items(), key=lambda x: x[1]['volatility'])[0]

                # Calculate weekend effect
                self.logger.debug("Computing weekend effect")
                weekday_returns = [
                    self._ensure_float(day_stats.get(d, {}).get('mean_return', 0))
                    for d in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                ]
                weekend_returns = [
                    self._ensure_float(day_stats.get(d, {}).get('mean_return', 0))
                    for d in ['Saturday', 'Sunday']
                ]

                weekend_to_weekday_ratio_denominator = sum(weekday_returns)
                weekend_sum = sum(weekend_returns)

                try:
                    if weekend_to_weekday_ratio_denominator != 0:
                        weekend_to_weekday_return_ratio = weekend_sum / weekend_to_weekday_ratio_denominator
                    else:
                        self.logger.warning("Denominator is zero for weekend/weekday ratio calculation")
                        weekend_to_weekday_return_ratio = 0
                except (TypeError, decimal.InvalidOperation) as e:
                    self.logger.error(f"Error calculating weekend ratio: {e}")
                    self.logger.error(
                        f"weekend_sum: {type(weekend_sum)}, denominator: {type(weekend_to_weekday_ratio_denominator)}")
                    weekend_to_weekday_return_ratio = 0

            # Return the compiled statistics
            self.logger.info("Weekly cycle analysis completed successfully")
            return {
                'day_of_week_stats': day_stats,
                'weekly_momentum_patterns': week_momentum,
                'average_week_pattern': avg_week_pattern,
                'best_day': best_day,
                'worst_day': worst_day,
                'most_volatile_day': most_volatile_day,
                'weekend_effect': {
                    'fri_to_mon_correlation': week_momentum.get('Friday to Monday', {}).get('correlation', 0),
                    'weekend_to_weekday_return_ratio': weekend_to_weekday_return_ratio
                }
            }
        except Exception as e:
            self.logger.error(f"Error during weekly cycle analysis: {str(e)}", exc_info=True)
            raise

    def analyze_monthly_seasonality(self, processed_data: pd.DataFrame, years_back: int = 3) -> Dict:
        """
                Аналізує сезонність цін на актив протягом місяців і кварталів за останні роки.

                Параметри:
                ----------
                processed_data : pd.DataFrame
                    Фрейм даних із щоденними цінами активу. Має містити колонку 'close' і індекс типу DatetimeIndex.
                years_back : int, optional (default=3)
                    Кількість останніх років, які будуть включені до аналізу.

                Повертає:
                ---------
                Dict
                    Словник з результатами аналізу, що містить:
                    - 'monthly_stats': статистика по кожному місяцю:
                        * середнє та медіанне значення прибутковості,
                        * волатильність,
                        * максимальні та мінімальні місячні прибутки,
                        * частка позитивних та негативних місяців,
                        * річні значення прибутковості для кожного місяця.
                    - 'quarterly_stats': аналогічна статистика по кварталах.
                    - 'best_month': назва місяця з найвищим середнім прибутком.
                    - 'worst_month': назва місяця з найнижчим середнім прибутком.
                    - 'most_volatile_month': місяць з найбільшою волатильністю.
                    - 'january_effect': словник з перевіркою "ефекту січня", зокрема:
                        * річна прибутковість,
                        * січнева прибутковість,
                        * чи передбачила січнева прибутковість напрямок річного тренду,
                        * точність прогнозування.
                    - 'seasonality_significance':
                        * Z-оцінки місячної сезонності,
                        * статистично значущі місяці (|Z| > 1.96),
                        * сила сезонності (частка значущих місяців).
                    - 'years_analyzed': кількість років, використаних для аналізу.
                    - 'data_timespan': діапазон дат, охоплених аналізом (start_date, end_date).

                Винятки:
                --------
                ValueError
                    Якщо вхідні дані мають некоректний індекс або відсутню колонку 'close'.
                Exception
                    У випадку інших помилок при виконанні аналізу.
                """
        try:
            self.logger.info(f"Starting monthly seasonality analysis for past {years_back} years")
            # Ensure we have the required data
            if not isinstance(processed_data.index, pd.DatetimeIndex):
                self.logger.error("DataFrame index must be a DatetimeIndex")
                raise ValueError("DataFrame index must be a DatetimeIndex")

            if 'close' not in processed_data.columns:
                self.logger.error("DataFrame must have a 'close' column")
                raise ValueError("DataFrame must have a 'close' column")

            # Create a copy to avoid modifying the original data
            df = processed_data.copy()

            # Convert any Decimal values to float
            if 'close' in df.columns:
                df['close'] = df['close'].apply(self._ensure_float)
                self.logger.debug("Converted close prices to float type")

            # Filter data based on years_back parameter - vectorized
            cutoff_date = df.index.max() - pd.DateOffset(years=years_back)
            df = df[df.index >= cutoff_date]
            self.logger.debug(f"Filtered data to {df.shape[0]} rows from {df.index.min()} to {df.index.max()}")

            # Extract month and year - vectorized
            df['month'] = df.index.month
            df['year'] = df.index.year
            df['year_month'] = df['year'] * 100 + df['month']  # Create year-month identifier

            # Calculate monthly returns for each year-month group
            self.logger.debug("Calculating monthly returns")
            monthly_returns_data = []

            # Group data by year and month
            for (year, month), group in df.groupby(['year', 'month']):
                if len(group) > 0:
                    try:
                        first_price = self._ensure_float(group['close'].iloc[0])
                        last_price = self._ensure_float(group['close'].iloc[-1])
                        month_return = (last_price / first_price) - 1

                        monthly_returns_data.append({
                            'year': year,
                            'month': month,
                            'month_return': month_return,
                            'start_date': group.index.min(),
                            'end_date': group.index.max()
                        })
                    except (TypeError, decimal.InvalidOperation, ZeroDivisionError) as e:
                        self.logger.error(f"Error calculating monthly return for {year}-{month}: {e}")
                        self.logger.error(f"first_price: {type(first_price)}, last_price: {type(last_price)}")

            # Convert to DataFrame for vectorized operations
            monthly_df = pd.DataFrame(monthly_returns_data)
            self.logger.debug(f"Created monthly returns dataframe with {monthly_df.shape[0]} rows")

            # Calculate statistics by month - vectorized
            month_stats = {}
            month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                           'July', 'August', 'September', 'October', 'November', 'December']

            if not monthly_df.empty:
                # Group by month and compute statistics at once
                self.logger.debug("Computing monthly statistics")
                grouped_monthly = monthly_df.groupby('month')['month_return'].agg([
                    ('mean_return', 'mean'),
                    ('median_return', 'median'),
                    ('volatility', 'std'),
                    ('max_gain', 'max'),
                    ('max_loss', 'min'),
                    ('sample_size', 'count')
                ])

                # Add positive/negative month rates
                grouped_positive = monthly_df.groupby('month')['month_return'].apply(lambda x: (x > 0).mean())
                grouped_negative = monthly_df.groupby('month')['month_return'].apply(lambda x: (x < 0).mean())

                # Generate returns by year for each month
                returns_by_year = {}
                for month_num in range(1, 13):
                    month_data = monthly_df[monthly_df['month'] == month_num]
                    if not month_data.empty:
                        returns_by_year[month_num] = dict(zip(month_data['year'], month_data['month_return']))

                # Combine all stats into the month_stats dictionary
                for month_num in range(1, 13):
                    if month_num in grouped_monthly.index:
                        month_name = month_names[month_num - 1]
                        stats = grouped_monthly.loc[month_num].to_dict()
                        stats['positive_months'] = grouped_positive.get(month_num, 0)
                        stats['negative_months'] = grouped_negative.get(month_num, 0)
                        stats['returns_by_year'] = returns_by_year.get(month_num, {})
                        month_stats[month_name] = stats

                self.logger.debug(f"Collected statistics for {len(month_stats)} months")

            # Calculate quarterly statistics - vectorized
            self.logger.debug("Computing quarterly statistics")
            quarters = {
                'Q1': [1, 2, 3],  # Jan-Mar
                'Q2': [4, 5, 6],  # Apr-Jun
                'Q3': [7, 8, 9],  # Jul-Sep
                'Q4': [10, 11, 12]  # Oct-Dec
            }

            quarter_stats = {}

            # First create a quarterly grouper
            df['quarter'] = pd.cut(
                df['month'],
                bins=[0, 3, 6, 9, 12],
                labels=['Q1', 'Q2', 'Q3', 'Q4'],
                right=True,
                include_lowest=False
            )

            # Calculate quarterly returns
            quarterly_returns_data = []

            # We still need to loop through year-quarters due to calendar considerations
            for (year, quarter), group in df.groupby(['year', 'quarter']):
                if len(group) > 0:
                    try:
                        first_date = group.index.min()
                        last_date = group.index.max()

                        first_price = self._ensure_float(df.loc[first_date, 'close'])
                        last_price = self._ensure_float(df.loc[last_date, 'close'])

                        quarter_return = (last_price / first_price) - 1

                        quarterly_returns_data.append({
                            'year': year,
                            'quarter': quarter,
                            'quarter_return': quarter_return
                        })
                    except (TypeError, decimal.InvalidOperation, ZeroDivisionError) as e:
                        self.logger.error(f"Error calculating quarterly return for {year}-{quarter}: {e}")
                        self.logger.error(f"first_price: {type(first_price)}, last_price: {type(last_price)}")

            # Convert to DataFrame for vectorized operations
            quarterly_df = pd.DataFrame(quarterly_returns_data)
            self.logger.debug(f"Created quarterly returns dataframe with {quarterly_df.shape[0]} rows")

            if not quarterly_df.empty:
                # Group by quarter and compute statistics vectorized
                grouped_quarterly = quarterly_df.groupby('quarter')['quarter_return'].agg([
                    ('mean_return', 'mean'),
                    ('median_return', 'median'),
                    ('volatility', 'std'),
                    ('max_gain', 'max'),
                    ('max_loss', 'min'),
                    ('sample_size', 'count')
                ])

                # Add positive/negative quarter rates
                grouped_qtr_positive = quarterly_df.groupby('quarter')['quarter_return'].apply(lambda x: (x > 0).mean())
                grouped_qtr_negative = quarterly_df.groupby('quarter')['quarter_return'].apply(lambda x: (x < 0).mean())

                # Combine all stats into the quarter_stats dictionary
                for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
                    if quarter in grouped_quarterly.index:
                        stats = grouped_quarterly.loc[quarter].to_dict()
                        stats['positive_quarters'] = grouped_qtr_positive.get(quarter, 0)
                        stats['negative_quarters'] = grouped_qtr_negative.get(quarter, 0)
                        quarter_stats[quarter] = stats

            # Check for January effect - vectorized approach
            self.logger.debug("Analyzing January effect")
            january_effect = {}
            jan_effect_accuracy = 0

            if not monthly_df.empty:
                # Get January data
                jan_data = monthly_df[monthly_df['month'] == 1]

                # Calculate full year returns
                yearly_returns_data = []

                for year in jan_data['year'].unique():
                    year_data = df[df['year'] == year]
                    if len(year_data) > 0:
                        try:
                            first_price = self._ensure_float(year_data['close'].iloc[0])
                            last_price = self._ensure_float(year_data['close'].iloc[-1])
                            year_return = (last_price / first_price) - 1

                            yearly_returns_data.append({
                                'year': year,
                                'full_year_return': year_return
                            })
                        except (TypeError, decimal.InvalidOperation, ZeroDivisionError) as e:
                            self.logger.error(f"Error calculating yearly return for {year}: {e}")
                            self.logger.error(f"first_price: {type(first_price)}, last_price: {type(last_price)}")

                # Convert to DataFrame
                yearly_df = pd.DataFrame(yearly_returns_data)

                # Merge January data with yearly data
                if not yearly_df.empty:
                    jan_year_df = pd.merge(
                        jan_data[['year', 'month_return']],
                        yearly_df[['year', 'full_year_return']],
                        on='year'
                    )

                    # Rename for clarity
                    jan_year_df = jan_year_df.rename(columns={'month_return': 'january_return'})

                    # Calculate if January predicted year direction
                    jan_year_df['january_predicted_year'] = (
                            (jan_year_df['january_return'] > 0) & (jan_year_df['full_year_return'] > 0) |
                            (jan_year_df['january_return'] < 0) & (jan_year_df['full_year_return'] < 0)
                    )

                    # Convert to dictionary
                    january_effect = {
                        row['year']: {
                            'january_return': row['january_return'],
                            'full_year_return': row['full_year_return'],
                            'january_predicted_year': row['january_predicted_year']
                        }
                        for _, row in jan_year_df.iterrows()
                    }

                    # Calculate January effect accuracy
                    jan_effect_accuracy = jan_year_df['january_predicted_year'].mean() if len(jan_year_df) > 0 else 0
                    self.logger.debug(f"January effect accuracy: {jan_effect_accuracy:.2f}")

            # Calculate seasonality significance - vectorized
            self.logger.debug("Calculating seasonality significance")
            monthly_std = 0
            z_scores = {}
            significant_months = {}
            seasonality_strength = 0

            if month_stats:
                # Extract mean returns for each month
                mean_returns_by_month = {month: stats['mean_return'] for month, stats in month_stats.items()}

                # Calculate standard deviation vectorized
                monthly_means = np.array(list(mean_returns_by_month.values()))
                monthly_std = np.std(monthly_means)
                mean_of_means = np.mean(monthly_means)

                # Calculate z-scores vectorized
                z_scores = {
                    month: (mean - mean_of_means) / monthly_std if monthly_std > 0 else 0
                    for month, mean in mean_returns_by_month.items()
                }

                # Find significant months vectorized
                significant_months = {month: score for month, score in z_scores.items() if abs(score) > 1.96}
                seasonality_strength = len(significant_months) / len(month_stats) if month_stats else 0
                self.logger.debug(f"Found {len(significant_months)} statistically significant months")

            # Calculate best/worst months vectorized
            best_month = None
            worst_month = None
            most_volatile_month = None

            if month_stats:
                self.logger.debug("Computing best and worst months")
                best_month = max(month_stats.items(), key=lambda x: x[1]['mean_return'])[0]
                worst_month = min(month_stats.items(), key=lambda x: x[1]['mean_return'])[0]
                most_volatile_month = max(month_stats.items(), key=lambda x: x[1]['volatility'])[0]

            # Return the compiled statistics
            self.logger.info("Monthly seasonality analysis completed successfully")
            return {
                'monthly_stats': month_stats,
                'quarterly_stats': quarter_stats,
                'best_month': best_month,
                'worst_month': worst_month,
                'most_volatile_month': most_volatile_month,
                'january_effect': {
                    'accuracy': jan_effect_accuracy,
                    'yearly_data': january_effect
                },
                'seasonality_significance': {
                    'monthly_z_scores': z_scores,
                    'significant_months': significant_months,
                    'seasonality_strength': seasonality_strength
                },
                'years_analyzed': years_back,
                'data_timespan': {
                    'start_date': df.index.min().strftime('%Y-%m-%d') if not df.empty else None,
                    'end_date': df.index.max().strftime('%Y-%m-%d') if not df.empty else None
                }
            }
        except Exception as e:
            self.logger.error(f"Error during monthly seasonality analysis: {str(e)}", exc_info=True)
            raise