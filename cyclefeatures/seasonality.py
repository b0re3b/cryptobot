from typing import Dict
import numpy as np

import pandas as pd


class TemporalSeasonalityAnalyzer:
    def __init__(self, market_features):
        self.market_features = market_features

    def analyze_weekly_cycle(self, processed_data: pd.DataFrame) -> Dict:
        # Ensure we have the required data
        if not isinstance(processed_data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        if 'close' not in processed_data.columns:
            raise ValueError("DataFrame must have a 'close' column")

        # Create a copy to avoid modifying the original data
        df = processed_data.copy()

        # Extract day of week
        df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6

        # Calculate daily returns vectorized
        df['daily_return'] = df['close'].pct_change().fillna(0)

        # Calculate statistics by day of week using groupby operations
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_stats = {}

        # Group by day of week and compute all statistics at once
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

        # Calculate weekly momentum patterns
        week_momentum = {}

        # Create a shifted dataframe for previous day returns
        df_shifted = df.copy()
        df_shifted['prev_day_return'] = df_shifted['daily_return'].shift(1)
        df_shifted['prev_day'] = (df_shifted.index - pd.Timedelta(days=1))

        # Loop through day pairs (still needed for proper date handling)
        for i in range(1, 7):  # Tuesday through Sunday
            prev_day = i - 1
            current_day = i

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
        avg_week_pattern = df.groupby('day_of_week')['close'].mean().to_dict()
        avg_week_pattern = {day_names[day_num]: value for day_num, value in avg_week_pattern.items()
                            if day_num in range(7)}

        # Calculate best/worst days vectorized
        if day_stats:
            best_day = max(day_stats.items(), key=lambda x: x[1]['mean_return'])[0]
            worst_day = min(day_stats.items(), key=lambda x: x[1]['mean_return'])[0]
            most_volatile_day = max(day_stats.items(), key=lambda x: x[1]['volatility'])[0]

            # Calculate weekend effect
            weekend_to_weekday_ratio_denominator = sum(
                day_stats.get(d, {}).get('mean_return', 0)
                for d in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            )

            weekend_to_weekday_return_ratio = (
                (day_stats.get('Saturday', {}).get('mean_return', 0) +
                 day_stats.get('Sunday', {}).get('mean_return', 0))
                / weekend_to_weekday_ratio_denominator if weekend_to_weekday_ratio_denominator != 0 else 0
            )
        else:
            best_day = None
            worst_day = None
            most_volatile_day = None
            weekend_to_weekday_return_ratio = 0

        # Return the compiled statistics
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

    def analyze_monthly_seasonality(self, processed_data: pd.DataFrame, years_back: int = 3) -> Dict:
        # Ensure we have the required data
        if not isinstance(processed_data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        if 'close' not in processed_data.columns:
            raise ValueError("DataFrame must have a 'close' column")

        # Create a copy to avoid modifying the original data
        df = processed_data.copy()

        # Filter data based on years_back parameter - vectorized
        cutoff_date = df.index.max() - pd.DateOffset(years=years_back)
        df = df[df.index >= cutoff_date]

        # Extract month and year - vectorized
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['year_month'] = df['year'] * 100 + df['month']  # Create year-month identifier

        # Calculate monthly returns for each year-month group
        # Create a list to store the monthly returns data
        monthly_returns_data = []

        # Group data by year and month
        for (year, month), group in df.groupby(['year', 'month']):
            if len(group) > 0:
                first_price = group['close'].iloc[0]
                last_price = group['close'].iloc[-1]
                month_return = (last_price / first_price) - 1

                monthly_returns_data.append({
                    'year': year,
                    'month': month,
                    'month_return': month_return,
                    'start_date': group.index.min(),
                    'end_date': group.index.max()
                })

        # Convert to DataFrame for vectorized operations
        monthly_df = pd.DataFrame(monthly_returns_data)

        # Calculate statistics by month - vectorized
        month_stats = {}
        month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']

        if not monthly_df.empty:
            # Group by month and compute statistics at once
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

        # Calculate quarterly statistics - vectorized
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
                first_date = group.index.min()
                last_date = group.index.max()

                first_price = df.loc[first_date, 'close']
                last_price = df.loc[last_date, 'close']

                quarter_return = (last_price / first_price) - 1

                quarterly_returns_data.append({
                    'year': year,
                    'quarter': quarter,
                    'quarter_return': quarter_return
                })

        # Convert to DataFrame for vectorized operations
        quarterly_df = pd.DataFrame(quarterly_returns_data)

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
        january_effect = {}

        if not monthly_df.empty:
            # Get January data
            jan_data = monthly_df[monthly_df['month'] == 1]

            # Calculate full year returns
            yearly_returns_data = []

            for year in jan_data['year'].unique():
                year_data = df[df['year'] == year]
                if len(year_data) > 0:
                    first_price = year_data['close'].iloc[0]
                    last_price = year_data['close'].iloc[-1]
                    year_return = (last_price / first_price) - 1

                    yearly_returns_data.append({
                        'year': year,
                        'full_year_return': year_return
                    })

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
            else:
                jan_effect_accuracy = 0
        else:
            jan_effect_accuracy = 0

        # Calculate seasonality significance - vectorized
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
        else:
            monthly_std = 0
            z_scores = {}
            significant_months = {}
            seasonality_strength = 0

        # Calculate best/worst months vectorized
        if month_stats:
            best_month = max(month_stats.items(), key=lambda x: x[1]['mean_return'])[0]
            worst_month = min(month_stats.items(), key=lambda x: x[1]['mean_return'])[0]
            most_volatile_month = max(month_stats.items(), key=lambda x: x[1]['volatility'])[0]
        else:
            best_month = None
            worst_month = None
            most_volatile_month = None

        # Return the compiled statistics
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