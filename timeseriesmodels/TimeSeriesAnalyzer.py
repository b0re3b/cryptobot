# Файл time_series_analyzer.py
from typing import Dict

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose


class TimeSeriesAnalyzer:
    def __init__(self, logger):
        self.logger = logger



    def detect_seasonality(self, data: pd.Series) -> Dict:

        self.logger.info("Початок аналізу сезонності")

        # Перевірка на наявність пропущених значень
        if data.isnull().any():
            self.logger.warning("Дані містять пропущені значення (NaN). Видаляємо їх перед аналізом.")
            data = data.dropna()

        # Перевірка достатньої кількості точок даних
        if len(data) < 24:
            self.logger.error("Недостатньо даних для надійного виявлення сезонності (мінімум 24 точки)")
            return {
                "status": "error",
                "message": "Недостатньо даних для надійного виявлення сезонності",
                "has_seasonality": False,
                "seasonal_periods": [],
                "details": {}
            }

        # Ініціалізація результату
        result = {
            "status": "success",
            "message": "Аналіз сезонності завершено",
            "has_seasonality": False,
            "seasonal_periods": [],
            "details": {}
        }

        try:

            # 2. Автокореляційний аналіз
            from statsmodels.tsa.stattools import acf, pacf

            # Обчислюємо максимальну кількість лагів (до половини довжини ряду або 50)
            max_lags = min(len(data) // 2, 50)

            # Обчислюємо ACF (автокореляційна функція)
            acf_values = acf(data, nlags=max_lags, fft=True)

            # Обчислюємо PACF (часткова автокореляційна функція)
            pacf_values = pacf(data, nlags=max_lags)

            # 3. Пошук значимих лагів у ACF
            # Поріг значимості (зазвичай 1.96/sqrt(n) для 95% довірчого інтервалу)
            significance_threshold = 1.96 / np.sqrt(len(data))

            # Знаходимо значимі лаги (з автокореляцією вище порогу)
            significant_lags = [lag for lag in range(2, len(acf_values))
                                if abs(acf_values[lag]) > significance_threshold]

            result["details"]["acf_analysis"] = {
                "significant_lags": significant_lags,
                "significance_threshold": significance_threshold
            }

            # 4. Типові сезонні періоди для фінансових даних
            typical_periods = [7, 14, 30, 90, 365]  # Тижневий, двотижневий, місячний, квартальний, річний

            # Визначаємо потенційні сезонні періоди зі значимих лагів
            potential_seasonal_periods = []

            # Шукаємо локальні піки в ACF як потенційні сезонні періоди
            for lag in range(2, len(acf_values) - 1):
                if (acf_values[lag] > acf_values[lag - 1] and
                        acf_values[lag] > acf_values[lag + 1] and
                        abs(acf_values[lag]) > significance_threshold):
                    potential_seasonal_periods.append({
                        "lag": lag,
                        "acf_value": acf_values[lag],
                        "strength": abs(acf_values[lag]) / abs(acf_values[0])  # Відносна сила
                    })

            # Сортуємо за силою кореляції
            potential_seasonal_periods.sort(key=lambda x: x["strength"], reverse=True)

            # 5. Декомпозиція часового ряду
            try:
                # Визначаємо період для декомпозиції
                if len(potential_seasonal_periods) > 0:
                    decomposition_period = potential_seasonal_periods[0]["lag"]
                else:
                    # Використовуємо найбільш ймовірний період з типових
                    for period in typical_periods:
                        if period < len(data) // 2:
                            decomposition_period = period
                            break
                    else:
                        decomposition_period = min(len(data) // 4, 7)  # Резервний варіант

                # Перевіряємо, що період більше 1 і підходить для декомпозиції
                if decomposition_period < 2:
                    decomposition_period = 2

                # Сезонна декомпозиція
                decomposition = seasonal_decompose(
                    data,
                    model='additive',
                    period=decomposition_period,
                    extrapolate_trend='freq'
                )

                # Витягуємо сезонний компонент
                seasonal_component = decomposition.seasonal

                # Розраховуємо силу сезонності як відношення дисперсії сезонного компоненту до загальної дисперсії
                seasonal_strength = np.var(seasonal_component) / np.var(data)

                result["details"]["decomposition"] = {
                    "period_used": decomposition_period,
                    "seasonal_strength": seasonal_strength,
                    "model": "additive"
                }

                # Якщо сила сезонності значна, вважаємо що ряд має сезонність
                if seasonal_strength > 0.1:  # Поріг 10%
                    result["has_seasonality"] = True

            except Exception as e:
                self.logger.warning(f"Помилка під час сезонної декомпозиції: {str(e)}")
                result["details"]["decomposition"] = {
                    "error": str(e)
                }

            # 6. Тест на наявність сезонності за допомогою спектрального аналізу
            try:
                from scipy import signal

                # Створюємо рівномірні часові точки для спектрального аналізу
                if not isinstance(data.index, pd.DatetimeIndex):
                    t = np.arange(len(data))
                else:
                    # Для часового індексу конвертуємо в дні від початку
                    t = (data.index - data.index[0]).total_seconds() / (24 * 3600)

                # Розраховуємо спектр за допомогою періодограми
                freqs, spectrum = signal.periodogram(data.values, fs=1.0)

                # Виключаємо нульову частоту (постійний компонент)
                freqs = freqs[1:]
                spectrum = spectrum[1:]

                # Знаходимо піки в спектрі
                peaks, _ = signal.find_peaks(spectrum, height=np.max(spectrum) / 10)

                if len(peaks) > 0:
                    # Конвертуємо частоти в періоди (періоди = 1/частота)
                    peak_periods = [round(1.0 / freqs[p]) for p in peaks if freqs[p] > 0]

                    # Відфільтровуємо занадто великі або малі періоди
                    filtered_periods = [p for p in peak_periods if 2 <= p <= len(data) // 3]

                    result["details"]["spectral_analysis"] = {
                        "detected_periods": filtered_periods,
                        "peak_count": len(peaks)
                    }

                    # Доповнюємо список можливих сезонних періодів
                    for period in filtered_periods:
                        if period not in [p["lag"] for p in potential_seasonal_periods]:
                            potential_seasonal_periods.append({
                                "lag": period,
                                "source": "spectral",
                                "strength": 0.8  # Приблизна оцінка сили
                            })

                else:
                    result["details"]["spectral_analysis"] = {
                        "detected_periods": [],
                        "peak_count": 0
                    }

            except Exception as e:
                self.logger.warning(f"Помилка під час спектрального аналізу: {str(e)}")
                result["details"]["spectral_analysis"] = {
                    "error": str(e)
                }

            # 7. Формуємо підсумковий список сезонних періодів з оцінкою впевненості
            seasonal_periods = []

            # Сумуємо всі знайдені потенційні періоди
            for period in potential_seasonal_periods:
                lag = period["lag"]

                # Розраховуємо впевненість на основі сили та інших факторів
                confidence = period.get("strength", 0.5)

                # Підвищуємо впевненість, якщо період відповідає типовим
                if any(abs(lag - typical) / typical < 0.1 for typical in typical_periods):
                    confidence += 0.2

                # Підвищуємо впевненість, якщо період підтверджено кількома методами
                sources = []
                if "acf_value" in period:
                    sources.append("acf")
                if period.get("source") == "spectral":
                    sources.append("spectral")
                if result.get("has_seasonality") and abs(
                        lag - result["details"]["decomposition"].get("period_used", 0)) < 2:
                    sources.append("decomposition")

                confidence = min(confidence + 0.1 * (len(sources) - 1), 1.0)  # Максимум 1.0

                seasonal_periods.append({
                    "period": lag,
                    "confidence": confidence,
                    "sources": sources
                })

            # Сортуємо за впевненістю
            seasonal_periods.sort(key=lambda x: x["confidence"], reverse=True)

            # Видаляємо дублікати та близькі періоди (залишаємо більш впевнений)
            filtered_periods = []
            for period in seasonal_periods:
                if not any(abs(period["period"] - existing["period"]) / existing["period"] < 0.1
                           for existing in filtered_periods):
                    filtered_periods.append(period)

            # Записуємо в результат
            result["seasonal_periods"] = filtered_periods

            # Визначаємо наявність сезонності на основі всіх факторів
            if len(filtered_periods) > 0 and filtered_periods[0]["confidence"] > 0.7:
                result["has_seasonality"] = True
                result["primary_period"] = filtered_periods[0]["period"]

            self.logger.info(f"Аналіз сезонності завершено: {result['has_seasonality']}")
            if result["has_seasonality"]:
                self.logger.info(f"Основний сезонний період: {result.get('primary_period')}")

            return result

        except Exception as e:
            self.logger.error(f"Помилка під час аналізу сезонності: {str(e)}")
            return {
                "status": "error",
                "message": f"Помилка під час аналізу сезонності: {str(e)}",
                "has_seasonality": False,
                "seasonal_periods": [],
                "details": {}
            }

    def find_optimal_params(self, data: pd.Series, max_p: int = 5, max_d: int = 2,
                            max_q: int = 5, seasonal: bool = False) -> Dict:

        self.logger.info("Starting optimal parameters search")

        if data.isnull().any():
            self.logger.warning("Data contains NaN values. Removing them before parameter search.")
            data = data.dropna()

        if len(data) < 10:
            self.logger.error("Not enough data points for parameter search")
            return {
                "status": "error",
                "message": "Not enough data points for parameter search",
                "parameters": None,
                "model_info": None
            }

        try:
            # Використовуємо auto_arima для автоматичного пошуку параметрів
            if seasonal:
                # Для SARIMA: визначаємо можливий сезонний період
                # Типові значення: щотижнева (7), щомісячна (30/31), квартальна (4)
                seasonal_periods = [7, 12, 24, 30, 365]

                # Автовизначення сезонного періоду, якщо достатньо даних
                if len(data) >= 2 * max(seasonal_periods):
                    # Аналізуємо автокореляцію для виявлення можливого сезонного періоду
                    from statsmodels.tsa.stattools import acf
                    acf_values = acf(data, nlags=max(seasonal_periods))

                    # Шукаємо піки в ACF, які можуть вказувати на сезонність
                    potential_seasons = []
                    for period in seasonal_periods:
                        if period < len(acf_values) and acf_values[period] > 0.2:  # Поріг кореляції
                            potential_seasons.append((period, acf_values[period]))

                    # Вибираємо період з найсильнішою автокореляцією
                    if potential_seasons:
                        potential_seasons.sort(key=lambda x: x[1], reverse=True)
                        seasonal_period = potential_seasons[0][0]
                        self.logger.info(f"Detected potential seasonal period: {seasonal_period}")
                    else:
                        # За замовчуванням
                        seasonal_period = 7  # Тижнева сезонність для фінансових даних
                        self.logger.info(f"No strong seasonality detected, using default: {seasonal_period}")
                else:
                    seasonal_period = 7
                    self.logger.info(f"Not enough data for seasonal detection, using default: {seasonal_period}")

                # Запускаємо auto_arima з урахуванням сезонності
                model = auto_arima(
                    data,
                    start_p=0, max_p=max_p,
                    start_q=0, max_q=max_q,
                    max_d=max_d,
                    start_P=0, max_P=2,
                    start_Q=0, max_Q=2,
                    max_D=1,
                    m=seasonal_period,
                    seasonal=True,
                    trace=True,  # Виведення інформації про процес пошуку
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True,  # Покроковий пошук для прискорення
                    information_criterion='aic',  # AIC або BIC як критерій
                    random_state=42
                )

                order = model.order
                seasonal_order = model.seasonal_order

                result = {
                    "status": "success",
                    "message": "Optimal parameters found",
                    "parameters": {
                        "order": order,
                        "seasonal_order": seasonal_order,
                        "seasonal_period": seasonal_period
                    },
                    "model_info": {
                        "aic": model.aic(),
                        "bic": model.bic(),
                        "model_type": "SARIMA"
                    }
                }
            else:
                # Для несезонної ARIMA
                model = auto_arima(
                    data,
                    start_p=0, max_p=max_p,
                    start_q=0, max_q=max_q,
                    max_d=max_d,
                    seasonal=False,
                    trace=True,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True,
                    information_criterion='aic',
                    random_state=42
                )

                order = model.order

                result = {
                    "status": "success",
                    "message": "Optimal parameters found",
                    "parameters": {
                        "order": order
                    },
                    "model_info": {
                        "aic": model.aic(),
                        "bic": model.bic(),
                        "model_type": "ARIMA"
                    }
                }

            self.logger.info(f"Found optimal parameters: {result['parameters']}")
            return result

        except Exception as e:
            self.logger.error(f"Error during parameter search: {str(e)}")
            return {
                "status": "error",
                "message": f"Error during parameter search: {str(e)}",
                "parameters": None,
                "model_info": None
            }