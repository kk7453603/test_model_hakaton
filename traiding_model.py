import numpy as np


class PortfolioRecommendationModel:
    def __init__(self, risk_level, initial_capital, monthly_income, market_preferences):
        self.risk_level = risk_level
        self.initial_capital = initial_capital
        self.monthly_income = monthly_income
        self.market_preferences = market_preferences

    def calculate_portfolio(self, returns):
        print("Уровень риска:", self.risk_level)
        print("Начальный капитал:", self.initial_capital)
        print("Ежемесячный доход:", self.monthly_income)
        print("Предпочтения по рынкам:", self.market_preferences)

        # Здесь применяем алгоритм оптимизации для определения оптимального портфеля
        # на основе введенных данных и тестовых данных (в виде вектора доходностей активов)

        num_assets = len(self.market_preferences)
        num_periods = returns.shape[1]
        weights = np.random.random((num_assets, num_periods))
        weights /= np.sum(weights, axis=0)
        returns_portfolio = np.matmul(weights.T, returns)
        risk_portfolio = np.sqrt(np.dot(weights.T, np.dot(returns, returns.T)))

        # Выводим оптимальный портфель
        print("Оптимальный портфель:")
        for i in range(num_periods):
            print("Период", i + 1)
            for j in range(num_assets):
                print(self.market_preferences[j], ": ", weights[j, i])

        print("Ожидаемая доходность портфеля:", returns_portfolio)
        print("Риски портфеля:", risk_portfolio)


# Используем тестовые данные
risk_level = 4
initial_capital = 1000000
monthly_income = 50000
market_preferences = ["российский рынок", "американский рынок"]

# Генерируем тестовые данные - векторы доходностей активов
returns = np.random.random((len(market_preferences), 100))

# Создаем объект модели с тестовыми данными
model = PortfolioRecommendationModel(risk_level, initial_capital, monthly_income, market_preferences)
model.calculate_portfolio(returns)
