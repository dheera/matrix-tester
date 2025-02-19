from quoter import Quoter

# Fake quoter that uses minute aggregates +/- slippage to generate bid and ask prices

class AggregateQuoter(Quoter):
    def __init__(self, data, slippage = 0.0):
        self.data = data
        self.slippage = slippage

    def get_bid_ask(self, timestamp, ticker):
        d = self.data.iloc[timestamp]
        return (d[ticker, "close"] - self.slippage, d[ticker, "close"] + self.slippage)
