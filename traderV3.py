import math
import random
from tabnanny import verbose
from turtle import color
import yfinance as yf
import numpy as np
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
# from LSTM import model

global inventory
global funds 
inventory = []
funds = 10000

class trader:

    def __init__(self , symbol, start, end, availiableFunds):
        
        self.symbol = symbol 
        self.start = start
        self.end = end
        self.availiableFunds = availiableFunds

        self.buySitSell()
        #return self.buySitSell(), self.availiableFunds

    # def getInventory(self):
    #     return self.inventory

    # def getFunds(self):
    #     return self.availiableFunds

    #https://www.investopedia.com/terms/s/stochasticoscillator.asp
    #If value over 80: stock overbought. Under 20: stock oversold.
    def Stoc_Osc(self, stock_data):

        C = stock_data['Close'][-1]
        L14 = np.min(stock_data['Low'])
        H14 = np.max(stock_data['High'])

        return 100*(C - L14)/(H14-L14)

    #https://www.investopedia.com/terms/m/macd.asp
    def MACD(self, price, slow, fast, smooth):
        
        exp1 = price.ewm(span = fast, adjust = False).mean()
        exp2 = price.ewm(span = slow, adjust = False).mean()
        macd = exp1 - exp2

        signal = macd.ewm(span = smooth, adjust = False).mean()
        hist = macd - signal

        
        #If hist > 0 Buy, if hist < 0 sell

        if hist[-1] > 0 and hist[-2] < 0: #Buy 
            return 1
        elif hist[-1] < 0 and hist[-2] > 0: #Sell
            #print(hist[-1], hist[-2], self.symbol in inventory, self.symbol, inventory)
            return -1
        else:
            return 0
        
    def SMA(self, stock_data):
        sma50 = np.mean(stock_data[len(stock_data) - 50:])
        sma200 = np.mean(stock_data[len(stock_data) - 200:])

        if sma50/sma200 < 1.01: #Should Buy
            return 1
        elif sma50/sma200 < 0.99: #Should Sell
            return -1
        else: #Should Sit
            return 0

    def getFactors(self, start, end):
        # today = date.today()
        # start = today - timedelta(days = 300)

        #CHeck 50 day MAV and 100 day MAV

        stock_data = yf.download(self.symbol, start= start, end= end, progress = False)

        if len(stock_data) < 1:
            return 0,0,0

        #FACTORS NEEDED
        # RSI , MFI, MACD, Bollinger Bands, Stochastics

        #sma50 = np.mean(stock_data['Close'][len(stock_data) - 50:])
        #sma200 = np.mean(stock_data['Close'][len(stock_data) - 200:])

        #K = self.Stoc_Osc(stock_data[len(stock_data) - 14:])

        sma = self.SMA(stock_data['Close'])
        macd = self.MACD(stock_data['Close'], 12, 26, 2)

        #print(macd, K, sma50, sma200)

        return macd, sma, stock_data['Close'][-1] #sma50, sma200, K
 
    def buySitSell(self):
        #1 is Buy, 0 is Sit, -1 is Sell 
        global funds
        #sma50, sma200, K, 
        macd, sma, currPrice = self.getFactors(self.start, self.end)

        x = sma#np.mean(sma,macd)
        #action = 0
        if self.symbol not in inventory and x == 1 and funds >= currPrice:
            inventory.append(self.symbol)
            #action = 1
            funds -= currPrice
            print(f'Bought {self.symbol} for {currPrice}. Funds Remaining: ${funds}')
        if self.symbol in inventory and x == -1:
            inventory.remove(self.symbol)
            #action = -1
            funds += currPrice
            print(f'Sold {self.symbol} for {currPrice}. Funds Remaining: ${funds}')

        #return action

def simulate(days):
    symbols = generateList()
    #random.shuffle(symbols)
    today = date.today() - timedelta(days = 1)
    startingFunds = funds
    for i in range(days, 0, -1):
        for symbol in symbols[:10]:
            end = today - timedelta(days = i)
            start = end - timedelta(days = i)

            t = trader(symbol, start, end, funds)

    profit = funds - startingFunds
    print(inventory)
    for symbol in inventory:
        ticker = yf.Ticker(symbol)
        currPrice = ticker.history(period='1d')['Close'][0]
        print(currPrice, symbol)
        profit += currPrice
        inventory.remove(symbol)

    print(f'Profit: {profit}')
    print(f'Stocks being held: {inventory}')

def generateList():
    payload=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    first_table = payload[0]
    second_table = payload[1]

    df = first_table

    symbols = df['Symbol'].values.tolist()
    return symbols

def main():
    simulate(100)

if __name__ == '__main__':
    main()