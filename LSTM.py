import torch
import torch.nn as nn
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#Design different models using pytorch

#Functions needed

#train test split
def splitData(X, Y, trainSize):
    x_train = X[:int(X.shape[0]*trainSize)]
    y_train = Y[:int(X.shape[0]*trainSize)]

    x_test = X[int(X.shape[0]*trainSize):]
    y_test = Y[int(X.shape[0]*trainSize):]

    return x_train, x_test, y_train, y_test

#makeModel
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

#trainModel
def trainModel(model, num_epochs, x_train, y_train):

    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Training Model ...\n")
    for t in range(num_epochs):
        y_pred = model(x_train)

        loss = criterion(y_pred, y_train)
        # print("Epoch ", t, "MSE: ", loss.item())

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    return model, y_pred

def testModel(model, x_test, y_test):
    y_pred = np.zeros(y_test.shape)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    with torch.no_grad():
        y_pred = model(x_test)
        loss = criterion(y_pred, y_test)
        print("Test Loss: ", loss.item() )

    
    return y_pred

def makeData():
    ticker = 'AAPL'
    data = pd.DataFrame(yf.Ticker(ticker).history(period ='5y'))
    data.sort_values('Date')
    price = data[['Close']]
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))
    price = price.to_numpy()

    X = []
    y = []
    timeframe = 10
    for i in range(len(data)-timeframe):
        X.append(price[i: i + timeframe])
        y.append(price[i+timeframe])

    X = np.array(X)
    # X = np.reshape(X,(X.shape[0], X.shape[1]))
    y = np.array(y)
    # y = np.reshape(y,(y.shape[0]))
    return X, y, data, scaler
    


X, y, raw_data, scaler  = makeData()
x_train, x_test, y_train, y_test = splitData(X,y,0.8)

# # x_train = np.reshape(x_train,(1,x_train.shape[0],x_train.shape[1]))
# # a
x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)

S
input_dim = 1
hidden_dim = 60
num_layers = 3
output_dim = 1
num_epochs = 120

model = LSTM(input_dim, hidden_dim, num_layers, output_dim)
model, y_pred_train = trainModel(model, num_epochs, x_train,y_train)
y_pred_test = testModel(model, x_test, y_test)

x = raw_data.index

y_test = scaler.inverse_transform(y_test.detach().numpy())
y_pred_test = scaler.inverse_transform(y_pred_test.detach().numpy())

y_train = scaler.inverse_transform(y_train.detach().numpy())
y_pred_train = scaler.inverse_transform(y_pred_train.detach().numpy())


fig, (ax1, ax2) = plt.subplots(2)
fig.set_size_inches(10.5, 10.5)

ax1.plot(x[len(x) - len(y_test):], y_test)
ax1.plot(x[len(x) - len(y_test):], y_pred_test)
ax1.set_title('Test Data')
ax1.set(xlabel = 'Date', ylabel = 'Price ($)')

ax2.plot(x[:len(y_train)], y_train)
ax2.plot(x[:len(y_train)], y_pred_train)
ax2.set_title('Train Data')
ax2.set(xlabel = 'Date', ylabel = 'Price ($)')

plt.tight_layout()
plt.show()


