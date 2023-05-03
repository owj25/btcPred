import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
trainingDays = 10
totDays = 100

df = pd.read_csv('/Users/student/Desktop/BTC-USD.csv', index_col='Date', parse_dates=True)
df.drop(columns=['Adj Close'], inplace=True)
df.head(5)


plt.plot(df.Close)
plt.xlabel("Time")
plt.ylabel("Price (USD)")
plt.title("Bitcoin price over time")
plt.savefig("initial_plot.png", dpi=250)
# plt.show()


X, y = df.drop(columns=['Close']), df.Close.values
# X.shape, y.shape

mm = MinMaxScaler()
ss = StandardScaler()

X_trans = ss.fit_transform(X)
y_trans = mm.fit_transform(y.reshape(-1, 1))


# split a multivariate sequence past, future samples (X and y)
def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix, -1]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)


X_ss, y_mm = split_sequences(X_trans, y_trans, 100, trainingDays)
# print(X_ss.shape, y_mm.shape)

assert y_mm[99].all() == y_trans[99:totDays].squeeze(1).all()
y_trans[99:totDays].squeeze(1)

total_samples = len(X)
train_test_cutoff = round(0.90 * total_samples)

X_train = X_ss[:-150]
X_test = X_ss[-150:]

y_train = y_mm[:-150]
y_test = y_mm[-150:]

# print("Training Shape:", X_train.shape, y_train.shape)
# print("Testing Shape:", X_test.shape, y_test.shape)

# convert to pytorch tensors
X_train_tensors = torch.Tensor(X_train)
X_test_tensors = torch.Tensor(X_test)
y_train_tensors = torch.Tensor(y_train)
y_test_tensors = torch.Tensor(y_test)

X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 100, X_train_tensors.shape[2]))
X_test_tensors_final = torch.reshape(X_test_tensors,(X_test_tensors.shape[0], 100, X_test_tensors.shape[2]))

# print("Training Shape:", X_train_tensors_final.shape, y_train_tensors.shape)
# print("Testing Shape:", X_test_tensors_final.shape, y_test_tensors.shape)

X_check, y_check = split_sequences(X, y.reshape(-1, 1), 100, trainingDays)
# print(X_check[-1][0:4])
# print(X.iloc[-100:-145])
print(df.Close[-trainingDays:])


class LSTM(nn.Module):
    def __init__(self, numClasses, inputSize, hiddenSize, numLayers):
        super().__init__()
        self.numClasses = numClasses
        self.numLayers = numLayers
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.lstm = nn.LSTM(
            input_size=inputSize, hidden_size=hiddenSize, num_layers=numLayers, batch_first=True, dropout=0.2
        )
        self.fc_1 = nn.Linear(hiddenSize, 128)
        self.fc_2 = nn.Linear(128, numClasses)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = torch.zeros(self.numLayers, x.size(0), self.hiddenSize)
        c_0 = torch.zeros(self.numLayers, x.size(0), self.hiddenSize)

        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hiddenSize)
        out = self.relu(hn)
        out = self.fc_1(out)  # first dense
        out = self.relu(out)  # elu
        out = self.fc_2(out)  # final output
        return out

    def training(n_epochs, lstm, optimiser, loss_fn, X_train, y_train,
              X_test, y_test):
        for epoch in range(n_epochs):
            lstm.train()
            outputs = lstm.forward(X_train)  # forward pass
            optimiser.zero_grad()  # calculate the gradient, manually setting to 0
            # obtain the loss function
            loss = loss_fn(outputs, y_train)
            loss.backward()  # calculates the loss of the loss function
            optimiser.step()  # improve from loss, i.e backprop
            # test loss
            lstm.eval()
            test_preds = lstm(X_test)
            test_loss = loss_fn(test_preds, y_test)
            if epoch % 100 == 0:
                print("Epoch: %d, train loss: %1.5f, test loss: %1.5f" % (epoch,
                                                                          loss.item(),
                                                                          test_loss.item()))


warnings.filterwarnings('ignore')

n_epochs = 1000  # 1000 epochs
learning_rate = 0.001  # 0.001 lr

input_size = 4  # number of features
hidden_size = 2  # number of features in hidden state, we gonna change this one
num_layers = 1  # number of stacked lstm layers, we gonna change this one too -_-

num_classes = trainingDays  # number of output classes

lstm = LSTM(num_classes,
              input_size,
              hidden_size,
              num_layers)

loss_fn = torch.nn.MSELoss()    # mean-squared error for regression
optimiser = torch.optim.Rprop(lstm.parameters(), lr=learning_rate)


print(LSTM.training(n_epochs=n_epochs,
              lstm=lstm,
              optimiser=optimiser,
              loss_fn=loss_fn,
              X_train=X_train_tensors_final,
              y_train=y_train_tensors,
              X_test=X_test_tensors_final,
              y_test=y_test_tensors))

df_X_ss = ss.transform(df.drop(columns=['Close']))  # old transformers
df_y_mm = mm.transform(df.Close.values.reshape(-1, 1))  # old transformers
# split the sequence
df_X_ss, df_y_mm = split_sequences(df_X_ss, df_y_mm, 100, trainingDays)
# converting to tensors
df_X_ss = torch.Tensor(df_X_ss)
df_y_mm = torch.Tensor(df_y_mm)
# reshaping the dataset
df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 100, df_X_ss.shape[2]))

train_predict = lstm(df_X_ss)  # forward pass
data_predict = train_predict.data.numpy()  # numpy conversion
dataY_plot = df_y_mm.data.numpy()

data_predict = mm.inverse_transform(data_predict)  # reverse transformation
dataY_plot = mm.inverse_transform(dataY_plot)
true, preds = [], []
for i in range(len(dataY_plot)):
    true.append(dataY_plot[i][0])
for i in range(len(data_predict)):
    preds.append(data_predict[i][0])
plt.figure(figsize=(10, 6))  # plotting
plt.axvline(x=train_test_cutoff, c='r', linestyle='--')  # size of the training set

plt.plot(true, label='Actual Data') # actual plot
plt.plot(preds, label='Predicted Data') # predicted plot
plt.title('Time-Series Prediction')
plt.legend()
plt.savefig("whole_plot.png", dpi=300)
plt.show()

test_predict = lstm(X_test_tensors_final[-1].unsqueeze(0))  # get the last sample
test_predict = test_predict.detach().numpy()
test_predict = mm.inverse_transform(test_predict)
test_predict = test_predict[0].tolist()

test_target = y_test_tensors[-1].detach().numpy()  # last sample again
test_target = mm.inverse_transform(test_target.reshape(1, -1))
test_target = test_target[0].tolist()

plt.plot(test_target, label="Actual Data")
plt.plot(test_predict, label="LSTM Predictions")
plt.savefig("small_plot.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 6))  # plotting
a = [x for x in range(2500, len(y))]
plt.plot(a, y[2500:], label='Actual data')
c = [x for x in range(len(y)-trainingDays, len(y))]
plt.plot(c, test_predict, label='One-shot multi-step prediction (1 day)')
plt.axvline(x=len(y)-trainingDays, c='r', linestyle='--')
plt.legend()
plt.show()
print(preds)
