import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam

def load_data(database_name, table_name):
    import sqlite3
    conn = sqlite3.connect(database_name)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


# Verilerin önişlenmesi
def preprocess_data(df):
    df['Açılış Fiyatı'] = df['Açılış Fiyatı'].astype(float)
    df['Kapanış Fiyatı'] = df['Kapanış Fiyatı'].astype(float)
    
    # Sadece kapanış fiyatlarını kullanarak basit bir model kuruyoruz
    all_closing_prices = df['Kapanış Fiyatı'].values.reshape(-1,1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    all_closing_prices_scaled = scaler.fit_transform(all_closing_prices)
    
    # Eğitim ve test setlerini belirlemek
    train_size = int(len(all_closing_prices_scaled) * 0.80)
    test_size = len(all_closing_prices_scaled) - train_size
    train, test = all_closing_prices_scaled[0:train_size,:], all_closing_prices_scaled[train_size:len(all_closing_prices_scaled),:]
    
    return train, test, scaler

# Veri setlerini oluşturma
def create_dataset(dataset, look_back=10):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# LSTM Modelini oluşturma
def build_model(look_back):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Veritabanı dosyasını ve tablo ismini ekleme
database_name = 'veritabani.db'
table_name = 'Data'

# Verileri yükleme
df = load_data(database_name, table_name)

# Verilerin önişlenmesi
train, test, scaler = preprocess_data(df)

# Veri setlerinin oluşturulması
look_back = 10
X_train, Y_train = create_dataset(train, look_back)
X_test, Y_test = create_dataset(test, look_back)

# Verileri yeniden şekillendirmesi
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Modeli oluşturulması ve eğitimi
model = build_model(look_back)
model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=2)

# Tahminleri yapın-lması ve ölçeklendirmenin geri alınması
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

model.save('my_model.keras')


import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

# Eğitim ve test hatalarını (MSE) hesaplayalım
train_mse = mean_squared_error(Y_train[0], train_predict[:,0])
test_mse = mean_squared_error(Y_test[0], test_predict[:,0])

print(f'Eğitim MSE: {train_mse}')
print(f'Test MSE: {test_mse}')

# Gerçek ve tahmin edilen fiyatları grafik üzerinde gösterelim
plt.figure(figsize=(30, 12))

# Eğitim seti tahminlerini gösterelim.
plt.subplot(1, 2, 1)
plt.plot(Y_train[0], label='Gerçek Fiyatlar', marker='.')
plt.plot(train_predict[:,0], label='Tahmin Edilen Fiyatlar', alpha=0.7, marker='x')
plt.title('Eğitim Seti Tahminleri')
plt.xlabel('Zaman (Örnek indeksi)')
plt.ylabel('Fiyat')
plt.legend()

# Test seti tahminlerini gösterelim.
plt.subplot(1, 2, 2)
plt.plot(Y_test[0], label='Gerçek Fiyatlar', marker='.')
plt.plot(test_predict[:,0], label='Tahmin Edilen Fiyatlar', alpha=0.7, marker='x')
plt.title('Test Seti Tahminleri')
plt.xlabel('Zaman (Örnek indeksi)')
plt.ylabel('Fiyat')
plt.legend()

plt.tight_layout()
plt.show()
