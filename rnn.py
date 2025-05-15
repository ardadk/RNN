import numpy as np 
import pandas as pd
import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import warnings
warnings.filterwarnings("ignore")

import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


df = pd.read_csv("/mnt/c/Users/Arda/Desktop/RNN/international-airline-passengers.csv")
df.columns = ["Month","Passengers"]
# print(df.head())   ilk verileri verir

# print(df.tail())   son verileri verir
# print(df.shape)    kaç satır kaç sütun var onu verir
# print(df.dtypes)   değişken türlerini verir
# print(df.isnull().sum()) boş değer var mı?
# print(df.describe().T) matematiksel değerleri verir bkz. standart sapma,medyan,ortalama

df = df[:144] # son satır null diye almadık

# print(df.info()) Genel bilgi verir

"df["Month"] = pd.to_datetime(df["Month"]) # Ay değişkenini kategorik veriden zaman verisine çevirdik (türü değişti)
# print("Minimum Tarih:",df["Month"].min())   Ay sütunundaki min değer
# print("Maksimum Tarih:",df["Month"].max())  Ay sütunundaki max değer


df.index = df["Month"] # Month'u indexledik ve artık tek bir değer var o da passengers sayıları Month sadece bir index


df.drop("Month", axis=1, inplace=True) # Sütun olan Month'u (yani index olmayan) sildik

result_df = df.copy() #DataFrame'in kopyası alındı

df.plot(figsize=(14,8),title="Monthly airline passengers"); #Veri Görselleştirildi

plt.show()

data = df["Passengers"].values # Passengers değerleri numpy matrisindeki değerlere döndürüldü

data = data.astype("float32") # matris float64'ten float32'ye çevrildi

data = data.reshape(-1,1) # tek boyutlu matrisi 2 boyutlu matrise çevirdik ilk parametre (-1) özel parametre otomatik hesaplar;ikinci parametre(1) sütun sayısını verir



def split_data(dataframe,test_size):
    position=int(round(len(dataframe)*(1-test_size))) #train ve test set'lerini bölmek için test_size parametresinden aldığımız değeri dataframedeki veri sayısıyla çarptık
    train=dataframe[:position]  #dataframe matrisindeki 0'dan position'a kadar olanları train setine ver
    test=dataframe[position:]   #dataframe matrisindeki position'dan sonuncuya kadar olanları test setine ver
    return train,test,position #değerleri döndür

train, test , position = split_data(df,0.33) #fonksiyonu çağırdık ve train %67;test %33 bölündü 
#print(train.shape,test.shape)


scaler_train = MinMaxScaler(feature_range=(0,1)) # train setindeki verileri 0 ile 1 arasındaki karşılıklarına çevirecek methodu yazdık
#örneğin dizi [1,2,3,4,5] olsaydı  1-->0 2-->0.25 3-->0.5 4-->0.75 5-->1'e denk gelecekti
#bu değer kullanıcı tarafından belirlenir 0 ve 1 yerine farklı değerler girilseydi
#o iki değer arasındaki karşılıklara denk gelirdi
train=scaler_train.fit_transform(train) #çevirme işlemi uygulandı.

scaler_test = MinMaxScaler(feature_range=(0,1))# aynı işlem test seti için
test=scaler_test.fit_transform(test)           # uygulandı


def create_features(data,lookback):   # RNN ağında tahmin yapılırken en çok etkileneceği parametre
    X,Y = [], []                        # bir önceki değişken olacağı için loopback(yani kaydırma)
    for i in range(lookback,len(data)): # yapmış olduk
        X.append(data[i-lookback:i,0])
        Y.append(data[i,0])
    return np.array(X), np.array(Y)
lookback=1
X_train,y_train = create_features(train,lookback) #Train setinin özellikleri çıkartıldı 

X_test,y_test = create_features(test,lookback)    #Test setinin özellikleri çıkartıldı

X_train = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1])) #Sinir ağının anlayabilmesi için 3.boyut eklendi
X_test=np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))       #Sinir ağının anlayabilmesi için 3.boyut eklendi
y_train=y_train.reshape(-1,1)    
y_test=y_test.reshape(-1,1)



model = Sequential()  #Giriş katmanı
#RNN katmanı
model.add(SimpleRNN(units=50,              #50 sinir ağı sokuldu
                    activation="relu",     #ReLu aktivasyon fonksiyonu
                    input_shape=(X_train.shape[1],lookback)))# alınacak verinin özellikleri, ilk parametre X_train'în 2.sütunu yani yolcu sayısı;
                                                             #ikinci parametre öğrenme işlemini yapmak için en çok hangi veriye bakayım 
model.add(Dropout(0.2))     # %0.2 sönümleme yapıldı
model.add(Dense(1))         # Çıkış katmanı olarak tek nöron seçtik

model.compile(loss="mean_squared_error",optimizer="adam")  #Loss fonksiyonumuz seçildi ve optimizasyonu için adam seçildi

callbacks = [EarlyStopping(monitor="val_loss", #val_loss'a göre izleneceği seçildi
                           patience=3,         #kaçıncı periyottan sonra early stopping uygualanacağı belirlendi
                           verbose=1,          #Her adımda ekrana yazacağını seçtik
                           mode="min"),        #overfit'in engellenmesi val_loss'taki değişim parametresi minimuma göre ayarlandık
             ModelCheckpoint(filepath="mymodel.keras", # model ismi girildi
                             monitor="val_loss",    # val_loss'a göre kaydedilecek
                             mode="min",            # val_loss'un değişim parametresi minimum olacak
                             save_best_only=True,   #en iyi model kaydedilecek
                             save_weights_only=False,#model kaydetmek istediğimiz için sadece ağırlıkların kaydedilmesine false girdik
                             verbose=1)]             #kayıt edildiğini her adımda yazdıracak şekilde ayarlandı 

history=model.fit(x=X_train,   #Bağımsız değişken
                  y=y_train,   #Bağımlı değişken
                  epochs=50,   #Kaç kez öğrenme yapıcak
                  batch_size=1,#Veri grubunun boyutunu girdik
                  validation_data=(X_test,y_test), #Doğrulama verisi yani test setimizi girdik
                  callbacks= callbacks, #model çalışırken geri çağıracağımız fonksiyonlar kümemizi callbacks listesinde belirtmiştik
                  shuffle=False)  #verileri shuffle'lama yani mixleme
 



