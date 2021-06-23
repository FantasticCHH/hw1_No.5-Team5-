from preprocess import*
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
# Second dimension of the feature is dim2
feature_dim_2 = 50#50 

# Save data to array file first
save_data_to_array(max_len=feature_dim_2)

# # Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test()
#X_train, y_train = get_train_test()
# # Feature dimension
feature_dim_1 = 20#20
channel = 1
epochs =200
batch_size = 50#50 35
verbose = 1
num_classes = 8


# Reshaping to perform 2D convolution
X_train = X_train.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)
X_test = X_test.reshape(X_test.shape[0], feature_dim_1, feature_dim_2, channel)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)
#------------------------------訓練------------------------
'''
def get_model():
    model=Sequential()
    model.add(BatchNormalization(input_shape=(feature_dim_1, feature_dim_2, channel)))
    model.add(Conv2D(64, kernel_size=(2,2),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, kernel_size=(2,2), activation='relu'))    
    model.add(MaxPooling2D(pool_size=(2,2)))
   
    model.add(Conv2D(256, kernel_size=(2,2), activation='relu'))    
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    
    model.add(Dense(256, activation='relu'))   
    model.add(Dropout(0.3))
   
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    
    return model

# Predicts one sample
def predict(filepath, model):
    sample = wav2mfcc(filepath)
    sample = sample**2
    sample = sample**0.5
    sample_reshaped = sample.reshape( feature_dim_1, feature_dim_2, channel)
    return get_labels()[0][
            np.argmax(model.predict(sample_reshaped))
    ]


model = get_model()
model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot))
model.save('my_model0623.h5')
'''
#%%plot loss acc

#%%-------=-------------------測試-------------

from keras.models import load_model
model = load_model("my_model0623.h5")

predict1 = model.predict_classes(X_train)
print(predict1)

#%%存檔CSV檔

#labels = os.listdir("./demotest/666/")#demotest
labels = os.listdir("./test_set/0/")

import csv
c=0

#補標題
table=[['wav','label']]
with open('output.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
      # 寫入二維表格
        writer.writerows(table)
        
for a in predict1:
    table=[
        [labels[c],a],
        ]
    c+=1
    with open('output.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
      # 寫入二維表格
        writer.writerows(table)
#補最後一個       
table=[['509142.wav','2']]       
with open('output.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
      # 寫入二維表格
        writer.writerows(table)
