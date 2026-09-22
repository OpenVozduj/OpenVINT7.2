from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

def OV_MLP():
    model_mlp = Sequential()
    model_mlp.add(Input(shape=(12,)))
    model_mlp.add(Dense(32, activation='relu'))
    model_mlp.add(BatchNormalization())
    model_mlp.add(Dense(32, activation='relu'))
    model_mlp.add(BatchNormalization())
    model_mlp.add(Dense(5, activation='sigmoid'))
    
    model_mlp.compile(optimizer=Adam(), loss='mean_squared_error', 
                      metrics=[RootMeanSquaredError()])
    return model_mlp