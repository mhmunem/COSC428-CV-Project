from d01_data.load_data import download_data
from d03_processing.create_training_data import processed_data
from d04_modeling import train_model
from tensorflow.keras.optimizers import Adam

# Variables
DATASET = "PU"
TEST_RATIO = 0.80
WINDOW_SIZE = 11
DATA_PATH = "./data/01_raw/"
LOSS ='categorical_crossentropy'
OPTIMIZER = Adam(learning_rate=0.001)
MATRICS = ['acc']
MODEL_FOLDER = "./data/04_models/"

download_data(data_type=DATASET, save_folder=DATA_PATH)
X_train,y_train,X_test,y_test = processed_data(dataset=DATASET,test_ratio=TEST_RATIO,window_size=WINDOW_SIZE,data_path=DATA_PATH)
S = X_train.shape[1]
L = X_train.shape[-2]

model = train_model.create_model(S=S,L=L,dataset=DATASET)
model.summary()
model.compile(
    loss = LOSS,
    optimizer = OPTIMIZER,
    metrics = MATRICS
)

train_model.train_model(model=model,
                        save_folder=MODEL_FOLDER,
                        dataset=DATASET,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=None,
                        y_val=None,
                        epochs=50,
                        batch_size=32)
