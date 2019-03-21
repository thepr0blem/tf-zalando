# Global variabless
IMG_SIZE = 28
IMG_CHANNELS = 1
N_CLASSES = 10
BATCH_SIZE = 128
LR = 1e-3
DROPOUT = 0.2

model_path = r"./model/"
model_name = "final"

path_train = r"./data/fashion-mnist_train.csv"
path_test = r"./data/fashion-mnist_test.csv"

samp_photo = "sample_photo.jpg"

labels_dict = {0: "T-shirt_top",
               1: "Trouser",
               2: "Pullover",
               3: "Dress",
               4: "Coat",
               5: "Sandal",
               6: "Shirt",
               7: "Sneaker",
               8: "Bag",
               9: "Ankle_boot"}
