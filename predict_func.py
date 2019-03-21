from main import predict_arr
from img_process_and_plots import show_img, load_real_img
from global_vars import *


def predict(photo=samp_photo):

    test_real = labels_dict[predict_arr(load_real_img(img=photo))[0]]
    show_img(samp_photo, processed=True, real=True, title=test_real)


predict()
