# Arda Mavi

from keras.models import Sequential
from keras.models import model_from_json


def predict(model, X):
    Y = model.predict(X)
    Y = np.argmax(Y, axis=1)
    Z = [str(i) for i in Y]
    print(Y)
    for m in range(0, len(Y)):
        if Y[m] == 0:
            Z[m] = 'cat'
        else:
            Z[m] = 'dog'
        print('It is a ' + Z[m] + ' !')


    return Z


def video_capture(file_name):
    import cv2

    cap = cv2.VideoCapture(file_name)
    if not cap.isOpened():
        print("failed to open file")
        return

    image_index = 0
    file_name_list: list = []
    while (cap.isOpened()):
        print(f"frame {image_index}")
        ret, frame = cap.read()
        if ret == True :
            # path_text = r'\Data\Videos\test_frames\'
            path_text = ""
            path_text += str(image_index) + ".jpg"

            cv2.imwrite(path_text, frame)
            file_name_list.append(path_text)

            image_index += 1
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    return file_name_list, image_index


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        img_dir = sys.argv[1]
    else:
        img_dir = r"C:\Users\TomerGilboa\PycharmProjects\Python\Dog-Cat-Classifier\data\Videos" \
              r"\cats_and_dogs.mp4"

    from get_dataset import get_img
    import numpy as np

    if ".mp4" in img_dir:  # checking if the file is video
        img_dir_list, total_imgs = video_capture(img_dir)
        print(img_dir_list)
        # img = get_img(img_dir_list[0])
        X = np.zeros((total_imgs, 64, 64, 3), dtype='float64')
        for i in range(0, total_imgs):
            X[i] = get_img(img_dir_list[i])

    else:
        img = get_img(img_dir)
        X = np.zeros((1, 64, 64, 3), dtype='float64')
        X[0] = img



    # Getting model:
    model_file = open('Data/Model/model.json', 'r')
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)
    # Getting weights
    model.load_weights("Data/Model/weights.h5")
    Y = predict(model, X)
    # print('It is a ' + Y + ' !')
