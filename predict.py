# Arda Mavi
import asyncio

from keras.models import Sequential
from keras.models import model_from_json
from time import sleep
import cv2


async def save_file():
    Z=[]
    for m in range(0, len(Y)):
        if Y[m] == 0:
            Z.append('cat')

        else:
            Z.append('dog')

        print('It is a ' + Z[m] + ' !')
        output_path = 'C:\\Users\\TomerGilboa\\PycharmProjects\\Python\\Dog-Cat-Classifier' \
                      '\\output\\' + Z[m] + 's\\' + str(m) + ".jpg"
        files_path = 'C:\\Users\\TomerGilboa\\PycharmProjects\\Python\\Dog-Cat-Classifier' \
                     '\\Data\\Videos\\test_frames\\' + str(m) + ".jpg"
        img = get_img(files_path)

    cv2.imwrite(output_path, img)
    sleep(30)


def predict(model, X):
    global Y
    Y = model.predict(X)
    Y = np.argmax(Y, axis=1)


    loop = asyncio.get_event_loop()
    loop.run_until_complete(save_file())

    return


def video_capture(file_name):
    cap = cv2.VideoCapture(file_name)
    if not cap.isOpened():
        print("failed to open file")
        return

    image_index = 0
    file_name_list: list = []
    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            frames_path = 'C:\\Users\\TomerGilboa\\PycharmProjects\\Python\\Dog-Cat-Classifier' \
                          '\\Data\Videos\\test_frames\\' + str(image_index) + ".jpg"
            # print(frames_path )
            # path_text = ""
            # path_text += str(image_index) + ".jpg"

            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cv2.imwrite(frames_path, frame)
            file_name_list.append(frames_path)

            image_index += 1
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return file_name_list, image_index


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        img_dir = sys.argv[1]
    else:
        img_dir = "C:\\Users\\TomerGilboa\\PycharmProjects\\Python\\Dog-Cat-Classifier\\data" \
                  "\\Videos\\cats_and_dogs.mp4"

    from get_dataset import get_img
    import numpy as np

    if ".mp4" in img_dir:  # checking if the file is video
        img_dir_list, total_imgs = video_capture(img_dir)
        X = np.zeros((total_imgs, 64, 64, 3), dtype='float64')
        for i in range(0, total_imgs):
            file_name = img_dir_list[i]
            X[i] = get_img(file_name)

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
    predict(model, X)
    # print('It is a ' + Y + ' !')
