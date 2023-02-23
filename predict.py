# Arda Mavi

import sys
import numpy as np
import cv2
import time
import os

from threading import Thread
from get_dataset import get_img
from get_dataset import resize_image
from keras.models import Sequential
from keras.models import model_from_json

def predict_image(model, img):
    prob = model.predict(img)
    best_guess = np.argmax(prob, axis=1)
    result = 'cat' if best_guess[0] == 0 else 'dog'
    
    return result

def init_model():
    # Getting model:
    model_file = open('Data/Model/model.json', 'r')
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)
    # Getting weights
    model.load_weights("Data/Model/weights.h5")

    return model

def verify_output_folders():
    if not os.path.isdir("output/cats"):
        os.makedirs("output/cats")
    if not os.path.isdir("output/dogs"):
        os.makedirs("output/dogs")

# simulate a long background operation
def process_video_frame(frame, index, results):
    time.sleep(10)

    img = resize_image(frame, 64)
    X = np.zeros((1, 64, 64, 3), dtype='float64')
    X[0] = img
    result = predict_image(model, X)

    # opencv works with BGR. convert back to bgr to align
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    save_res = cv2.imwrite(f'output/{result}s/frame_{index}.png', img)
    results[index] = result

def process_video(video_path: str):
    threads = []
    results = {}
    verify_output_folders()

    cap = cv2.VideoCapture(video_path)

    index = 0
    while (True):
        ret, frame = cap.read()
        if (ret):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = Thread(target=process_video_frame, args=(frame, index, results))
            threads.append(t)
            t.start()
            index += 1
        else:
            break

    cap.release()        

    # wait for all background processing to complete
    for t in threads:
        t.join()

    # print prediction results
    with open('output/task2.txt', 'w') as f:
        for index in sorted(results.keys()):
            msg = f'{index} - {results[index]}'
            f.write(f'{msg}\n')
            print(msg)

    test_all_results(results)
    print('All done')

def test_all_results(results):
    acc_cats = test_class_results(results, 'Data/Train_Data/cat', 'cat', 99, 0)
    acc_dogs = test_class_results(results, 'Data/Train_Data/dog', 'dog', 99, 100)

    print(f'Accuracy of video/images predictions of Cats: {acc_cats}')
    print(f'Accuracy of video/images predictions of Dogs: {acc_dogs}')

def test_class_results(results, source: str, img_prefix:str, count: int, match_offset: int) -> float:
    matches = 0
    for index in range(count):
        img_name = f'{img_prefix}.{index}.jpg'
        result = process_image(os.path.join(source, img_name), False)

        # compare the prediction class we got from the image file to the video frame prediction result
        if results[match_offset + index] == result:
            matches += 1

        if index == count:
            break

    return matches / count

def process_image(image_path: str, print_result:bool):
    img = get_img(image_path)
    X = np.zeros((1, 64, 64, 3), dtype='float64')
    X[0] = img
    result = predict_image(model, X)
    
    if print_result:
        print('It is a ' + result + ' !')
    
    return result


if __name__ == '__main__':
    target = sys.argv[1]
    model = init_model()

    if target.endswith('.mp4'):
        process_video(target)
    else:
        process_image(target, True)
