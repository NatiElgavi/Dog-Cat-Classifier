# Arda Mavi

from keras.models import Sequential
from keras.models import model_from_json

def predict(model, X):
    Y = model.predict(X)
    Y = np.argmax(Y, axis=1)
    Y = 'cat' if Y[0] == 0 else 'dog'
    return Y

# performing predict from a saved image
def predictImg(model):
    img_dir = sys.argv[2]
    from get_dataset import get_img
    img = get_img(img_dir)
    print(img.shape)
    X = np.zeros((1, 64, 64, 3), dtype='float64')
    X[0] = img
    Y = predict(model, X)
    print('It is a ' + Y + ' !')


# performing predict frames of a video
def predictVideo(model):
    import cv2
    videoFileName = sys.argv[2]
    cap = cv2.VideoCapture(videoFileName)
    if not cap.isOpened():
        print("Error: Cannot read video file" + videoFileName)
        return
    count = 0
    moreFrames = True
    while(moreFrames):
        moreFrames, frame = cap.read()
        if moreFrames:
            X = np.zeros((1, 64, 64, 3), dtype='float64')
            X[0] = cv2.resize(frame, (64,64))
            Y = predict(model, X)
            print('Frame ' + str(count) + ' is a ' + Y + ' !')
            count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Getting model:
    model_file = open('Data/Model/model.json', 'r')
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)
    # Getting weights
    model.load_weights("Data/Model/weights.h5")
    #Getting the command option and calling the relevent function
    import sys
    import numpy as np
    mode = sys.argv[1]
    if mode == "-img":
        predictImg(model)
    elif mode == "-video":
        predictVideo(model)
    else:
        print("Wrong running mode, should be either -img or -video")
