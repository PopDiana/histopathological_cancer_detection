from fastai.vision import *
from torchvision.models import *
from PIL import Image
import cv2

SIZE = 90
TEMP = 'static/temp/image.tiff'


def predict_value(img):
    # load model
    empty_data = ImageDataBunch.load_empty('static/learner/')
    model = cnn_learner(empty_data, densenet201).load('densenet201')
    # convert tiff image to tensor
    tensor_image = to_tensor(img)
    if str(model.predict(tensor_image)[0]) == '1':
        return 'Positive'
    else:
        return 'Negative'


def to_tiff(path):
    img = Image.open(path)
    temp_path = TEMP
    img.save(temp_path)


def to_tensor(path):
    bgr_image = cv2.imread(path)
    b, g, r = cv2.split(bgr_image)
    rgb_image = cv2.merge([r, g, b])
    h, w, c = rgb_image.shape
    rgb_image = rgb_image[(h - SIZE) // 2:(SIZE + (h - SIZE) // 2), (h - SIZE) // 2:(SIZE + (h - SIZE) // 2), :] / 256
    # center crop - 32px
    return vision.Image(px=pil2tensor(rgb_image, np.float32))


