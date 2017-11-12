import os
import pickle
import tornado.web
from tornado.ioloop import IOLoop

import numpy as np
from PIL import Image
from io import BytesIO
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import img_to_array

def predict(img, model):
    target_size = (224, 224)
    if img.size != target_size:
        img = img.resize(target_size)

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    y = model.predict(x)
    return y[0]

model = load_model('model/chest_scan_classifier_fp.h5')


class BaseHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html')


class UploadHandler(tornado.web.RequestHandler):
    def post(self):
        # Get the file
        img = self.request.files.get('file')[0]
        img_body = BytesIO(img['body'])
        # Save the file
        with open(os.path.join('./static/uploads', img['filename']), 'wb') as f:
            image = Image.open(img_body)
            image = image.resize((224, 224))
            image.save(f, format='PNG')

        with open('class_list', 'rb') as f:
            classes = pickle.load(f)

        image = Image.open(img_body)
        prediction = predict(image, model)
        prediction = sorted(zip(classes, prediction), key=lambda x: x[1], reverse=True)[:5]
        print(img['filename'])
        print(prediction)

        self.render('predict.html', pred=prediction, file=img['filename'])


class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r'/?', BaseHandler),
            (r'/predict/?', UploadHandler)
        ]

        settings = {
            'template_path': os.path.join(os.path.dirname(__file__), 'templates'),
            'static_path': os.path.join(os.path.dirname(__file__), 'static'),
            'debug': True
        }

        super(Application, self).__init__(handlers, **settings)


def main():
    app = Application()
    print('Starting your application at port number 8000')
    app.listen(8000)
    IOLoop.instance().start()


if __name__ == '__main__':
    main()
