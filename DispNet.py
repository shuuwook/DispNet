import os, sys
sys.path.append('../')

from keras import layers, models
from keras.preprocessing.image import img_to_array, load_img
from keras import backend
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import scipy
from PIL import Image

from keras.utils import plot_model
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

IMAGE_SIZE_X = 768
IMAGE_SIZE_Y = 384
BATCH_SIZE = 12
TRAINING_ROUNDS = 50
LEARNING_RATE = 1e-5

IMAGE_DIR = 'D:/data/'
GROUND_DIR = 'D:/data/ground/'
MODELS_DIR = 'D:/data/model/'
RESULT_DIR = 'D:/data/output/'

class Data:
    def __init__(self):
        # Load dataset
        left_images = sorted(os.listdir(IMAGE_DIR + 'left/'))
        right_images = sorted(os.listdir(IMAGE_DIR + 'right/'))

        left_len = len(left_images)
        right_len = len(right_images)
        assert left_len == right_len, 'Wrong number of left and right images'

        left_images = np.reshape(left_images, (left_len, 1))
        right_images = np.reshape(left_images, (right_len, 1))

        ground_truth = self.load_gt(GROUND_DIR + 'gt.pkl')

        x = np.concatenate((left_images, right_images), axis=1)
        y = ground_truth

        # Split dataset into train and test
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(x, y, test_size=0.20, random_state=22)

        # Prepare dataset
        self.train_x, self.train_y = self.prepare_data(self.train_x, self.train_y)
        self.test_x, self.test_y = self.prepare_data(self.test_x, self.test_y)

        self.test_y = [backend.eval(self.test_y[num]) for num in range(6)]


    def load_gt(self, path):
        with open(path, 'rb') as f:
            gt = pickle.load(f)

        return gt


    def prepare_data(self, path, gt):
        image = []
        for img in path:
            left_img = scipy.misc.imread(IMAGE_DIR + '/left/' + img[0], mode='RGB').astype(np.float)
            right_img = scipy.misc.imread(IMAGE_DIR + '/right/' + img[1], mode='RGB').astype(np.float)

            x_l = np.zeros((384, 768, 3), dtype=np.float32)
            x_r = np.zeros((384, 768, 3), dtype=np.float32)
            for c in range(3):
                for i in range(77, 461):
                    for j in range(96, 864):
                        x_l[i - 77, j - 96, c] = left_img[i, j, c]
                        x_r[i - 77, j - 96, c] = right_img[i, j, c]
            
            x = np.concatenate([x_l, x_r], axis=-1)
            image.append(x)
        
        image = np.array(image)
        gt = self.transform_ground_truth(gt)

        return image, gt


    def transform_ground_truth(self, ground_truth):
        ground_truth = np.array(ground_truth)
        L, H, W = list(ground_truth.shape)
        # (?, 384, 768, 1)
        ground_truth = ground_truth.reshape([L, H, W, 1])
        gt_tensor = backend.variable(ground_truth)
        # 6, ?, x, x, 1
        # [(?, 6, 12, 1), (?, 12, 24, 1), (?, 24, 48, 1), (?, 48, 96, 1), (?, 96, 192, 1), (?, 192, 384, 1)]
        gt_transformed = []
        gt_transformed.append(layers.AveragePooling2D(pool_size=(64, 64), strides=None, padding='same')(gt_tensor))
        gt_transformed.append(layers.AveragePooling2D(pool_size=(32, 32), strides=None, padding='same')(gt_tensor))           
        gt_transformed.append(layers.AveragePooling2D(pool_size=(16, 16), strides=None, padding='same')(gt_tensor))
        gt_transformed.append(layers.AveragePooling2D(pool_size=(8, 8), strides=None, padding='same')(gt_tensor))
        gt_transformed.append(layers.AveragePooling2D(pool_size=(4, 4), strides=None, padding='same')(gt_tensor))
        gt_transformed.append(layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='same')(gt_tensor))

        return gt_transformed
        

class DispNet(models.Model):
    def __init__(self, input_shape=(IMAGE_SIZE_Y, IMAGE_SIZE_X, 6), batch_size=BATCH_SIZE):
        self.batch_size = batch_size
        self.width = IMAGE_SIZE_X
        self.height = IMAGE_SIZE_Y

        # Build network
        x = layers.Input(shape=(self.height, self.width, 6), dtype='float32')
        h = self.convolutional_network(x)
        h = self.upconvolutional_network(h)
        y1, y2, y3, y4, y5, y6 = h
       
        super().__init__(inputs=x, outputs=[y1, y2, y3, y4, y5, y6])
        self.compile(loss=self.inter_loss,
                     loss_weights= [1/32, 1/32, 1/16, 1/8, 1/4, 1/2],
                     optimizer='adam')


    def inter_loss(self, y_true, y_pred):
        inter_loss = backend.sqrt(backend.mean(backend.square(y_pred - y_true)))

        return inter_loss
                

    def convolutional_network(self, inputs):
        # Input data is concatnation of left and right
        assert inputs.shape[-1] == 6, 'Inputs channels Error' # (batch, 384, 768, 6)

        # Conv 1 (batch, 192, 384, 64)
        conv_1 = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inputs)
        conv_1 = layers.Activation('relu')(conv_1)
        self.conv_1 = conv_1
        # Conv 2 (batch, 96, 192, 128)
        conv_2 = layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same')(conv_1)
        conv_2 = layers.Activation('relu')(conv_2)
        self.conv_2 = conv_2
        # Conv 3a (batch, 48, 96, 256)
        conv_3a = layers.Conv2D(256, kernel_size=(5, 5), strides=(2, 2), padding='same')(conv_2)
        conv_3a = layers.Activation('relu')(conv_3a)
        # Conv 3b (batch, 48, 96, 256)
        conv_3b = layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_3a)
        conv_3b = layers.Activation('relu')(conv_3b)
        self.conv_3b = conv_3b
        # Conv 4a (batch, 24, 48, 512)
        conv_4a = layers.Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv_3b)
        conv_4a = layers.Activation('relu')(conv_4a)
        # Conv 4b (batch, 24, 48, 512)
        conv_4b = layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_4a)
        conv_4b = layers.Activation('relu')(conv_4b)
        self.conv_4b = conv_4b
        # Conv 5a (batch, 12, 24, 512)
        conv_5a = layers.Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv_4b)
        conv_5a = layers.Activation('relu')(conv_5a)
        # Conv 5b (batch, 12, 24, 512)
        conv_5b = layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_5a)
        conv_5b = layers.Activation('relu')(conv_5b)
        self.conv_5b = conv_5b
        # Conv 6a (batch, 6, 12, 1024)
        conv_6a = layers.Conv2D(1024, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv_5b)
        conv_6a = layers.Activation('relu')(conv_6a)
        # Conv 6b (batch, 6, 12, 1024)
        conv_6b = layers.Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_6a)
        conv_6b = layers.Activation('relu')(conv_6b)
        self.conv_6b = conv_6b
        # Prediction_Loss 6 (batch, 6, 12, 1)
        prediction = layers.Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_6b)
        prediction = layers.Activation('relu', name='6x12')(prediction)
        self.pre_1 = prediction
        

        return conv_6b


    def upconvolutional_network(self, inputs):
        # Upconv 6 (batch, 12, 24, 512)
        upconv_5 = layers.Conv2DTranspose(512, kernel_size=(4, 4), strides=(2, 2), padding='same')(inputs)
        upconv_5 = layers.BatchNormalization(axis=-1)(upconv_5)
        upconv_5 = layers.Activation('relu')(upconv_5)
        # Iconv 5 (batch, 12, 24, 512)
        c = layers.Concatenate(axis=-1)([upconv_5, self.conv_5b])
        iconv_5 = layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
        iconv_5 = layers.Activation('relu')(iconv_5)
        # Prediction_Loss 5 (batch, 12, 24, 1)
        prediction = layers.Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_5)
        prediction = layers.Activation('relu', name='12x24')(prediction)
        self.pre_2 = prediction
        

        # Upconv 4 (batch, 24, 48, 256)
        upconv_4 = layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), padding='same')(iconv_5)
        upconv_4 = layers.BatchNormalization(axis=-1)(upconv_4)
        upconv_4 = layers.Activation('relu')(upconv_4)
        # Iconv 4 (batch, 24, 48, 256)
        c = layers.Concatenate(axis=-1)([upconv_4, self.conv_4b])
        iconv_4 = layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
        iconv_4 = layers.Activation('relu')(iconv_4)
        # Prediction_Loss 4 (batch, 24, 48, 1)
        prediction = layers.Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_4)
        prediction = layers.Activation('relu', name='24x48')(prediction)
        self.pre_3 = prediction
        

        # Upconv 3 (batch, 48, 96, 128)
        upconv_3 = layers.Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same')(iconv_4)
        upconv_3 = layers.BatchNormalization(axis=-1)(upconv_3)
        upconv_3 = layers.Activation('relu')(upconv_3)
        # Iconv 3 (batch, 48, 96, 128)
        c = layers.Concatenate(axis=-1)([upconv_3, self.conv_3b])
        iconv_3 = layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
        iconv_3 = layers.Activation('relu')(iconv_3)
        # Prediction_Loss 3 (batch, 48, 96, 1)
        prediction = layers.Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_3)
        prediction = layers.Activation('relu', name='48x96')(prediction)
        self.pre_4 = prediction
        

        # Upconv 2 (batch, 96, 192, 64)
        upconv_2 = layers.Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='same')(iconv_3)
        upconv_2 = layers.BatchNormalization(axis=-1)(upconv_2)
        upconv_2 = layers.Activation('relu')(upconv_2)
        # Iconv 2 (batch, 96, 192, 64)
        c = layers.Concatenate(axis=-1)([upconv_2, self.conv_2])
        iconv_2 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
        iconv_2 = layers.Activation('relu')(iconv_2)
        # Prediction_Loss 2 (batch, 96, 192, 1)
        prediction = layers.Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_2)
        prediction = layers.Activation('relu', name='96x192')(prediction)
        self.pre_5 = prediction
        

        # Upconv 1 (batch, 192, 384, 32)
        upconv_1 = layers.Conv2DTranspose(32, kernel_size=(4, 4), strides=(2, 2), padding='same')(iconv_2)
        upconv_1 = layers.BatchNormalization(axis=-1)(upconv_1)
        upconv_1 = layers.Activation('relu')(upconv_1)
        # Iconv 1 (batch, 192, 384, 32)
        c = layers.Concatenate(axis=-1)([upconv_1, self.conv_1])
        iconv_1 = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
        iconv_1 = layers.Activation('relu')(iconv_1)
        # Prediction_Loss 1 (batch, 192, 384, 1)
        prediction = layers.Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_1)
        prediction = layers.Activation('relu', name='192x384')(prediction)
        self.pre_6 = prediction
        

        return self.pre_1, self.pre_2, self.pre_3, self.pre_4, self.pre_5, self.pre_6


class Machine:
    def __init__(self, batch_size=8):
        self.batch_size = batch_size

        self.data = Data()
        print("Data prepared.")
        self.model = DispNet(input_shape=(IMAGE_SIZE_Y, IMAGE_SIZE_X, 6), batch_size=self.batch_size)
        print("Model prepared.")
        plot_model(self.model, to_file='DispNet_network.png')
        print("Network printed.")      


    def run(self, epochs=100):
        batch_num = int(len(self.data.train_x) / self.batch_size)
        for epoch in range(epochs):
            print('({} / {}) epoch training...'.format(epoch+1, epochs))
            for batch in range(batch_num):
                print('({} / {}) batch training...'.format(batch+1, batch_num))
                batch_img, batch_gt = self.load_batch_data(batch, self.batch_size)
                self.model.train_on_batch(batch_img, batch_gt)

            total_loss = self.model.evaluate(self.data.test_x, self.data.test_y, verbose=0)
            print('[{}] loss : <{}> {}'.format(epoch+1, self.model.metrics_names[0], total_loss[0]))


    def save_result(self, data):
        img_data = self.model.predict(data)
        img_data = np.array(img_data[-1]).astype(np.uint8)
        
        L, H, W, C = img_data.shape
        img_data = img_data.reshape(L, H, W)

        for num in range(L):
            img = Image.fromarray(img_data[num], mode='L')
            img.save(RESULT_DIR + 'result_' + str(num+1) + '.png')


    def load_batch_data(self, batch_num, batch_size):
        start_inx = batch_size * batch_num
        end_inx = batch_size * (batch_num+1)

        batch_img = self.data.train_x[start_inx:end_inx]
        batch_gt = [self.data.train_y[num][start_inx:end_inx] for num in range(6)]

        return batch_img, batch_gt
            


def main():
    m = Machine()
    m.run()
    m.save_result(m.data.test_x)
    print("Result Image saved.")

if __name__ == '__main__':
    main()

