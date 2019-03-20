#基于flow_layers层 不使用Build_basic_model和CoupleWrapper等层，用自带层替换
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from flow_layers import *
import imageio
import numpy as np
from scipy import misc
import glob
import os
from scipy.linalg import norm,orth
import matplotlib.pyplot as plt

class Scale(Layer):
    """尺度变换层
    """
    def __init__(self, **kwargs):
        super(Scale, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(1, input_shape[1].value,input_shape[2].value,input_shape[3].value),
                                      initializer='glorot_normal',
                                      trainable=True)
        super(Scale, self).build(input_shape)
    def call(self, inputs):
        self.add_loss(-K.sum(self.kernel)) # 对数行列式
        return K.exp(self.kernel) * inputs
    def inverse(self):
        scale = K.exp(-self.kernel)
        return Lambda(lambda x: scale * x)


#nice 加强版glow
class Glow:
    def __init__(self,img_size,in_channel,level,depth,isInverse=False):
        self.hidden_dim = 128
        self.in_channel = in_channel
        self.img_size = img_size
        self.level = level
        self.depth = depth
        self.isInverse = isInverse
        self.inner_layers = []
        self.outer_layers = []
        # self.split = Split()
        # self.add = keras.layers.Add()
        # self.concat = Concat()
        self.scale = Scale()
        for i in range(6):
            self.inner_layers.append([])
        for i in range(2):
            self.outer_layers.append([])
    
    def __call__(self):
        inputs = keras.layers.Input(shape=(self.img_size,self.img_size,self.in_channel))
        x = inputs
        if not self.isInverse:
            for i in range(self.level):
                squeeze = Squeeze()
                self.outer_layers[0].append(squeeze)
                x = squeeze(x)
                for j in range(self.depth):
                    channel = self.in_channel*4**(i+1)//2
                    permute = Permute(mode='random')
                    split = Split()
                    inner_con1 = keras.layers.Conv2D(self.hidden_dim,(3,3),padding='same',activation='relu')
                    inner_con2 = keras.layers.Conv2D(self.hidden_dim,(1,1),padding='same',activation='relu')
                    inner_con3 = keras.layers.Conv2D(channel,(3,3),padding='same')
                    concat = Concat()
                    # add = keras.layers.Add()
                    self.inner_layers[0].append(permute)
                    self.inner_layers[1].append(split)
                    self.inner_layers[2].append(inner_con1)
                    self.inner_layers[3].append(inner_con2)
                    self.inner_layers[4].append(inner_con3)
                    # self.inner_layers[5].append(add)
                    self.inner_layers[5].append(concat)

                    x = permute(x)
                    x1,x2 = split(x)
                    #加性耦合
                    t_x1 = inner_con1(x1)
                    t_x1 = inner_con2(t_x1)
                    t_x1 = inner_con3(t_x1)
                    x2 = keras.layers.add([x2,t_x1])
                    x = concat([x1,x2])
            for i in range(self.level):
                unsqueeze = UnSqueeze()
                self.outer_layers[1].append(unsqueeze)
                x = unsqueeze(x)
            x = self.scale(x)
            encoder = keras.models.Model(inputs,x)
            for l in encoder.layers:
                if hasattr(l,'logdet'):
                    encoder.add_loss(l.logdet)
            loss = K.mean(0.5*K.sum(x**2,[1,2,3])+ 0.5 * np.log(2*np.pi) * self.img_size**2*self.in_channel)
            encoder.add_loss(loss)
            return encoder
        else:
            x = self.scale.inverse()(x)
            for i in reversed(range(self.level)):
                x = self.outer_layers[1][i].inverse()(x)
            for i in reversed(range(self.level)):
                for j in reversed(range(self.depth)):
                    index = i *self.depth+j
                    x1,x2 = self.inner_layers[5][index].inverse()(x)
                    t_x1 = self.inner_layers[2][index](x1)
                    t_x1 = self.inner_layers[3][index](t_x1)
                    t_x1 = self.inner_layers[4][index](t_x1)
                    x2 = keras.layers.subtract([x2,t_x1])
                    x = self.inner_layers[1][index].inverse()([x1,x2])
                    x = self.inner_layers[0][index].inverse()(x)
                x = self.outer_layers[0][i].inverse()(x)
            decoder = keras.models.Model(inputs,x)
            return decoder
    
    def inverse(self):
        model = Glow(self.img_size,self.in_channel,self.level,self.depth,isInverse=True)
        model.inner_layers = self.inner_layers
        model.outer_layers = self.outer_layers
        model.scale = self.scale
        return model
            

                    
                    



class Evaluate(Callback):
    def __init__(self):
        self.lowest = 1e10
    def on_epoch_end(self, epoch, logs=None):
        path = 'glow_samples/test_%s.png' % epoch
        sample(1, path)
        if logs['val_loss'] <= self.lowest:
            self.lowest = logs['val_loss']
            encoder.save_weights('best_glow_encoder.weights')
        elif (logs['val_loss'] > 0 and epoch > 10) or  np.isnan(logs['val_loss']) :
            """在后面，loss一般为负数，一旦重新变成正数，
            就意味着模型已经崩溃，需要降低学习率。
            In general, loss is less than zero.
            If loss is greater than zero again, it means model has collapsed.
            We need to reload the best model and lower learning rate.
            """
            encoder.load_weights('best_glow_encoder.weights')
            K.set_value(encoder.optimizer.lr, 1e-4)


def _parse_function(img):
    height,width = img.shape
    img = tf.reshape(img,[height,width,1])
    return img,0

        

if __name__ == "__main__":
    if not os.path.exists('glow_samples'):
        os.mkdir('glow_samples')
    #加载数据
    (x_train, y_train_), (x_test_, y_test_) = keras.datasets.mnist.load_data()
    # x_train = x_train.astype('float32') 
    height,width = x_train[0].shape
    # width = x_train[0][0]
    center_height = int((height - width) / 2)
    x_train = x_train.astype(np.float32)/ 255.
    x_test = x_test_.astype(np.float32)/255.
    img_size = 28  # for a fast try, please use img_size=32
    depth = 5  # orginal paper use depth=32
    level = 2  # orginal paper use level=6 for 256*256 CelebA HQ

    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(batch_size=60)
    dataset = dataset.repeat()


    glow = Glow(img_size,1,level,depth)
    encoder = glow()
    encoder.summary()
    # encoder.compile(optimizer=keras.optimizers.Adam())
    decoder = glow.inverse()()

    def sample(std, path):
        """采样查看生成效果（generate glow_fashion_samples1 per epoch）
        """
        n = 9
        figure = np.zeros((img_size * n, img_size * n))
        for i in range(n):
            for j in range(n):
                decoder_input_shape = [1,img_size,img_size,1]
                z_sample = np.array(np.random.randn(*decoder_input_shape)) * std
                x_decoded = decoder.predict(z_sample)
                digit = x_decoded[0].reshape(img_size, img_size)
                figure[i * img_size: (i + 1) * img_size,
                    j * img_size: (j + 1) * img_size] = digit
        figure = np.clip((figure)*255, 0, 255)
        if not np.isnan(figure).any():
            imageio.imwrite(path, figure)

    evaluator = Evaluate()
    weights1 = encoder.get_weights()
    sess = keras.backend.get_session()
    # encoder.fit(dataset,
    #             steps_per_epoch=1000,
    #             epochs=10,
    #             validation_data=dataset,
    #             validation_steps=1000,
    #             callbacks=[evaluator])
    encoder.load_weights('best_glow_encoder.weights')
    weights2 = encoder.get_weights()

    # encoder=keras.models.load_model('glow.h5')