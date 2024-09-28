
#environment import
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, Callback
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#GPU quota
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)])

#ACNN model
class ACNN(keras.Model):
    def __init__(self, **kwargs):
        super(ACNN, self).__init__(**kwargs)
        """model initialize"""

        #digital layers initialise
        self.batchnomalization_1 = layers.BatchNormalization()
        self.batchnomalization_2 = layers.BatchNormalization()
        self.maxpooling_1 = layers.MaxPooling2D((3,3))
        self.maxpooling_2 = layers.MaxPooling2D((2,2))
        self.activation_1 = layers.Activation('relu')
        self.conv_1 = layers.Conv2D(64,(3,3),activation='relu')
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(0.4)
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(84, activation='relu')
        self.dense3 = layers.Dense(10, activation='softmax')
        
        #optical parameters initialise
        self.size=84        #number of samples of the training phase
        self.kernel_num=16  #number of the optical kernel
        self.size_f=167     #number of samples in the spatial frequency domain
        self.P_pad=np.int32(((self.size_f+1)/2-self.size)/2)    #put pupil function in the periphery to fill 0 to self.size_f when doing Autocorrelation
        self.X_size=90      #resize input images
        self.X_pad=np.int32((self.size_f+1-self.X_size)/2)      #put resized input images in the periphery to fill 0 to self.size_f befering FFT
        self.center=120     #take the image center after optical convolution

        #pupil function initialise
        self.ran = np.random.uniform(-0.5*np.pi, 0.5*np.pi, [self.kernel_num, 15, 15])
        self.ran=tf.expand_dims(self.ran,axis=-1)
        self.ran=tf.image.resize(self.ran,[self.size,self.size])
        self.ran=tf.squeeze(self.ran)
        self.P = tf.Variable(self.ran,trainable=True,name='P',dtype=tf.float32)


    def call(self, inputs):
        """model propagation"""

        #FFT for inputs
        x = inputs
        x = tf.image.resize(x, (self.X_size, self.X_size))
        x = tf.pad(x, [[0,0],[self.X_pad, self.X_pad-1], [self.X_pad, self.X_pad-1],[0,0]], mode='CONSTANT')
        x = tf.squeeze(x, axis=-1)
        x = tf.cast(x, tf.complex64)

        x=tf.signal.fftshift(x)
        x=tf.signal.fft2d(x)
        x=tf.signal.fftshift(x)
        
        #optical convolution with parallel acceleration
        #paralled calculation
        x = tf.repeat(x, 16, axis=0)

        x, OTF = self.popagation(x, self.P)
        x = tf.math.real(x)
        # Split into 16 tensors of shape 16*167*167 along the first dimension
        split_tensor = tf.split(x, 16, axis=0)

        # Initialize an empty list to store results
        result = []

        # Operate on each 16*167*167 tensor
        for tensor in split_tensor:
            # Split the 16*167*167 tensor into two 8*167*167 tensors
            sub_tensors = tf.split(tensor, 2, axis=0)
            
            # Subtract the two tensors to get an 8*167*167 image
            sub_result = tf.subtract(sub_tensors[0], sub_tensors[1])
            
            # Transpose the 8*167*167 tensor to 167*167*8
            transposed_result = tf.transpose(sub_result, perm=[1, 2, 0])
            
            # Add the result to the result list
            result.append(transposed_result)

        # Stack the 16 resulting 167*167*8 images into a tensor of shape 16*167*167*8
        x = tf.stack(result, axis=0)

        noise_factor = 0.003
        x = x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)

        #digital backend
        x = self.get_center(x)
        x = self.activation_1(x)
        x = self.maxpooling_1(x)
        #x = self.conv_1(x)
        #x = self.maxpooling_2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        #x = self.dense1(x)
        #x = self.batchnomalization_1(x)
        #x = self.dense2(x)
        #x = self.batchnomalization_2(x)
        outputs = self.dense3(x)
        
        return outputs
    
    def Acorr(self,P):
        """Autocorrelation calculation without"""
        
        #size preparation
        PP=tf.math.conj(P)

        X=self.size+2*self.P_pad
        padx=np.int32(X/2)
        pady=np.int32(X/2)
        P = tf.pad(P, [[padx, padx-1], [pady, padx-1]], mode='CONSTANT')
        PP = tf.pad(PP, [[padx, padx-1], [pady, padx-1]], mode='CONSTANT')
        P = tf.expand_dims(tf.expand_dims(P,axis=0),axis=3)
        PP = tf.expand_dims(tf.expand_dims(PP,axis=2),axis=3)

        #convolution of self and self-conjugation
        P_real = tf.math.real(P)
        P_imag = tf.math.imag(P)
        PP_real = tf.math.real(PP)
        PP_imag = tf.math.imag(PP)

        strides=[1,1,1,1]
        padding='SAME'
        OTF_real = tf.nn.conv2d(P_real, PP_real, strides, padding) - tf.nn.conv2d(P_imag, PP_imag, strides, padding)
        OTF_imag = tf.nn.conv2d(P_real, PP_imag, strides, padding) + tf.nn.conv2d(P_imag, PP_real, strides, padding)

        #normalization
        P_amp = self.get_amp(self.size,self.size/2)
        amp=tf.reduce_sum(tf.math.abs(P_amp) ** 2)
        OTF_real=OTF_real/amp
        OTF_imag=OTF_imag/amp
        OTF = tf.complex(OTF_real, OTF_imag)
        OTF = tf.squeeze(OTF)

        return OTF
    
    def FFT_Acorr(self,P):
        """Autocorrelation calculation with FFT acceleration"""

        #size preparation
        X=self.size+2*self.P_pad
        padx=np.int32(X/2)
        pady=np.int32(X/2)
        P = tf.pad(P, [[0,0],[padx, padx-1], [pady, padx-1]], mode='CONSTANT')

        #FFT acceleration
        F_P=tf.signal.fft2d(P)
        F_PP=tf.math.conj(F_P)
        F_OTF=tf.multiply(F_P,F_PP)
        OTF = tf.signal.ifft2d(F_OTF) 
        OTF = tf.signal.fftshift(OTF)
        OTF_1, OTF_2 = tf.split(OTF, num_or_size_splits=2, axis=0)
        OTF = tf.concat([OTF_2, OTF_1], axis=0)
                
        #normalization
        P_amp = self.get_amp(self.size,self.size/2)
        amp=tf.reduce_sum(tf.math.abs(P_amp) ** 2)
        OTF_real = tf.math.real(OTF)
        OTF_imag = tf.math.imag(OTF)
        OTF_real=OTF_real/amp
        OTF_imag=OTF_imag/amp
        OTF = tf.complex(OTF_real, OTF_imag)

        return OTF 
    def get_amp(self,M,R):
        """get a central with a value of 1, surrounded by a distribution of 0"""

        arr = np.zeros((M, M))
        imgSize = M
        x, y = np.meshgrid(np.arange(-(imgSize-1)/2, (imgSize)/2), np.arange(-(imgSize-1)/2, (imgSize)/2))
        arr[x**2 + y**2 <= R**2] = 1
        arr=tf.cast(arr,tf.float32)
        tensor = tf.convert_to_tensor(arr)

        return tensor
    
    def get_center(self,input_tensor):
        """cut part of the image center"""

        center=self.center
        b=np.int32((self.size_f+1)/2-0.5*center)
        begin = [0, b, b, 0]
        size = [16, center, center, 8]
        output_tensor = tf.slice(input_tensor, begin, size)
        return output_tensor

    def popagation(self,x,phase):
        """optical propagation in the spatial frequency domain"""
        
        #pupil function preparation
        P_amp = self.get_amp(self.size,self.size/2)
        P_amp = tf.expand_dims(P_amp,axis=0)
        P_amp=tf.repeat(P_amp,16,axis=0)
        P_real=P_amp * tf.cos(phase)
        P_real=tf.pad(P_real,[[0,0], [self.P_pad,self.P_pad],[self.P_pad,self.P_pad]],mode='CONSTANT')
        P_imag=P_amp * tf.sin(phase)
        P_imag=tf.pad(P_imag,[[0,0], [self.P_pad,self.P_pad],[self.P_pad,self.P_pad]],mode='CONSTANT')
        P = tf.complex(P_real, P_imag)

        #Acorr
        OTF=self.FFT_Acorr(P)

        #size fit
        OTF_0 = OTF
        for i in range(15):
            OTF = tf.concat([OTF, OTF_0], axis=0)

        #multiplication in the spatial frequency domain
        x = tf.multiply(x,OTF)
        
        #IFFT
        x=tf.signal.fftshift(x)
        x=tf.signal.ifft2d(x)
        x=tf.signal.fftshift(x)

        return x,OTF_0

# Set initial learning rate
initial_learning_rate = 1e-2

# Set attenuation steps and attenuation rate
decay_steps = 1000
decay_rate = 0.95

# Create learning rate decay strategy
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate)

# Create optimizer instance
optimizer = Adam(learning_rate=lr_schedule)

model_P1 = ACNN()
model_P1.build((None, 28, 28, 1))
model_P1.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# dataset import
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = tf.expand_dims(x_train, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)

# Define noise factor
noise_factor = 0.04

# Add Gaussian noise
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# Ensure data is between 0 and 1
x_train = np.clip(x_train_noisy, 0., 1.)
x_test = np.clip(x_test_noisy, 0., 1.)

def augment(images, labels, x1=0.01, x2=0.1, x3=0.05):
    # Random rotation
    rotation_layer = tf.keras.layers.experimental.preprocessing.RandomRotation(factor=(-x1, x1), fill_mode='reflect', interpolation='bilinear')
    images = rotation_layer(images)
    
    # Random zoom
    zoom_layer = tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(-x2, x2), width_factor=(-x2, x2), fill_mode='reflect', interpolation='bilinear')
    images = zoom_layer(images)
    
    # Random horizontal and vertical translation
    translation_layer = tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=(-x3, x3), width_factor=(-x3, x3), fill_mode='reflect', interpolation='bilinear')
    images = translation_layer(images)
    
    return images, labels

# Create dataset and set batch size
BATCH_SIZE = 16
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.map(augment).shuffle(10000).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

csv_logger = CSVLogger('training.log')  # Create CSVLogger object, logs will be saved to a file named 'training.log'

class SaveBestModelVariables(Callback):
    def __init__(self, filepath, monitor='val_accuracy', mode='max'):
        super(SaveBestModelVariables, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best = -float('inf') if mode == 'max' else float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is not None:
            if (self.mode == 'max' and current > self.best) or (self.mode == 'min' and current < self.best):
                self.best = current
                trainable_variables = self.model.trainable_variables
                # Set font and font size设置字体和字号
                plt.rcParams['font.family'] = 'Arial'
                plt.rcParams['font.size'] = 20  

                # Use the model to make predictions
                y_pred = model_P1.predict(x_test, batch_size=16)

                # Convert prediction results to class labels
                y_pred_classes = np.argmax(y_pred, axis=1)

                # Calculate confusion matrix
                confusion_mtx = confusion_matrix(y_test, y_pred_classes)

                # Plot confusion matrix using seaborn
                plt.figure(figsize=(10, 10))
                sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap=plt.cm.Blues) 
                plt.xlabel('Predicted label')
                plt.ylabel('True label')

                # Save the image
                plt.savefig('confusion_matrix_cifar10_all.pdf')
                with open(self.filepath, 'wb') as f:
                    pickle.dump(trainable_variables, f)
                print(f"Epoch {epoch + 1}: {self.monitor} improved to {current}, saving model variables to {self.filepath}")

# Create custom callback function
save_best_model_variables = SaveBestModelVariables(filepath='best_model_variables.pkl', monitor='val_accuracy', mode='max')

#training without data qugumation
model_P1.fit(x_train, y_train, validation_data=(x_test, y_test),epochs=40, batch_size=16, callbacks=[csv_logger, save_best_model_variables])
#training with data qugumation
#model_P1.fit(train_ds, validation_data=test_ds,epochs=40, batch_size=16, callbacks=[csv_logger, save_best_model_variables])
