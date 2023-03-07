---
layout: single
title:  "실습 예제1"
categories: thoughts
tag: [python]
use_math: true
---


**`COPYRIGHT(C) 2022 THE CYBER UNIVERSITY OF KOREA ALL RIGHTS RESERVED.`**

본 파일의 외부 배포를 금지합니다.

# 1번과제 - Autoencoder와 Variational Autoencoder (33점)



```python
# gdown upgrade
!pip install --upgrade gdown==4.6.0
```

**템플릿**  
1번 과제의 템플릿 코드는 최종 과제와의 비교를 위해서 필요하므로 수정하지 않도록 합니다.  


```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, LeakyReLU, Flatten
from tensorflow.keras.layers import Dense, Conv2DTranspose
from tensorflow.keras.layers import Reshape, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

import os

# 재현 가능한 난수 생성
np.random.seed(0)
tf.random.set_seed(0)

z_dim = 2

# 여기에 SamplingLayer를 추가합니다.

# `tf.keras.backend.random_normal` 함수는 무엇을 하는 것인지 직접 찾아보고 적어주세요.
ans01 = """
여기에 기입하세요.
"""

# `tf.keras.backend.random_normal` 함수를 호출할 때 `shape=(batch, dim)`으로 설정하였습니다.  
#  이렇게 `shape`을 지정한 이유에 대해 적어주세요.
ans02 = """
여기에 기입하세요.
"""

encoder_input = keras.Input(shape=(28, 28, 1), name='encoder_input')
x = Conv2D(32, 3, strides=1, padding="same", name='encoder_conv_0')(encoder_input)
x = LeakyReLU()(x)
x = Conv2D(64, 3, strides=2, padding="same", name='encoder_conv_1')(x)
x = LeakyReLU()(x)
x = Conv2D(64, 3, strides=2, padding="same", name='encoder_conv_2')(x)
x = LeakyReLU()(x)
x = Conv2D(64, 3, strides=1, padding="same", name='encoder_conv_3')(x)
x = LeakyReLU()(x)
shape_before_flattening = K.int_shape(x)[1:]
x = Flatten()(x)
# VAE를 위하여 mu와 log_var를 출력하고 Sampling Layer를 통하여 z를 출력하도록 변경합니다.
encoder_output= Dense(z_dim, name='encoder_output')(x)
encoder = keras.Model(encoder_input, encoder_output, name='encoder')

decoder_input = keras.Input(shape=(z_dim,), name='decoder_input')
x = Dense(np.prod(shape_before_flattening))(decoder_input)
x = Reshape(shape_before_flattening)(x)
x = layers.Conv2DTranspose(64, 3, strides=1, padding="same", name='decoder_conv_t0')(x)
x = LeakyReLU()(x)
x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", name='decoder_conv_t1')(x)
x = LeakyReLU()(x)
x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", name='decoder_conv_t2')(x)
x = LeakyReLU()(x)
x = layers.Conv2DTranspose(1, 3, strides=1, padding="same", name='decoder_conv_t3')(x)
decoder_output = Activation('sigmoid')(x)
decoder = keras.Model(decoder_input, decoder_output, name="decoder")

class AutoEncoder(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    def call(self,inputs):
        latent = self.encoder(inputs)
        return self.decoder(latent)

@tf.function
def r_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis = [1,2,3])

# 여기에 VAEModel 코드를 추가합니다.

LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 200

# AutoEncoder에서 VAEModel로 model을 변경합니다.
model = AutoEncoder(encoder, decoder)

# 다음 코드에서, compile시에 loss 부분은 삭제합니다. 
# VAEModel의 경우 model의 train_step()에서 직접 손실함수를 계산하기 때문입니다.
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=r_loss)

# mnist 데이터 읽어오기
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))

x_train = x_train/255.
x_test = x_test/255.

model.fit(x_train[:1000], x_train[:1000], epochs=EPOCHS, batch_size=BATCH_SIZE)

n_to_show = 5000
grid_size = 15
figsize = 12

example_idx = np.random.choice(range(len(x_test)), n_to_show)
example_images = x_test[example_idx]
example_labels = y_test[example_idx]

# VAEModel의 encoder의 경우 mu, log_var, z_points의 3개가 출력되므로, 그에 맞도록 다음 줄의 코드를 수정합니다.
z_points = model.encoder.predict(example_images)

plt.figure(figsize=(figsize, figsize))
plt.scatter(z_points[:, 0] , z_points[:, 1] , cmap='rainbow' , c= example_labels
            , alpha=0.5, s=2)
plt.colorbar()
plt.xlim(-12.5, 12.5)
plt.ylim(-12.5, 12.5)
plt.show()

ans03 = z_points.copy()

# Autoencoder와 비교해서 latent vector의 분포가 어떻게 달라졌는지를 ans04에 기입합니다.
ans04 = """
여기에 기입하세요.
"""

# 분포가 VAE처럼 변화될 경우 Autoencoder보다 어떤 장점이 있는 지를 ans05에 기입합니다.
ans05 = """
여기에 기입하세요.
"""
```

**과제 내용**  

**1. Autoencoder**    

템플릿의 코드는 MNIST 필기체 이미지의 autoencoder입니다.
(12주차 실습파일인 `11_gan.ipynb`에 포함된 것과 거의 동일합니다.)  
입력 이미지 input_image가 입력될 때, 인코더는 2차원의 latent vector $z$를 출력합니다.  
`z = model.encoder(input_image)`  
이 latent vector $z$를 decoder에 입력하면, 생성이미지 reconst_image가 출력됩니다.  
`reconst_image = model.decoder(z)`  
    
Loss는 추론값인 `reconst_image`와 참값이자 입력값인 `input_image`와의 Mean Squared Error로 정의하였습니다.  
    
학습이 완료된 후, test set에서 `n_to_show = 5000`만큼의 이미지를 무작위로 추출하여, 
encoder의 출력값인 `z`를 얻고, 그것을 2차원에 출력한 결과를 그래프로 표시하였습니다.  
템플릿 코드를 실행하고 다시한번 코드와 그 결과를 리뷰합시다.  

**2. Variational Autoencoder(VAE)로의 전환을 위한 Sampling 레이어의 정의 추가**   

VAE는 encoder의 마지막 부분에서 $z$ 대신에 $\mu$와 $\sigma$라는 두 개의 텐서를 출력합니다.  
실제로는 $\sigma$ 대신에 분산의 로그값인 $\log\sigma^2$ (log of variance)로 대신합니다.  
왜일까요?  
$\sigma$는 표준편차로서 0보다 커야한다는 조건이 있는데, 신경망의 출력에서 제어하기 어렵습니다.    
또한 KL divergence의 식에서 $\log\sigma^2$ 항이 있는데 이 항은 $\sigma$가 0에서 $-\infty$가 되어서 네트워크에 부동소수점 오류를 일으키기 쉽습니다.  
따라서 $V\equiv\log\sigma^2$를 신경망의 출력값으로 정하고 그 대신에 $\sigma^2$은 $\exp(V)$로, $\sigma$는 $\exp(V/2)$로 계산하면 손실함수 계산에서 부동소수점 오류없이 계산이 가능해집니다. ($V$값은 $-\infty$부터 $+\infty$의 값을 가질 수 있기 때문입니다.)    
이제 표준정규분포에서 임의의 수 $\epsilon$를 sampling하고, $\mu$와 $\log\sigma^2$를 이용하여 $z = \mu+\sigma\epsilon$를 만드는 Sampling Layer를 정의하겠습니다.  
(실습파일 `12_deepfake.ipynb`에서 정의했던 것과 동일합니다.) 
```python
class Sampling(layers.Layer):
    """Uses (mu, log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        mu, log_var = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return mu + tf.exp(log_var/2) * epsilon
```
위의 sampling layer 코드를 읽고 이해해 봅시다.  
`tf.keras.backend.random_normal` 함수는 무엇을 하는 것인지 직접 찾아보고 `ans01`에 기입합니다.  
`tf.keras.backend.random_normal` 함수를 호출할 때 `shape=(batch, dim)`으로 설정하였습니다.  
이렇게 `shape`을 지정한 이유에 대해 `ans02`에 기입합니다.  
`random_normal()`함수에 대한 기본적인 개념은 다음의 코드를 참조하세요.(두번째 섹션)  
https://colab.research.google.com/github/kotech1/computervision/blob/master/appendix/12_norm_gradient.ipynb#scrollTo=2AHB-LtQajIN  
Sampling layer 코드를 과제 기입란에 추가합니다.
    
**3. VAE 전환을 위하여 encoder 모델의 변경**    

템플릿의 Autoencoder의 encoder부분의 마지막 코드는 다음과 같습니다.
Flatten 레이어 다음에 Latent space의 차원(z_dim)으로 축소하기위해 FC(Dense레이어)를 이용합니다.
```python
encoder_output= Dense(z_dim, name='encoder_output')(x)
encoder = keras.Model(encoder_input, encoder_output, name='encoder')
```
위의 코드를 다음과 같이 변경합니다.
```python
mu = Dense(z_dim, name='mu')(x)
log_var = Dense(z_dim, name='log_var')(x)
z = Sampling(name='encoder_output')([mu, log_var])
encoder_output = [mu, log_var, z]
encoder = keras.Model(encoder_input, encoder_output, name='encoder')
```
VAE를 위하여 두개의 출력($\mu$, $\log\sigma^2$)을 각각 `Dense()`를 이용하여 만들고,  
이것을 입력으로하는 Sampling 레이어를 통과하여 z를 출력합니다.  
encoder의 출력은 Autoencoder의 경우는 $z$(`encoder_output`)뿐이었으나, VAE의 경우에는  
$\mu$, $\log\sigma^2$, $z$의 3개 텐서를 출력합니다.  
$z$는 decoder의 입력으로 사용이 되고 $\mu$와 $\log\sigma^2$는 손실함수의 계산에 사용됩니다.  
이 코드는 모두 실습 시간에 다루었던 내용입니다.  
    
**4. VAEModel의 정의**  

이제 전체 VAE 모델을 정의하고 코드에 추가합니다.
```python
class VAEModel(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean( # batch에 대해 평균
                tf.reduce_sum( # 이미지의 각 pixel에 대해 합산
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)
                )
            )
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": tf.reduce_mean(total_loss),
            "reconstruction_loss": tf.reduce_mean(reconstruction_loss),
            "kl_loss": tf.reduce_mean(kl_loss),
        }

    def call(self,inputs):
        _,_,latent = self.encoder(inputs)
        return self.decoder(latent)
```
가장 크게 달라진 점은 `train_step()`의 재정의 부분입니다.  
`train_step()`은 모델이 학습될 때 한번의 batch마다 호출되는 기본 학습 함수입니다.  
보통의 모델에서는 `keras.Model` 객체에 정의된 기본 모듈을 호출하는 것으로 충분하였습니다.  
(우리가 `model.fit()`을 호출하면 batch마다 `keras.Model`에 정의된 `train_step()`이 내부적으로 호출됩니다.)  
VAE에서는 이것을 customize해야 합니다.  
이것은 손실함수가 보통의 경우 $\hat y$(추론결과)와 $y$(참값)만으로 계산되는 것에 비해서,  
VAE에서는 KL divergence 손실 항 때문에, $\mu$와 $\log\sigma^2$라는 모델의 중간 출력(여기서는 encoder부분의 출력)을  
사용해야 하기 때문입니다.  

`with Gradient.tape()`으로 둘러쌓여진 코드는 손실함수를 계산하는 부분입니다.  
손실함수를 계산하면서 그 미분(gradient)을 추적할 수 있도록 합니다.  
`grads = tape.gradient(...)`는 gradient 값을 계산하는 부분입니다.  
keras에서의 gradient를 계산하는 부분은 다음의 예제 코드를 보시면 쉽게 이해하실 수 있습니다.  
https://colab.research.google.com/github/kotech1/computervision/blob/master/appendix/12_norm_gradient.ipynb#scrollTo=4gromgOST0am  
`self.optimizer.apply_gradients()`는 계산된 각 gradient 값을 해당 weights에 optimizer를 이용하여 변경하는 부분입니다.  
이 부분이 실제 학습 parameter가 update되는 부분이라고 하겠습니다.  

관심있으신 분들은 `train_step()`의 원래 소스도 한번 비교해 봐 주십시오.  
https://github.com/keras-team/keras/blob/master/keras/engine/training.py  
1016줄부터 1050줄까지가 해당 함수입니다.  
거의 유사한 구조로 되어있음을 알 수 있습니다.  

이제 모델 생성 부분을 `Autoencoder()`에서 새로 정의한 `VAEModel()`로 다음과 같이 변경합니다.
```python
#model = AutoEncoder(encoder, decoder) #삭제 처리
model = VAEModel(encoder, decoder) # VAEModel로 변경
```

`model.compile()`부분에서 loss를 지정한 부분을 빼도록 합시다.  
VAE모델에서는 손실함수를 `train_step()`에서 직접 처리하였으므로 손실함수의 지정이 필요없기 때문입니다.  
다음과 같이 `model.compile()` 호출 부분을 수정합니다.  
```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))
```

`example_images`를 이용해서 latent 벡터를 호출하는 부분을 다음과 같이 변경합니다.  
VAE용 encoder에서는 z만 반환하는 것이 아니라, `mu`, `log_var`, `z`의 3개 텐서를 tuple로 반환하기 때문입니다.  
```python
#z_points = model.encoder.predict(example_images) # Autoencoder용
_,_,z_points = model.encoder.predict(example_images) # VAE용
```
생성된 `z_points`의 복사본을 `ans03`에 저장합니다. (이미 템플릿에 코드가 포함되어 있습니다.)  
    
    
**4. VAE의 결과 분석**  

이제 변경된 코드를 실행하고 결과를 분석합니다.  
Autoencoder와 비교해서 latent vector의 분포가 어떻게 달라졌는지를 `ans04`에 기입합니다.  
이렇게 분포가 VAE처럼 변화될 경우 Autoencoder보다 어떤 장점이 있는 지를 `ans05`에 기입합니다.
    

**과제 기입란**


```python
# 여기에 템플릿을 복사하고 수정하여 완성합니다.

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, LeakyReLU, Flatten
from tensorflow.keras.layers import Dense, Conv2DTranspose
from tensorflow.keras.layers import Reshape, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

import os

# 재현 가능한 난수 생성
np.random.seed(0)
tf.random.set_seed(0)

z_dim = 2

# 여기에 SamplingLayer를 추가합니다.
class Sampling(layers.Layer):
    """Uses (mu, log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        mu, log_var = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return mu + tf.exp(log_var/2) * epsilon

# `tf.keras.backend.random_normal` 함수는 무엇을 하는 것인지 직접 찾아보고 적어주세요.
ans01 = """
정규분포에서 무작위 값을 출력하는 함수입니다.
"""

# `tf.keras.backend.random_normal` 함수를 호출할 때 `shape=(batch, dim)`으로 설정하였습니다.  
#  이렇게 `shape`을 지정한 이유에 대해 적어주세요.
ans02 = """
출력 tensor의 차원을 일치시키기 위함입니다.
"""

encoder_input = keras.Input(shape=(28, 28, 1), name='encoder_input')
x = Conv2D(32, 3, strides=1, padding="same", name='encoder_conv_0')(encoder_input)
x = LeakyReLU()(x)
x = Conv2D(64, 3, strides=2, padding="same", name='encoder_conv_1')(x)
x = LeakyReLU()(x)
x = Conv2D(64, 3, strides=2, padding="same", name='encoder_conv_2')(x)
x = LeakyReLU()(x)
x = Conv2D(64, 3, strides=1, padding="same", name='encoder_conv_3')(x)
x = LeakyReLU()(x)
shape_before_flattening = K.int_shape(x)[1:]
x = Flatten()(x)
# VAE를 위하여 mu와 log_var를 출력하고 Sampling Layer를 통하여 z를 출력하도록 변경합니다.
mu = Dense(z_dim, name='mu')(x)
log_var = Dense(z_dim, name='log_var')(x)
z = Sampling(name='encoder_output')([mu, log_var])
encoder_output = [mu, log_var, z]
encoder = keras.Model(encoder_input, encoder_output, name='encoder')

decoder_input = keras.Input(shape=(z_dim,), name='decoder_input')
x = Dense(np.prod(shape_before_flattening))(decoder_input)
x = Reshape(shape_before_flattening)(x)
x = layers.Conv2DTranspose(64, 3, strides=1, padding="same", name='decoder_conv_t0')(x)
x = LeakyReLU()(x)
x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", name='decoder_conv_t1')(x)
x = LeakyReLU()(x)
x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", name='decoder_conv_t2')(x)
x = LeakyReLU()(x)
x = layers.Conv2DTranspose(1, 3, strides=1, padding="same", name='decoder_conv_t3')(x)
decoder_output = Activation('sigmoid')(x)
decoder = keras.Model(decoder_input, decoder_output, name="decoder")

class AutoEncoder(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    def call(self,inputs):
        latent = self.encoder(inputs)
        return self.decoder(latent)

@tf.function
def r_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis = [1,2,3])

# 여기에 VAEModel 코드를 추가합니다.
class VAEModel(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean( # batch에 대해 평균
                tf.reduce_sum( # 이미지의 각 pixel에 대해 합산
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)
                )
            )
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": tf.reduce_mean(total_loss),
            "reconstruction_loss": tf.reduce_mean(reconstruction_loss),
            "kl_loss": tf.reduce_mean(kl_loss),
        }

    def call(self,inputs):
        _,_,latent = self.encoder(inputs)
        return self.decoder(latent)

LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 200

# AutoEncoder에서 VAEModel로 model을 변경합니다.
# model = AutoEncoder(encoder, decoder) #삭제 처리
model = VAEModel(encoder, decoder) # VAEModel로 변경

# 다음 코드에서, compile시에 loss 부분은 삭제합니다. 
# VAEModel의 경우 model의 train_step()에서 직접 손실함수를 계산하기 때문입니다.
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=r_loss) # 삭제 처리
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))

# mnist 데이터 읽어오기
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))

x_train = x_train/255.
x_test = x_test/255.

model.fit(x_train[:1000], x_train[:1000], epochs=EPOCHS, batch_size=BATCH_SIZE)

n_to_show = 5000
grid_size = 15
figsize = 12

example_idx = np.random.choice(range(len(x_test)), n_to_show)
example_images = x_test[example_idx]
example_labels = y_test[example_idx]

# VAEModel의 encoder의 경우 mu, log_var, z_points의 3개가 출력되므로, 그에 맞도록 다음 줄의 코드를 수정합니다.
# z_points = model.encoder.predict(example_images) # Autoencoder용
_,_,z_points = model.encoder.predict(example_images) # VAE용

plt.figure(figsize=(figsize, figsize))
plt.scatter(z_points[:, 0] , z_points[:, 1] , cmap='rainbow' , c= example_labels
            , alpha=0.5, s=2)
plt.colorbar()
plt.xlim(-12.5, 12.5)
plt.ylim(-12.5, 12.5)
plt.show()

ans03 = z_points.copy()

# Autoencoder와 비교해서 latent vector의 분포가 어떻게 달라졌는지를 ans04에 기입합니다.
ans04 = """
분포가 크지않고, 같은 분류끼리 밀집도가 높게 형성되어 있습니다.
"""

# 분포가 VAE처럼 변화될 경우 Autoencoder보다 어떤 장점이 있는 지를 ans05에 기입합니다.
ans05 = """
같은 분류끼리 비교적 밀집도가 높게 형성되어 있고, 특성이 확실한 분류는 구별이 되는 장점이 있습니다.
"""
```

# 2번과제 - Latent 공간에서의 Smiling 벡터 적용하기 (33점)

**템플릿**


```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, LeakyReLU, Flatten
from tensorflow.keras.layers import Dense, Conv2DTranspose
from tensorflow.keras.layers import Reshape, Activation
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
import os
from glob import glob
import gdown
import matplotlib.pyplot as plt

# image_file, z_start, z_end 값을 `12_deepfake.ipynb`실습 실행 부분에서 복사하여 업데이트 합니다.
image_file = '008629.jpg'
z_start = np.array([ 0.51165247  , -1.1973262   , -0.18806714  ,  0.3593927   ,
       -0.9896692   ,  0.045576066 , -2.888544    ,  1.4802008   ,
       -0.2041829   , -1.1171055   ,  0.5371562   ,  0.42627877  ,
       -1.2309686   , -2.0550058   , -0.7537902   , -0.07680813  ,
        2.0452733   , -1.1311386   , -0.46550226  ,  1.5755489   ,
        0.58928674  ,  1.1791966   ,  0.94917655  ,  0.5987749   ,
        0.5815001   ,  0.28586265  ,  0.06696689  ,  0.9477665   ,
        1.8414605   ,  1.9119034   ,  1.0567036   ,  0.3615513   ,
       -0.4104105   , -0.59189254  ,  1.2174647   ,  0.9946472   ,
        0.5241927   ,  2.1389232   , -0.51414186  , -0.7504797   ,
       -0.19267002  ,  1.0049567   ,  0.006421149 ,  0.82332313  ,
       -1.2039635   ,  0.4026336   ,  0.59752977  , -0.6505483   ,
       -0.14467421  , -1.1081607   , -1.0387355   ,  0.22033802  ,
        1.2906396   , -0.7465883   , -1.3119192   ,  1.8650314   ,
        0.7424985   ,  0.11610429  ,  0.53767824  ,  0.53099585  ,
       -0.062205195 , -0.43539634  , -0.49824527  ,  2.2648535   ,
        2.1154332   , -1.6803362   , -0.3174094   , -1.7905765   ,
        0.9095981   , -0.9161087   , -1.2652929   ,  1.1605979   ,
        1.5348216   ,  0.41232628  ,  0.008829355 ,  0.9567587   ,
       -0.8195963   , -2.156269    , -2.6899538   ,  0.89053595  ,
        0.83673096  , -1.840965    , -0.633362    ,  1.0299332   ,
       -1.9534174   , -0.85963917  , -0.20637904  ,  2.2067776   ,
       -0.21448427  ,  0.7314191   ,  1.7930303   ,  0.4430193   ,
       -0.7764752   ,  0.22775775  , -3.3790393   , -1.9394097   ,
       -1.1407137   ,  0.48087054  ,  1.0355675   ,  2.1436706   ,
        0.0357095   ,  0.47831464  , -0.17529714  ,  1.6886916   ,
       -0.60205996  , -2.149939    , -0.0059643984, -1.1478593   ,
        0.54694295  ,  2.3754637   , -0.31710178  ,  0.5101158   ,
        3.7773082   , -2.9317229   ,  1.9140741   ,  2.9450448   ,
        1.1843758   , -0.42405456  ,  1.2523038   ,  1.0985551   ,
       -1.1079183   , -0.9630812   , -0.18411824  ,  1.7597651   ,
        1.8354247   , -1.2998254   ,  1.515741    , -0.42342514  ,
        0.24223435  , -0.018041998 , -0.6703321   , -0.25324938  ,
       -1.5229226   , -0.82424057  ,  1.5374343   ,  0.9183435   ,
        0.77907187  , -0.6780119   ,  0.5217359   ,  0.38324603  ,
        1.038603    ,  0.79478425  ,  0.4412514   , -0.7481872   ,
        0.07701512  , -0.36489725  , -0.057156853 ,  0.3818175   ,
       -0.38217247  , -1.7098536   , -1.6881626   ,  0.99543333  ,
        1.6156795   , -0.24023664  , -0.7471294   ,  0.4793784   ,
        0.24937457  ,  0.12624429  , -1.4913517   , -0.73625934  ,
        0.5789245   , -0.5684443   , -0.72188705  ,  0.12764175  ,
       -2.6191442   ,  0.700166    ,  0.39041132  ,  0.08008385  ,
       -0.37719724  , -1.0280637   ,  0.21381554  ,  0.07182546  ,
        1.173063    ,  0.23851879  , -1.134413    ,  0.628986    ,
       -1.6083932   , -0.7461231   ,  0.13498461  ,  0.32760397  ,
        0.8181643   , -2.3889465   , -0.030497007 , -0.07983138  ,
        0.5969469   ,  0.61000985  ,  1.461432    , -0.3340329   ,
        1.7357419   ,  0.49665862  ,  0.47224945  ,  0.75099295  ,
        2.3244333   ,  0.3104105   ,  1.5840006   , -0.053501666 ,
       -1.2293062   , -1.2836668   ,  3.0475667   ,  0.2972155   ],
      dtype='float32')
z_end = np.array([ 5.72977960e-01, -1.17467523e+00, -1.95226938e-01,  2.44418800e-01,
       -1.07260275e+00, -3.16603482e-02, -3.04139423e+00,  1.54305375e+00,
       -1.64888859e-01, -8.53234768e-01,  4.66521442e-01,  3.83821785e-01,
       -1.43313909e+00, -1.87543082e+00, -4.99349624e-01, -1.70870394e-01,
        1.77098560e+00, -1.21392393e+00, -4.80084658e-01,  1.64434516e+00,
        1.88662839e+00,  1.13736928e+00, -1.17789865e-01,  5.08254766e-01,
        5.75029373e-01,  3.71077269e-01,  1.09070159e-01,  1.00230491e+00,
        1.69477463e+00,  1.94024813e+00,  1.07472181e+00,  2.08851844e-01,
       -4.09724802e-01, -5.65167069e-01,  1.23436999e+00,  4.20415044e-01,
        4.03792560e-01,  2.17169404e+00, -5.13979256e-01, -7.00644433e-01,
       -4.12017405e-01,  9.01915193e-01,  4.75870430e-01,  8.44886363e-01,
       -1.25241423e+00,  4.50602233e-01,  6.26643658e-01, -6.78528130e-01,
       -5.26039004e-01, -7.46038795e-01, -7.62022972e-01,  2.15544313e-01,
        3.76785815e-01, -5.60305238e-01, -1.29251766e+00,  1.88069797e+00,
        6.80058718e-01, -7.69171864e-03,  5.51092267e-01,  5.25287628e-01,
       -1.20652840e-03, -4.22662735e-01, -3.14550579e-01,  2.87449431e+00,
        1.78322077e+00, -2.02736044e+00, -2.50060737e-01, -1.69495666e+00,
        8.50703776e-01, -8.20137918e-01, -1.39862287e+00,  1.12130570e+00,
        1.63355219e+00,  1.56464845e-01,  2.28159390e-02,  1.48026490e+00,
       -7.36107945e-01, -2.69309950e+00, -2.67915940e+00,  9.00430441e-01,
        8.59273136e-01, -1.87040365e+00, -5.33185482e-01,  9.74141240e-01,
       -2.00012422e+00, -8.29578996e-01, -4.01541889e-01,  2.28007412e+00,
       -2.36341104e-01,  6.12758756e-01,  1.82076514e+00,  5.61124325e-01,
       -8.30627978e-01,  3.54324281e-01, -2.40655708e+00, -1.78879988e+00,
       -1.16929054e+00,  4.51794624e-01,  4.65384901e-01,  2.12182879e+00,
        3.66598874e-01,  2.29912758e-01, -3.13030750e-01,  1.51989603e+00,
       -3.05434644e-01, -2.09333229e+00, -2.84276865e-02, -1.11931121e+00,
        4.25237447e-01,  2.52315187e+00, -2.79049665e-01,  3.10506999e-01,
        3.65646696e+00, -2.84021711e+00,  1.77909219e+00,  2.54825330e+00,
        1.07033658e+00,  1.03108704e-01,  1.32091248e+00,  1.03996384e+00,
       -2.13494015e+00, -8.67035031e-01, -1.26776934e-01,  1.75396907e+00,
        1.07416964e+00, -1.34220207e+00,  1.65193868e+00, -4.75422591e-01,
        1.55517697e-01, -1.14617862e-01, -6.67772591e-01, -3.13118905e-01,
       -1.39692926e+00, -8.44218612e-01,  2.50200295e+00,  9.17499244e-01,
        7.17299461e-01, -1.70274711e+00,  6.24216914e-01,  4.03311908e-01,
        6.48391664e-01,  8.70243251e-01,  4.31221575e-01,  2.20583677e-02,
        1.71008140e-01, -3.80540580e-01, -1.39102489e-01,  3.67843717e-01,
       -7.73660719e-01, -1.74468589e+00, -1.73297906e+00,  1.03721666e+00,
        1.66422296e+00, -7.99888730e-01, -1.00115657e+00,  4.83160436e-01,
        2.40218386e-01,  3.61453891e-02, -1.24380159e+00, -1.15480435e+00,
        6.45831466e-01, -3.64400446e-01, -6.47031784e-01,  3.84977460e-02,
       -2.63333535e+00,  5.28283715e-01,  3.88369769e-01,  7.97789097e-02,
       -4.78187680e-01, -9.69784558e-01,  1.41845986e-01,  1.58796579e-01,
        1.13848770e+00,  2.02747107e-01, -8.36311102e-01,  5.71853220e-01,
       -1.48306406e+00, -6.87719882e-01, -1.65692091e-01,  2.45054603e-01,
        7.54548907e-01, -3.58206987e+00, -9.21691954e-02,  1.52216822e-01,
        5.73537290e-01,  6.43846810e-01,  1.43375707e+00, -3.37227434e-01,
        1.85679078e+00,  4.50727373e-01,  4.55063194e-01,  8.84635687e-01,
        2.12412310e+00,  4.25927699e-01,  1.81570625e+00, -6.59308583e-02,
       -1.15431726e+00, -1.32918453e+00,  2.89771724e+00,  2.50184685e-01],
      dtype='float32')


INPUT_DIM = (128,128,3)
BATCH_SIZE = 32

md5 = 'b387a8f59bd8bc09ee1eb12a80294379'  
url = 'https://drive.google.com/uc?id=19m6cQVNqXRhD6iEGkjA8ZZcOdev_b2V1'
output = 'vae_weights.tar.gz'

# 모델 weights 다운로드
gdown.cached_download(url, output, md5=md5)

# 폴더 만들기 (리눅스 명렁어 실행)
!mkdir -p vae_data
# 다운로드한 압축 파일 해제 (리눅스 명령어 실행)
!tar xvzf vae_weights.tar.gz -C vae_data

class Sampling(layers.Layer):
    """Uses (mu, log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        mu, log_var = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return mu + tf.exp(log_var/2) * epsilon

z_dim = 200
r_loss_factor = 10000

encoder_input = keras.Input(shape=INPUT_DIM, name='encoder_input')
x = Conv2D(32, 3, strides=2, padding="same", name='encoder_conv_0')(encoder_input)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Conv2D(64, 3, strides=2, padding="same", name='encoder_conv_1')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate = 0.25)(x)
x = Conv2D(64, 3, strides=2, padding="same", name='encoder_conv_2')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate = 0.25)(x)
x = Conv2D(64, 3, strides=2, padding="same", name='encoder_conv_3')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate = 0.25)(x)
shape_before_flattening = K.int_shape(x)[1:]
x = Flatten()(x)
mu = Dense(z_dim, name='mu')(x)
log_var = Dense(z_dim, name='log_var')(x)
z = Sampling(name='encoder_output')([mu, log_var])
encoder = keras.Model(encoder_input, [mu, log_var, z], name = 'encoder')

decoder_input = keras.Input(shape=(z_dim,), name='decoder_input')
x = Dense(np.prod(shape_before_flattening))(decoder_input)
x = Reshape(shape_before_flattening)(x)
x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", name='decoder_conv_t0')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate = 0.25)(x)
x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", name='decoder_conv_t1')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate = 0.25)(x)
x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", name='decoder_conv_t2')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate = 0.25)(x)
x = layers.Conv2DTranspose(3, 3, strides=2, padding="same", name='decoder_conv_t3')(x)
decoder_output = Activation('sigmoid')(x)
decoder = keras.Model(decoder_input, decoder_output, name="decoder")

class VAEModel(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAEModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= r_loss_factor
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": tf.reduce_mean(total_loss),
            "reconstruction_loss": tf.reduce_mean(reconstruction_loss),
            "kl_loss": tf.reduce_mean(kl_loss),
        }

    def call(self,inputs):
        _,_,latent = self.encoder(inputs)
        return self.decoder(latent)

VAE = VAEModel(encoder, decoder)
LEARNING_RATE = 0.0005
VAE.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))

SAVE_FOLDER = 'vae_data'
save_folder = os.path.join(SAVE_FOLDER, 'weights')
VAE.load_weights(save_folder+'/'+'checkpoint')

smiling_vec = (z_end - z_start)/4
factors = [0,1,2,3,4]
fig = plt.figure(figsize=(18, 10))
for counter, factor in enumerate(factors):
    changed_z_point = z_start + smiling_vec * factor
    changed_image = VAE.decoder.predict(np.array([changed_z_point]))[0]
    img = changed_image.squeeze()
    sub = fig.add_subplot(1, len(factors) + 1, 1+counter)
    sub.axis('off')
    sub.imshow(img)


#  z텐서의 배열 내에 `attribute==1` 혹은 `attribute==-1`라는 boolean tensor를 입력하였습니다.  
#  텐서의 배열에 boolean tensor를 넣으면 역할을 하는지 각자 공부해 보고 그 역할에 대해 `ans06`에 기입합니다. 
ans06 = """
여기에 기입하세요.
"""

# `np.sum(z_POS, axis = 0)` 합계를 할 때에 `axis=0`을 주었습니다.
#  그 이유에 대해 생각해보고 `ans07`에 기입합니다.  
ans07 = """
여기에 기입하세요.
"""

#  `if np.sum([movement_POS, movement_NEG]) < 0.08:` 이 조건을 만족시키면 평균 계산 루프가 중단되도록 되어 있습니다.  
#  이 조건은 어떤 의미인지 `ans08`에 기입합니다. 
ans08 = """
여기에 기입하세요.
"""

ans09 = image_file
ans10 = z_start.copy()
ans11 = z_end.copy()
```

**과제 내용**

이번 과제는 13주차 실습에 진행하였던, `12_deepfake.ipynb`에서 latent vector를 움직여서 얼굴의 특징을 변화시키는 내용입니다.  
`12_deepfake.ipynb` 노트북에서 실행하고 여러분이 직접 얼굴 이미지를 선택하여 그 데이터를 복사하고 소스코드를 분석하는 과제입니다.    

이 과제를 통하여, VAE의 소스코드를 조금 더 자세하게 이해하게 될 것입니다.

**1. `12_deepfake.ipynb`실습 파일을 열어서, 처음부터 차례차례 수행합니다.**  

중간에 **Training** cell은 실행하다가 중지해 주십시오. (미리 학습된 weights를 불러와서 진행할 것이기 때문입니다.)  
`def add_vector_to_images(feature_vec):`가 정의된 cell까지 실행해 주십시오.  
(실습파일을 진행할 때 gdown오류가 발생하는 것은 `!pip install --upgrade gdown` 이 cell을 먼저 실행하지 않아서 그렇습니다.  
오류가 발생하면 런타임=>런타임 다시시작을 선택하신후 처음부터 차근차근 진행하도록 합시다.)  

다음 cell은 평균 smiling vector를 추출하는 부분입니다.  
eyeglasses_vec를 구하는 부분을 comment out하고, smiling_vec를 구하는 부분을 수행하도록 다음과 같이 변경합니다. 

```python
BATCH_SIZE = 500
# attractive_vec = get_vector_from_label('Attractive', BATCH_SIZE)
# mouth_open_vec = get_vector_from_label('Mouth_Slightly_Open', BATCH_SIZE)
smiling_vec = get_vector_from_label('Smiling', BATCH_SIZE)
# lipstick_vec = get_vector_from_label('Wearing_Lipstick', BATCH_SIZE)
# young_vec = get_vector_from_label('High_Cheekbones', BATCH_SIZE)
# male_vec = get_vector_from_label('Male', BATCH_SIZE)
# eyeglasses_vec = get_vector_from_label('Eyeglasses', BATCH_SIZE)
# blonde_vec = get_vector_from_label('Blond_Hair', BATCH_SIZE)
```
위와 같은 코드로 수정후에 실행합니다.  
이 코드는 Latent 공간에서 `Smiling` 벡터를 계산하고 산출할 것입니다.  
    
    
**2. `ImageLabelLoader()`함수를 읽고 이해합시다.**  

함수내에서 keras의 DataLoader 함수가 호출됩니다.  
DataLoader는 대용량의 학습데이터를 읽어들이는 모듈의 기본 형식 중 하나입니다.  
keras학습에서 자주 사용되므로, 많이 사용해보고 익숙해 질 필요가 있습니다.  
학습 데이터의 크기가 대용량이 되면, 거의 대부분 DataLoader의 형식으로 읽어서 `model.fit()`에 입력으로 사용합니다.  
학습이 진행되는 동안 GPU가 사용될 때, 남는 CPU자원을 활용하여 미리 데이터를 읽어서 학습 속도를 향상시키는 기능등으로 확장 가능하기 때문입니다.  
(이번 VAE에서는 이러한 prefetch의 기능은 사용하지 않았습니다.)  

함수내의 `flow_from_dataframe()` 라이브러리에 대해 학습합시다.  
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_dataframe  
(번역본)  
https://keras.io/ko/preprocessing/image/  

`ImageLabelLoader()`함수는 두 종류로 데이터를 생성할 수 있도록 준비되어 있습니다. `build`시에 `label`을 지정하거나, 지정하지 않거나 할 경우 입니다.  
`label` 지정시에, `y_col=label`로서, CSV파일의 해당 column을 y값으로 반환합니다. 이 경우 `class_mode='raw'`를 이용합니다.  
이렇게 한 batch의 데이터를 loading하게 되면, 
```python
data_flow_label = imageLoader.build(att, batch_size, label = label)
batch = next(data_flow_label) # 한 배치를 추출해서 가져옴.
```
`batch[0]`에는 이미지의 batch 데이터가, `batch[1]`에는 해당 칼럼의 속성값의 batch가 들어오게 됩니다.  
이렇게 준비한 데이터는 latent 공간에서 특징 벡터를 추출하는 `get_vector_from_label()`에서 사용됩니다.  

`label` 미지정시에는 `class_mode='input'`을 이용하고, 이 경우 `batch[0]`, `batch[1]` 모두 이미지 batch 데이터가 들어있습니다.  
실습 파일에서는 무작위로 이미지를 불러오는 부분에서 사용하였습니다.  

시간이 나시는 분들은 `flow_from_directory()`에 대해서도 공부합시다.  
(자주 사용되는 DataLoader입니다.)  
실습파일에서 학습 이미지를 불러오는 것에 사용되었습니다.  
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_directory  
(번역본)  
https://keras.io/ko/preprocessing/image/   
    
    
**3. 실습파일에서 `get_vector_from_label()` 함수의 정의를 읽어보고 분석합니다.**  

`label`에 속성 column명이 지정되었을 경우, 해당 속성과 해당 속성이 없는 것을 분류하고 각각의 latent vector의 평균을 구한 후에,
두 평균 vector의 차이를 구하는 함수입니다. 
해당 속성이 있는 것을 positive(`POS`), 없는 것을 negative(`NEG`)라고 변수명에 postfix로 사용하고 있습니다.  

먼저 충분한 갯수만큼 loop가 수행됩니다. 
```python
while (current_n_POS < 10000):
```
다음은 이미지와 속성을 한 batch만큼 가져오는 DataLoader 부분입니다.
```python
batch = next(data_flow_label)
im = batch[0]
attribute = batch[1]
```
이제 이미지로부터 latent vector $z$를 출력합니다.  
```python
_,_,z = vae.encoder.predict(np.array(im))
```
다음은 z의 batch로부터 attribute가 positive인 경우와 negative인 경우로 데이터를 분리하는 모듈입니다.
```python
z_POS = z[attribute==1]
z_NEG = z[attribute==-1]
```
z텐서의 배열 내에 `attribute==1` 혹은 `attribute==-1`라는 boolean tensor를 입력하였습니다.  
텐서의 배열에 boolean tensor를 넣으면 역할을 하는지 각자 공부해 보고 그 역할에 대해 `ans06`에 기입합니다.

다음 부분은 계속해서 `POS`와 `NEG` 속성의 latent vector z를 계속 더하고 그 갯수를 기록해서 평균값을 계산할 수 있도록 합니다.
```python
if len(z_POS) > 0:
    current_sum_POS = current_sum_POS + np.sum(z_POS, axis = 0)
    current_n_POS += len(z_POS)
    new_mean_POS = current_sum_POS / current_n_POS
    movement_POS = np.linalg.norm(new_mean_POS-current_mean_POS)

if len(z_NEG) > 0: 
    current_sum_NEG = current_sum_NEG + np.sum(z_NEG, axis = 0)
    current_n_NEG += len(z_NEG)
    new_mean_NEG = current_sum_NEG / current_n_NEG
    movement_NEG = np.linalg.norm(new_mean_NEG-current_mean_NEG)
```
여기서 `np.sum(z_POS, axis = 0)` 합계를 할 때에 `axis=0`을 주었습니다.
그 이유에 대해 생각해보고 `ans07`에 기입합니다.  
`if np.sum([movement_POS, movement_NEG]) < 0.08:` 이 조건을 만족시키면 평균 계산 루프가 중단되도록 되어 있습니다.  
이 조건은 무엇을 의미하는지 `ans08`에 기입합니다.  

최종적으로 두 평균벡터의 차이를 계산한 후에 벡터를 normalize합니다. (길이가 1이되도록 만듭니다.)  
```python
current_vector = current_vector / current_dist
```
우리가 두 속성 차이 벡터의 방향만 알 수 있고, 그 절대 크기를 알 수 없으므로 latent vector에 이것을 가감할때에,  
적당한 factor를 곱해주면서 테스트 해 보아야 합니다.  
    
    
**4. 입력 이미지의 latent vector에 속성 vector를 더하는 함수**  

기존의 `add_vector_to_images()`함수를 약간 변형하여 `add_vector_to_images2()`함수를 만들었습니다.  
`12_deepfake.ipynb`에 cell을 추가하고 다음 코드를 추가합니다.  
```python
def add_vector_to_images2(feature_vec):
    n_to_show = 5
    factors = [0,1,2,3,4]

    example_batch = next(data_flow_generic)
    example_images = example_batch[0]
    example_labels = example_batch[1]

    _,_,z_points = vae.encoder.predict(example_images)

    fig = plt.figure(figsize=(18, 10))

    counter = 1
    for i in range(n_to_show):
        img = example_images[i].squeeze()
        sub = fig.add_subplot(n_to_show, len(factors) + 1, counter)
        sub.axis('off')        
        sub.imshow(img)

        counter += 1

        for factor in factors:
            changed_z_point = z_points[i] + feature_vec * factor
            changed_image = vae.decoder.predict(np.array([changed_z_point]))[0]

            img = changed_image.squeeze()
            sub = fig.add_subplot(n_to_show, len(factors) + 1, counter)
            sub.axis('off')
            sub.imshow(img)

            counter += 1
    plt.show()
    return z_points  

print('Smiling Vector')
z_points = add_vector_to_images2(smiling_vec)  

idx = (data_flow_generic.batch_index - 1) * data_flow_generic.batch_size
idx_list = data_flow_generic.index_array[idx : idx + 5].tolist()
image_file_name = [data_flow_generic.filenames[j] for j in idx_list]
```

변경점은 factor를 -4부터 4까지 변화시키는 것이아니라 0부터 4까지만 변화시키도록 하였습니다.  
그리고 z_points를 출력하여, latent 벡터의 값 자체를 확인할 수 있도록 하였습니다.  
실습 파일에 위 cell을 추가하고 실행해 봅시다.  

무작위로 다섯개의 원본이미지와 그 이미지의 latent vector에 smiling vector를 더한 이미지를 출력하도록 하였습니다.  
마음에 드는 이미지의 변화가 나타났나요?  
혹시 마음에 들지 않으면 몇 번 더 cell을 재실행하면서 확인합시다.  
5개 중에 마음에 드는 이미지 번호를 선택합니다. (0번, 1번, 2번, 3번, 4번 중에 선택)  
먼저 맨 왼쪽의 원본이미지가 무표정이고 맨 오른쪽의 변화된 이미지가 웃는 얼굴이면 OK입니다.  
(얼굴은 여러분들의 취향이니, 여러분이 마음에 드는 이미지로 선택합시다.)  

다시 cell을 추가하고 다음의 코드를 추가합니다. 변수 `i`에는 여러분이 선택한 번호를 입력합니다. (0,1,2,3,4 중에 선택)  
다음 코드는 단순히 여러분이 선택한 것과 일치하는 지를 이미지 출력으로 확인하는 과정에 불과합니다.

```python
i = 0 # 몇번째 그림인지 선택하세요 0,1,2,3,4 중에 번호를 변경하세요.

# 여러분이 선택한 이미지가 맞는 지 확인하세요.
factors = [0,1,2,3,4]
img = plt.imread('./vae_data/celeb/img_align_celeba/%s'%image_file_name[i])
fig = plt.figure(figsize=(18, 10))
sub = fig.add_subplot(1, len(factors) + 1, 1)
sub.axis('off')        
sub.imshow(img)
for counter, factor in enumerate(factors):
    changed_z_point = z_points[i] + smiling_vec * factor
    changed_image = vae.decoder.predict(np.array([changed_z_point]))[0]
    img = changed_image.squeeze()
    sub = fig.add_subplot(1, len(factors) + 1, 2+counter)
    sub.axis('off')
    sub.imshow(img)
```

선택이 끝나셨으면, cell을 다시 추가하고 다음의 코드를 실행합니다.

```python
print("image_file = '%s'"%image_file_name[i])
z = z_points[i]
z_start_str = 'z_start = np.'+repr(z).replace("float32", "'float32'")
print(z_start_str)
z_end_str = 'z_end = np.'+repr(z+4*smiling_vec).replace("float32", "'float32'")
print(z_end_str)
```
원본이미지 파일의 번호와 시작 latent vector, 끝 latent vector를 출력하는 코드입니다.  
출력 결과를 메모장에 복사하십시오.  
여기까지 수행하셨으면, `12_deepfake.ipynb`실습파일은 종료하셔도 됩니다.  
    
    
**5. 템플릿을 과제 기입란에 복사한 후에, 메모장에 복사해두었던 코드로 해당 데이터를 업데이트 하십시오.**  

템플릿의 코드는 실습파일의 VAE Model과 거의 동일합니다.  
코드내의 데이터 업데이트가 끝나면, 실행해서 동일한 얼굴이 생성되는 지 확인하세요.  
`image_file`과 `z_start`, `z_end`값을 `ans09`, `ans10`, `ans11`에 기록합니다. (이미 템플릿 코드에 포함되어 있습니다.)  
   
   
**6. 우리는 무엇을 얻었습니까?**  

실습 노트북과, 과제 노트북에서 서로 공유된 것은 Model과 그 weights(학습 저장 데이터)입니다.  
그 이외에는 `z_start`, `z_end`의 두 latent vector를 전달하였습니다. (이미지 파일명은 단순 참고용입니다.)   
이것으로 두 노트북사이에 시작과 끝, 두 개의 이미지를 다시 복원할 수 있습니다.  
혹은 여러분들이 과제 제출을 하게 되면 200개의 z값 만으로 저는 여러분이 선정한 이미지를 복원할 수 있습니다. 

생성된 이미지는 (128,128,3)의 shape입니다. 이 데이터는 약 50KB입니다. 
이 이미지를 JPEG으로 압축하더라도 약 3KB의 용량이 됩니다.  
하지만 전달된 데이터는 float32(4byte) 200개입니다. 총 800byte 입니다.  
얼굴 데이터 셋만을 대상으로 하였을 경우에는 압도적으로 적은 데이터로 얼굴 영상을 전달하였습니다.  
Autoencoder 혹은 Variational Autoencoder의 encoder와 decoder는 각각 영상의 압축 및 해제와 비슷한 기능을 한다고 하겠습니다.  
두 개의 이미지를 가지고 중간 이미지도 재생성 할 수 있다는 것은 보너스입니다.  
(첫번째 이미지가 `z_start`로 부터 생성된 것이고, 다섯번째 이미지가 `z_end`로부터 생성된 것이며, 중간의 세개는 latent공간 상의 중간 점들로부터 만들어진 것입니다.)

모두들 무뚝뚝한 얼굴 이미지를 웃는 얼굴 이미지로 바꾸는데 성공하셨나요?  
여러분의 앞날에도 항상 웃음이 가득하길 기원합니다.  

**과제 기입란**


```python
# 여기에 템플릿을 복사하고 데이터를 수정하여 코드를 완성합니다.
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, LeakyReLU, Flatten
from tensorflow.keras.layers import Dense, Conv2DTranspose
from tensorflow.keras.layers import Reshape, Activation
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
import os
from glob import glob
import gdown
import matplotlib.pyplot as plt

# image_file, z_start, z_end 값을 `12_deepfake.ipynb`실습 실행 부분에서 복사하여 업데이트 합니다.
image_file = '182365.jpg'
z_start = np.array([-0.7857776 , -1.0831673 ,  0.12167341,  0.74563766, -1.8886566 ,
       -0.70017725, -1.9186776 ,  2.070943  ,  0.8208169 ,  1.6189873 ,
       -1.6551436 , -0.5808564 , -0.08176261, -0.02787526, -1.0032463 ,
        0.09791145, -1.4182472 , -0.7649198 , -1.5242203 ,  1.2972594 ,
       -1.1921791 ,  0.34002057, -0.6355352 , -1.3618774 , -1.2139633 ,
        0.18897474, -0.17338507,  0.34917215, -0.1424432 ,  1.2539146 ,
        0.09252285,  0.8849762 ,  1.5560184 ,  0.6755763 , -0.19676924,
       -1.4116725 , -0.16788474,  0.58483565,  0.71764964, -0.34908196,
        0.34162304,  1.1648604 , -0.43399343,  0.02296865, -0.08399577,
       -1.1502635 , -0.5681314 , -0.00666177, -0.86747646, -0.4319929 ,
        0.21958095,  2.537614  ,  0.98936707, -0.67434216, -1.4719253 ,
        0.81731606,  0.21415755, -0.7729034 ,  1.7024475 ,  2.0477931 ,
       -0.09551914,  1.0533173 , -0.1859307 , -0.23313454,  1.7019776 ,
       -0.6743349 , -1.0486696 , -0.83831614, -0.08553559,  0.7914605 ,
       -0.6709576 ,  0.568254  ,  0.3453726 , -0.5705106 , -0.32677922,
        0.30829513,  0.714722  , -0.3566245 , -0.7425292 ,  0.7431099 ,
        0.17161816,  1.8033266 ,  1.0403152 , -1.0410036 ,  0.40595457,
       -1.79278   ,  0.43583375,  0.9066742 ,  0.6428877 , -0.26046872,
       -0.34312317, -0.5930437 ,  0.71753216, -0.8564427 ,  0.14673775,
       -1.1046023 , -0.8736439 , -0.98655295,  1.2229702 , -0.5727693 ,
       -1.2993276 , -0.7115949 , -0.33541125,  0.04330885,  0.67887366,
        1.4149271 , -2.0071726 ,  0.14318693,  0.21423373, -0.8801714 ,
        1.2949407 ,  0.1517939 , -1.2557898 , -1.3676909 , -0.18259569,
        0.6582577 , -1.0108805 , -1.2738644 ,  0.70809877, -1.5486261 ,
       -1.444074  ,  1.7405248 , -0.00310344, -0.30420083,  1.7914203 ,
       -0.44310814, -0.32690126,  1.3412085 ,  1.9615213 , -0.7626301 ,
        0.6524621 , -0.7212972 , -0.9442592 , -0.03343979,  0.01105291,
       -2.0504122 ,  0.80174804, -0.5440545 ,  2.0264046 , -0.17477708,
       -0.58434415,  2.4031065 , -1.2692498 ,  0.55098855,  0.78233486,
        0.55950415, -0.27807584, -0.712328  , -0.36800778, -0.07856395,
        0.02902108, -0.9870939 , -0.9369652 , -0.4216978 , -0.8157892 ,
        0.24327745, -0.82748044, -0.46750015,  0.52135104,  0.20224772,
       -1.9504147 , -0.8235267 , -0.20908995,  0.17402178, -2.0196416 ,
       -0.5514116 ,  1.226393  ,  1.9680924 ,  0.3949719 ,  0.13669854,
        0.30449265,  0.15263024, -0.07868423,  1.268955  ,  0.21578723,
        1.1982564 , -0.03940457,  1.3907814 ,  1.8238695 , -0.82782775,
       -0.5606899 ,  0.8992175 ,  2.7250204 , -1.1844633 , -2.424691  ,
        1.6874193 , -0.15004304, -0.06332335, -1.5636511 ,  1.7721343 ,
        0.76050985, -0.7937807 ,  1.5352987 , -0.53196585, -0.75061625,
       -0.23709965,  0.17588317,  0.8596381 ,  0.3009922 ,  0.15276484],
      dtype='float32')
z_end = np.array([-7.44126379e-01, -1.12093997e+00,  1.03942364e-01,  5.76356828e-01,
       -1.82659400e+00, -7.72283971e-01, -2.03388047e+00,  2.16192532e+00,
        8.47569942e-01,  1.92221332e+00, -1.68894255e+00, -6.10346496e-01,
       -1.74684927e-01,  9.80510414e-02, -7.13772058e-01, -8.56481194e-02,
       -1.64291167e+00, -8.24081540e-01, -1.50173783e+00,  1.32603419e+00,
        3.70699167e-02,  3.02471489e-01, -1.67227197e+00, -1.27567911e+00,
       -1.25405848e+00,  2.37449288e-01, -6.34907037e-02,  4.44472373e-01,
       -2.62023866e-01,  1.35478282e+00,  1.81313992e-01,  8.54560077e-01,
        1.55634558e+00,  6.36815488e-01, -2.37744689e-01, -2.13700461e+00,
       -2.71136433e-01,  7.35099137e-01,  6.33809566e-01, -3.15696836e-01,
        1.59073770e-01,  1.11661911e+00,  6.19207621e-02,  1.09063089e-02,
       -1.54373497e-01, -1.17748427e+00, -5.34898043e-01, -1.19669557e-01,
       -1.30072892e+00, -4.38845158e-02,  5.51878333e-01,  2.62827158e+00,
        8.90816450e-02, -4.83270049e-01, -1.51004958e+00,  8.51346314e-01,
        2.01476455e-01, -8.94471943e-01,  1.72199655e+00,  1.99083018e+00,
       -1.29224628e-01,  1.06288362e+00, -7.32765570e-02,  3.09879154e-01,
        1.35550904e+00, -1.05225217e+00, -1.02936327e+00, -8.27273428e-01,
       -9.13455039e-02,  9.59036589e-01, -7.95891583e-01,  6.15016580e-01,
        3.51062655e-01, -8.26616764e-01, -3.43857825e-01,  7.31218278e-01,
        7.73875237e-01, -8.40355754e-01, -6.83511853e-01,  8.86822224e-01,
        1.49946108e-01,  1.80200398e+00,  1.11508274e+00, -1.00202835e+00,
        3.87099028e-01, -1.76860166e+00,  2.51943111e-01,  8.65807414e-01,
        5.61416149e-01, -4.21944261e-01, -2.79262900e-01, -5.93682587e-01,
        7.59647965e-01, -8.54385197e-01,  1.17380023e+00, -9.85900223e-01,
       -8.99450719e-01, -1.05863535e+00,  7.31153309e-01, -5.48318326e-01,
       -9.92329359e-01, -8.98508430e-01, -4.55562472e-01, -6.98415861e-02,
        9.94596004e-01,  1.46183419e+00, -2.01653028e+00,  2.41548002e-01,
        2.54051685e-01, -6.38616204e-01,  1.28897595e+00, -2.29227543e-03,
       -1.37060022e+00, -1.23276067e+00, -1.99484989e-01,  3.31097156e-01,
       -1.12558019e+00, -7.67637253e-01,  8.04103255e-01, -1.57588971e+00,
       -2.49157143e+00,  1.80143833e+00, -6.78456500e-02, -2.25250840e-01,
        1.08389735e+00, -5.06474555e-01, -1.43423006e-01,  1.31715965e+00,
        1.87350333e+00, -9.22234237e-01,  6.58286273e-01, -8.25068891e-01,
       -1.01507545e+00, -1.10716119e-01,  8.75490606e-01, -2.02703834e+00,
        8.84289205e-01, -1.55314147e+00,  2.15946984e+00, -1.85077608e-01,
       -1.07636905e+00,  2.36605072e+00, -1.21699858e+00,  1.43222094e+00,
        8.09560359e-01,  6.12431347e-01, -2.52724260e-01, -7.60259271e-01,
       -7.17783332e-01, -1.50088221e-01,  1.28426626e-02, -9.65749085e-01,
       -8.24012518e-01, -1.04930985e+00, -9.56266880e-01,  1.98835582e-01,
       -8.16461980e-01, -5.53838015e-01,  7.76388407e-01, -1.56365171e-01,
       -1.95958972e+00, -7.32212365e-01, -2.66176611e-01,  7.53389373e-02,
       -2.11313009e+00, -5.71166992e-01,  1.33055699e+00,  2.01356387e+00,
        3.51107329e-01,  1.66398793e-01,  2.93369561e-01,  1.70064226e-01,
       -4.21603471e-02,  1.27052081e+00,  5.96553683e-01,  1.18067896e+00,
       -3.50442864e-02,  1.52930129e+00,  1.47749817e+00, -9.26050007e-01,
       -6.98280931e-01, -3.82974386e-01,  2.56999016e+00, -1.00733042e+00,
       -2.45067596e+00,  1.72703123e+00, -1.55829728e-01,  9.97790992e-02,
       -1.44556367e+00,  1.81633496e+00,  7.42532432e-01, -7.37433732e-01,
        1.30531287e+00, -4.42202389e-01, -6.33462727e-01, -3.39177072e-01,
        1.86002672e-01,  8.71792793e-01,  1.55165941e-01,  1.02612197e-01],
      dtype='float32')


INPUT_DIM = (128,128,3)
BATCH_SIZE = 32

md5 = 'b387a8f59bd8bc09ee1eb12a80294379'  
url = 'https://drive.google.com/uc?id=19m6cQVNqXRhD6iEGkjA8ZZcOdev_b2V1'
output = 'vae_weights.tar.gz'

# 모델 weights 다운로드
gdown.cached_download(url, output, md5=md5)

# 폴더 만들기 (리눅스 명렁어 실행)
!mkdir -p vae_data
# 다운로드한 압축 파일 해제 (리눅스 명령어 실행)
!tar xvzf vae_weights.tar.gz -C vae_data

class Sampling(layers.Layer):
    """Uses (mu, log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        mu, log_var = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return mu + tf.exp(log_var/2) * epsilon

z_dim = 200
r_loss_factor = 10000

encoder_input = keras.Input(shape=INPUT_DIM, name='encoder_input')
x = Conv2D(32, 3, strides=2, padding="same", name='encoder_conv_0')(encoder_input)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Conv2D(64, 3, strides=2, padding="same", name='encoder_conv_1')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate = 0.25)(x)
x = Conv2D(64, 3, strides=2, padding="same", name='encoder_conv_2')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate = 0.25)(x)
x = Conv2D(64, 3, strides=2, padding="same", name='encoder_conv_3')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate = 0.25)(x)
shape_before_flattening = K.int_shape(x)[1:]
x = Flatten()(x)
mu = Dense(z_dim, name='mu')(x)
log_var = Dense(z_dim, name='log_var')(x)
z = Sampling(name='encoder_output')([mu, log_var])
encoder = keras.Model(encoder_input, [mu, log_var, z], name = 'encoder')

decoder_input = keras.Input(shape=(z_dim,), name='decoder_input')
x = Dense(np.prod(shape_before_flattening))(decoder_input)
x = Reshape(shape_before_flattening)(x)
x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", name='decoder_conv_t0')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate = 0.25)(x)
x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", name='decoder_conv_t1')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate = 0.25)(x)
x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", name='decoder_conv_t2')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate = 0.25)(x)
x = layers.Conv2DTranspose(3, 3, strides=2, padding="same", name='decoder_conv_t3')(x)
decoder_output = Activation('sigmoid')(x)
decoder = keras.Model(decoder_input, decoder_output, name="decoder")

class VAEModel(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAEModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= r_loss_factor
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": tf.reduce_mean(total_loss),
            "reconstruction_loss": tf.reduce_mean(reconstruction_loss),
            "kl_loss": tf.reduce_mean(kl_loss),
        }

    def call(self,inputs):
        _,_,latent = self.encoder(inputs)
        return self.decoder(latent)

VAE = VAEModel(encoder, decoder)
LEARNING_RATE = 0.0005
VAE.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))

SAVE_FOLDER = 'vae_data'
save_folder = os.path.join(SAVE_FOLDER, 'weights')
VAE.load_weights(save_folder+'/'+'checkpoint')

smiling_vec = (z_end - z_start)/4
factors = [0,1,2,3,4]
fig = plt.figure(figsize=(18, 10))
for counter, factor in enumerate(factors):
    changed_z_point = z_start + smiling_vec * factor
    changed_image = VAE.decoder.predict(np.array([changed_z_point]))[0]
    img = changed_image.squeeze()
    sub = fig.add_subplot(1, len(factors) + 1, 1+counter)
    sub.axis('off')
    sub.imshow(img)


#  z텐서의 배열 내에 `attribute==1` 혹은 `attribute==-1`라는 boolean tensor를 입력하였습니다.  
#  텐서의 배열에 boolean tensor를 넣으면 역할을 하는지 각자 공부해 보고 그 역할에 대해 `ans06`에 기입합니다. 
ans06 = """
텐서 배열에 boolean tensor를 넣을 경우 해당 조건을 만족하는 값들만 추출되어 저장됩니다.
"""

# `np.sum(z_POS, axis = 0)` 합계를 할 때에 `axis=0`을 주었습니다.
#  그 이유에 대해 생각해보고 `ans07`에 기입합니다.  
ans07 = """
가로축끼리의 합만을 구하여 평균값의 계산을 반복적으로 수행하기 위해서입니다.
"""

#  `if np.sum([movement_POS, movement_NEG]) < 0.08:` 이 조건을 만족시키면 평균 계산 루프가 중단되도록 되어 있습니다.  
#  이 조건은 어떤 의미인지 `ans08`에 기입합니다. 
ans08 = """
속성이 있는 이미지의 벡터와 없는 벡터의 차이가 0.08보다 적게 날때 계산이 중단되도록 하기 위함입니다.
"""

ans09 = image_file
ans10 = z_start.copy()
ans11 = z_end.copy()
```

# 3번과제 - Fast Gradient Signed Method (34점)



**템플릿**


```python
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

pretrained_model = tf.keras.applications.MobileNetV2(include_top=True,
                                                 weights='imagenet')
pretrained_model.trainable = False

# ImageNet 클래스 레이블
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

# 이미지가 MobileNetV2에 사용하기 위한 level normalization 및 size normalization 함수
def preprocess(image):
  image = tf.cast(image, tf.float32)
  image = image/255
  image = tf.image.resize(image, (224, 224))
  image = image[None, ...]
  return image

# 예측 결과로부터 top1을 뽑아서 라벨을 출력하는 함수
def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]

# Labrador Retreiver 이미지 불러오기
image_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
image_raw = tf.io.read_file(image_path)
image = tf.image.decode_image(image_raw)

image = preprocess(image)
# 추론 
image_probs = pretrained_model.predict(image)

loss_object = tf.keras.losses.CategoricalCrossentropy()

def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)

  # 입력 이미지에 대한 손실 함수의 기울기를 구합니다.
  gradient = tape.gradient(loss, input_image)
  # 왜곡을 생성하기 위해 그래디언트의 부호를 구합니다.
  signed_grad = tf.sign(gradient)
  return signed_grad

# 인식하고 이미지 출력하는 함수 (label과 confidence를 반환하도록 변경)
def display_images(image, description):
  _, label, confidence = get_imagenet_label(pretrained_model.predict(image))
  plt.figure()
  plt.imshow(image[0])
  plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
                                                   label, confidence*100))
  plt.show()
  return label, confidence

# 이미지의 레이블 위치만 1로 설정 (기본적인 클래스 분류기 결과)
labrador_retriever_index = 208
label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
label = tf.reshape(label, (1, image_probs.shape[-1]))

confidence1 = confidence2 = confidence3 = 0.

# 공격 패턴 생성
perturbations = create_adversarial_pattern(image, label)

epsilons = [0, 0.01, 0.1, 0.15]
# epsilon은 0.01로 고정
eps = epsilons[1]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]

#원본 인식 (epsilon = 0)
display_images(image, descriptions[0])

#공격 진행
adv_x = image + eps*perturbations
adv_x = tf.clip_by_value(adv_x, 0, 1)
label0, confidence0 = display_images(adv_x, descriptions[1]) # epsilon = 0.01

# FSGM 결과 
print(label0, confidence0)

res = decode_predictions(image_probs, top=4)
print(res[0])

# 추론결과(image_probs[0])를 역순으로 정렬
top = np.argsort(image_probs[0])[::-1]
# top4만을 출력
print(top[:4])

# 추론 결과의 top4의 label class 및 label index 출력
print()
for r, index in zip(res[0], top[:4]):
    print('class name =', r[1], ', class index =', index)

# model predict의 결과인 `image_probs[0]`에는 무엇이 들어있습니까? 
ans12 = '''
여기에 기입합니다.
'''

# np.argsort(image_probs[0])[::-1]는 무엇을 얻기 위한 것입니까?
ans13 = '''
여기에 기입합니다.
'''

# top2 결과인 `eskimo_dog`의 class index는 얼마입니까?
ans14 = -1 # 값을 변경해 주세요.

# 원본이미지를 인식하였을 때, 4번째로 높은 추론값을 가진 결과(top4)는 무엇이었습니까?
# 인덱스 값으로 적어 주세요
ans15 = -1 # 값을 변경해 주세요.

    
# 이곳에 top2 index를 이용한 공격 code snippte을 추가합니다.

# 이곳에 top3 index를 이용한 공격 code snippte을 수정하여 추가합니다.

# 이곳에 top4 index를 이용한 공격 code snippte을 수정하여 추가합니다.

# 공격 결과 중 confidence score를 다음 변수에 기록합니다.
ans16 = confidence1 # top2 index를 이용한 공격결과의 confidence score
ans17 = confidence2 # top3 index를 이용한 공격결과의 confidence score
ans18 = confidence3 # top4 index를 이용한 공격결과의 confidence score

# 본 과목을 수강하시면서 느낀점, 의견 등이 있으시면 적어 주세요. (공란으로 두셔도 됩니다.)
ans19 = """
여기에 적어주세요.
"""
```

**과제 내용**  

이번 과제는 FGSM의 공격 패턴을 다양화하는 것입니다.  
강의 중의 공격은 원래의 결과가 나오는 것을 방해합니다. (원래 결과의 Loss를 증가하도록 패턴 생성)  
이번 과제는 원래의 결과중에 두번째, 세번째, 네번째 후보 중에 원하는 결과가 나오도록 공격 패턴을 생성합니다.  
(이 과제를 통하여 FGSM의 원리를 보다 더 깊이 이해할 수 있게 됩니다.)  

템플릿의 코드는 `13_fun_topics.ipynb` 실습 코드와 거의 동일합니다.  

다만 $\epsilon$을 0.01로 고정하였습니다. 

**1. FGSM의 간단 리뷰**  

원본 이미지(Labrado retriever)를 추론하고 원본 Label(208번)을 이용한 학습데이터 생성.  
추론값, GT(208 one-hot-encoding)를 이용한 손실함수의 정의(cross entropy).  
손실함수를 모델의 weights가 아닌 입력 이미지에 대해 gradient 계산 => perturbation으로 정의.  
원본 이미지에 $\epsilon$ $\cdot$ `pertrubation`을 **더하기** (손실의 증가)하여 공격 완성.

**2. FGSM은 원본 이미지에 공격 pattern을 주입함으로서 결과추론을 어렵게 하였습니다.**  

이것은 원래 추론결과와 다른 결과를 만들어 내는 것이 공격의 목표였습니다.  
이를 이용하여, 원하는 결과를 도출할 수 있는 간단한 실험을 해 보도록 하겠습니다.  
원리는 다음과 같습니다.  
먼저 원본 인식결과와는 다른 target label을 선정합니다.  
그리고 동일하게 추론 결과와 target label을 이용하여 손실함수(cross entropy)를 정의합니다.  
이것을 이용한 gradient를 계산하고 부호부분만을 계산하여 target_perturbation을 정의합니다.  
원본 이미지에 $\epsilon$ $\cdot$ `target_pertrubation`을 **빼기** 하여 target의 확률을 높입니다.
   
**3. FGSM은 매우 간단한 공격 방법으로 임의의 라벨로 변경하기는 까다롭습니다.**  

따라서 원래 원본 이미지의 결과 중에서 원래 결과인 top1대신 top2, top3, top4를 선택하여 공격자가 원하는 결과로 바꿔 보도록 하겠습니다.  
이를 위하여 template에서 top1부터 top4까지의 결과 및 label index를 출력하도록 템플릿에 마지막 코드를 추가해 두었습니다.  
다음은 템플릿 코드의 일부입니다.    
```python
res = decode_predictions(image_probs, top=4)
print(res[0])
# 추론결과(image_probs[0])를 역순으로 정렬
top = np.argsort(image_probs[0])[::-1]
# top4만을 출력
print(top[:4])
# pred(추론) 결과의 top4의 label class 및 label index 출력
print()
for r, index in zip(res[0], top[:4]):
    print('class name =', r[1], ', class index =', index)
```
위의 템플릿 코드를 읽고 분석합니다.  
model predict의 결과인 `image_probs[0]`에는 무엇이 들어있습니까?  
`ans12`에 기입합니다.  
`np.argsort(image_probs[0])[::-1]`는 무엇을 얻기 위한 것입니까?  
`ans13`에 기입합니다.  
top1 결과는 당연히 Labrado retriever입니다.
top2 결과인 `eskimo_dog`의 class index는 얼마입니까?  
(템플릿 코드를 실행해보시면 알 수 있습니다.)  
`ans14`에 그 index값을 기록합니다.  
원본이미지를 인식하였을 때, 4번째로 높은 추론값을 가진 결과(top4)는 무엇이었습니까?   
그 index 값을 `ans15`에 기록합니다.  
(참고로 강의에서 소개하였던 2010년부터의 imagenet competition은 top5내에 정답이 있으면, 맞는 것으로 하였었습니다.)  
    
**4. 이제 top2의 index를 이용하여, 원본 이미지에 대한 gradient를 구하고 그 부호값만을 취하여 target_perturabtion으로 정의하겠습니다.**  

다음의 code snippet을 템플릿의 맨 뒤에 추가합니다.    
**top2결과가 top1으로 나오게 하는 공격 snippet**  
```python
#top2 용 label 생성
target1_index = 248 
target1_label = tf.one_hot(target1_index, image_probs.shape[-1])
target1_label = tf.reshape(target1_label, (1, image_probs.shape[-1]))
# top2용 공격 패턴 생성
target_perturbations = create_adversarial_pattern(image, target1_label)
adv_x = image - eps*target_perturbations
adv_x = tf.clip_by_value(adv_x, 0, 1)  
# 공격 패턴이 적용된 이미지 adv_x에 대한 추론 및 결과 디스플레이
label1, confidence1 = display_images(adv_x, descriptions[1])
# FSGM 결과 
print(label1, confidence1)
```

`adv_x`를 계산할 때 `image - eps*target_perturbations`에서 보듯이 `-`(minus)가 사용되었습니다.  
`confidence1`의 결과를 `ans16`에 저장합니다. (템플릿에 이미 추가되어 있습니다.)    
    
**5. 위의 <공격 snippet>을 수정하여 top3의 index를 이용한 공격코드 및 결과코드를 추가합니다.**   

**공격 snippet**에서 `target2_index`, `targe2_label`등으로 변경하여 추가하시면 됩니다.  
label과 confidence값도 `label2`, `confidence2` 등으로 변경합니다  
`target2_index`의 값은 원본의 인식결과로 부터 출력된 top2의 index값으로 변경합니다.  
(snippet의 여러부분을 수정해야 하므로 실수하지 않도록 꼼꼼하게 확인합시다.)

**6. 위의 <공격 snippet>을 수정하여 top4의 index를 이용한 공격코드 및 결과코드를 추가합니다.**   

**공격 snippet**에서 `target3_index`, `targe3_label`등으로 변경하여 추가하시면 됩니다.  
label과 confidence값도 `label3`, `confidence3` 등으로 변경합니다  
`target3_index`의 값은 원본의 인식결과로 부터 출력된 top3의 index값으로 변경합니다.  
(snippet의 여러부분을 수정해야하므로 실수하지 않도록 꼼꼼하게 확인합시다.)

top3의 label을 이용한 결과인 `confidence2`를 `ans17`에, top4의 label을 이용한 `confidence3`를 `ans18`에 저장합니다.  
(이미 템플릿에 포함되어 있읍니다.)  

수정된 코드가 정상적으로 수행되면 총 다섯 개의 이미지와 결과가 출력될 것입니다.
맨처음 2개는 원본의 인식결과와, 템플릿에 있는 공격결과입니다.
다음 3개는 top2, top3, top4의 결과가 나오도록 하는 공격입니다.

원하는 공격 결과가 도출되었습니까?  

**과제 기입란**



```python
# 여기에 템플릿을 복사하고 수정하여 코드를 완성합니다. 
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

pretrained_model = tf.keras.applications.MobileNetV2(include_top=True,
                                                 weights='imagenet')
pretrained_model.trainable = False

# ImageNet 클래스 레이블
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

# 이미지가 MobileNetV2에 사용하기 위한 level normalization 및 size normalization 함수
def preprocess(image):
  image = tf.cast(image, tf.float32)
  image = image/255
  image = tf.image.resize(image, (224, 224))
  image = image[None, ...]
  return image

# 예측 결과로부터 top1을 뽑아서 라벨을 출력하는 함수
def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]

# Labrador Retreiver 이미지 불러오기
image_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
image_raw = tf.io.read_file(image_path)
image = tf.image.decode_image(image_raw)

image = preprocess(image)
# 추론 
image_probs = pretrained_model.predict(image)

loss_object = tf.keras.losses.CategoricalCrossentropy()

def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)

  # 입력 이미지에 대한 손실 함수의 기울기를 구합니다.
  gradient = tape.gradient(loss, input_image)
  # 왜곡을 생성하기 위해 그래디언트의 부호를 구합니다.
  signed_grad = tf.sign(gradient)
  return signed_grad

# 인식하고 이미지 출력하는 함수 (label과 confidence를 반환하도록 변경)
def display_images(image, description):
  _, label, confidence = get_imagenet_label(pretrained_model.predict(image))
  plt.figure()
  plt.imshow(image[0])
  plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
                                                   label, confidence*100))
  plt.show()
  return label, confidence

# 이미지의 레이블 위치만 1로 설정 (기본적인 클래스 분류기 결과)
labrador_retriever_index = 208
label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
label = tf.reshape(label, (1, image_probs.shape[-1]))

confidence1 = confidence2 = confidence3 = 0.

# 공격 패턴 생성
perturbations = create_adversarial_pattern(image, label)

epsilons = [0, 0.01, 0.1, 0.15]
# epsilon은 0.01로 고정
eps = epsilons[1]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]

#원본 인식 (epsilon = 0)
display_images(image, descriptions[0])

#공격 진행
adv_x = image + eps*perturbations
adv_x = tf.clip_by_value(adv_x, 0, 1)
label0, confidence0 = display_images(adv_x, descriptions[1]) # epsilon = 0.01

# FSGM 결과 
print(label0, confidence0)

res = decode_predictions(image_probs, top=4)
print(res[0])

# 추론결과(image_probs[0])를 역순으로 정렬
top = np.argsort(image_probs[0])[::-1]
# top4만을 출력
print(top[:4])

# 추론 결과의 top4의 label class 및 label index 출력
print()
for r, index in zip(res[0], top[:4]):
    print('class name =', r[1], ', class index =', index)

# model predict의 결과인 `image_probs[0]`에는 무엇이 들어있습니까? 
ans12 = '''
이미지넷으로 이미지를 보고 예측한 라벨 데이터가 있습니다.
'''

# np.argsort(image_probs[0])[::-1]는 무엇을 얻기 위한 것입니까?
ans13 = '''
예측한 라벨 데이터를 내림차순으로 정렬하여 추론 결과가 높은 라벨 데이터부터 출력하기 위함입니다.
'''

# top2 결과인 `eskimo_dog`의 class index는 얼마입니까?
ans14 = 248 # 값을 변경해 주세요.

# 원본이미지를 인식하였을 때, 4번째로 높은 추론값을 가진 결과(top4)는 무엇이었습니까?
# 인덱스 값으로 적어 주세요
ans15 = 250 # 값을 변경해 주세요.

    
# 이곳에 top2 index를 이용한 공격 code snippte을 추가합니다.
# top2 용 label 생성
target1_index = 248 
target1_label = tf.one_hot(target1_index, image_probs.shape[-1])
target1_label = tf.reshape(target1_label, (1, image_probs.shape[-1]))
# top2용 공격 패턴 생성
target_perturbations = create_adversarial_pattern(image, target1_label)
adv_x = image - eps*target_perturbations
adv_x = tf.clip_by_value(adv_x, 0, 1)  
# 공격 패턴이 적용된 이미지 adv_x에 대한 추론 및 결과 디스플레이
label1, confidence1 = display_images(adv_x, descriptions[1])
# FSGM 결과 
print(label1, confidence1)

# 이곳에 top3 index를 이용한 공격 code snippte을 수정하여 추가합니다.
# top3 용 label 생성
target2_index = 207 
target2_label = tf.one_hot(target2_index, image_probs.shape[-1])
target2_label = tf.reshape(target2_label, (1, image_probs.shape[-1]))
# top3용 공격 패턴 생성
target_perturbations = create_adversarial_pattern(image, target2_label)
adv_x = image - eps*target_perturbations
adv_x = tf.clip_by_value(adv_x, 0, 1)  
# 공격 패턴이 적용된 이미지 adv_x에 대한 추론 및 결과 디스플레이
label2, confidence2 = display_images(adv_x, descriptions[1])
# FSGM 결과 
print(label2, confidence2)

# 이곳에 top4 index를 이용한 공격 code snippte을 수정하여 추가합니다.
# top4 용 label 생성
target3_index = 250 
target3_label = tf.one_hot(target3_index, image_probs.shape[-1])
target3_label = tf.reshape(target3_label, (1, image_probs.shape[-1]))
# top3용 공격 패턴 생성
target_perturbations = create_adversarial_pattern(image, target3_label)
adv_x = image - eps*target_perturbations
adv_x = tf.clip_by_value(adv_x, 0, 1)  
# 공격 패턴이 적용된 이미지 adv_x에 대한 추론 및 결과 디스플레이
label3, confidence3 = display_images(adv_x, descriptions[1])
# FSGM 결과 
print(label3, confidence3)


# 공격 결과 중 confidence score를 다음 변수에 기록합니다.
ans16 = confidence1 # top2 index를 이용한 공격결과의 confidence score
ans17 = confidence2 # top3 index를 이용한 공격결과의 confidence score
ans18 = confidence3 # top4 index를 이용한 공격결과의 confidence score

# 본 과목을 수강하시면서 느낀점, 의견 등이 있으시면 적어 주세요. (공란으로 두셔도 됩니다.)
ans19 = """
여기에 적어주세요.
"""
```

# 과제 제출 방법

**다음은 답안의 형식을 확인하는 코드입니다. 실행해서 오류가 없는 지 확인합시다.**


```python
# 답안의 형식을 점검합니다.

from tensorflow import keras
import numpy as np

error = False

try:
    if type(ans01) != str:
        raise
except:
    error = True
    print('ans01 error')
    
try:
    if type(ans02) != str:
        raise
except:
    error = True
    print('ans02 error')
    
try:
    if ans03.shape != (5000, 2):
        raise
except:
    error = True
    print('ans03 error')
    
try:
    if type(ans04) != str:
        raise
except:
    error = True
    print('ans04 error')
    
try:
    if type(ans05) != str:
        raise
except:
    error = True
    print('ans05 error')
    
try:
    if type(ans06) != str:
        raise
except:
    error = True
    print('ans06 error')
    
try:
    if type(ans07) != str:
        raise
except:
    error = True
    print('ans07 error')
    
try:
    if type(ans08) != str:
        raise
except:
    error = True
    print('ans08 error')
    
try:
    if type(ans09) != str:
        raise
except:
    error = True
    print('ans09 error')
    
try:
    if ans10.shape != (200,):
        raise
except:
    error = True
    print('ans10 error')
    
try:
    if ans11.shape != (200,):
        raise
except:
    error = True
    print('ans11 error')

try:
    if type(ans12) != str:
        raise
except:
    error = True
    print('ans12 error')

try:
    if type(ans13) != str:
        raise
except:
    error = True
    print('ans13 error')

try:
    if not isinstance(ans14, int):
        raise
except:
    error = True
    print('ans14 error')

try:
    if not isinstance(ans15, int):
        raise
except:
    error = True
    print('ans15 error')

try:
    if not isinstance(ans16, float) and not isinstance(ans16, np.float32):
        raise
except:
    error = True
    print('ans16 error')

try:
    if not isinstance(ans17, float) and not isinstance(ans17, np.float32):
        raise
except:
    error = True
    print('ans17 error')

try:
    if not isinstance(ans18, float) and not isinstance(ans18, np.float32):
        raise
except:
    error = True
    print('ans18 error')

try:
    if not isinstance(ans19, str):
        raise
except:
    error = True
    print('ans19 error')

if error:
    print('답안을 확인하여 주세요')
else:
    print('답안의 형식 확인이 완료되었습니다.')
```

**과제 제출 방법**

1. **런타임** -> **다시 시작 및 모두 실행**을 수행하여 정상적으로 결과가 출력되는 지 다시 한번 확인합니다.  

2. **수정** -> **모든 출력 지우기**를 선택하여 cell의 출력을 지웁니다.

3. **파일** -> **`.ipynb`** 다운로드를 선택하여 노트북을 다운로드 합니다.

4. 파일 이름을 학번으로 변경합니다. 예) `202099999.ipynb`

5. 노트북 파일을 제출하시면 됩니다.


```python

```
