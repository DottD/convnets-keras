# Purpose

This is a modified version of [DL-UNAL/convnets-keras](https://github.com/DL-UNAL/convnets-keras.git), to adapt it to the new version of Keras 2.0.

> Only AlexNet has been ported by now. Any contibutions will be appreciated.

# Authors

- Leonard Blier, original work (see [DL-UNAL/convnets-keras](https://github.com/DL-UNAL/convnets-keras.git))
- Filippo Santarelli, porting to Keras 2.0

# Installation

The only dependencies are h5py, Theano and Keras. Run the following commands
```
pip install --user cython h5py
pip install --user git+https://github.com/Theano/Theano.git
pip install --user git+https://github.com/fchollet/keras.git
```

Then, you need to install the convnetskeras module :
```
git clone https://github.com/DottD/convnets-keras.git
cd convnets-keras
sudo python setup.py install
```