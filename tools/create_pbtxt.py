import tensorflow as tf
import keras

model = r"C:\Users\user\Documents\ML\expirements_models_denoised\NEW DATA\4443 images\unet model - 100 epochs - strides 1500 - NO mask - Binary - WITH IR - NOT FINISHED\DEPTH_20200903-132536.model_new"
keras_model = keras.models.load_model(model)
tf.io.write_graph(
  keras_model.output.graph,
  r'C:\Users\user\Documents\ML\expirements_models_denoised\NEW DATA\4443 images\unet model - 100 epochs - strides 1500 - NO mask - Binary - WITH IR - NOT FINISHED',
  'model.pbtxt',
  as_text=True,
)