import os
import tensorflow as tf
import keras
import cv2
import logging
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.keras import backend as K
from tensorflow import keras

# necessary !!!
tf.compat.v1.disable_eager_execution()

h5_path = r"C:\Users\user\Documents\new_ML\pbtxt\unet_membrane.hdf5"
model = keras.models.load_model(h5_path)
model.summary()
# save pb
with K.get_session() as sess:
    output_names = [out.op.name for out in model.outputs]
    input_graph_def = sess.graph.as_graph_def()
    for node in input_graph_def.node:
        node.device = ""
    graph = graph_util.remove_training_nodes(input_graph_def)
    graph_frozen = graph_util.convert_variables_to_constants(sess, graph, output_names)
    tf.io.write_graph(graph_frozen, r'C:\Users\user\Documents\new_ML\pbtxt', name='model.pb', as_text=False)
    tf.io.write_graph(graph_frozen, r'C:\Users\user\Documents\new_ML\pbtxt', name='model.pbtxt', as_text=True)
logging.info("save pb successfully！")



cv2.dnn.readNetFromTensorflow(r"C:\Users\user\Documents\new_ML\pbtxt\model.pb",
                              r"C:\Users\user\Documents\new_ML\pbtxt\model.pbtxt")