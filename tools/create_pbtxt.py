import os
import tensorflow as tf
import keras
import cv2
import logging
import tensorflow as tf
#from tensorflow.python.framework import graph_util
from tensorflow.python.keras import backend as K
from tensorflow import keras
from tensorflow.python.tools.freeze_graph import freeze_graph

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

def create_pbtxt():
    # necessary !!!
    tf.compat.v1.disable_eager_execution()

    h5_dir = r"C:\Users\user\Documents\new_ML\pbtxt\4"
    h5_path = h5_dir+ r"\unet_membrane.hdf5"
    model = keras.models.load_model(h5_path)
    # save pb
    with K.get_session() as sess:
        output_names = [out.op.name for out in model.outputs]
        input_graph_def = sess.graph.as_graph_def()
        for node in input_graph_def.node:
            node.device = ""
        #graph = graph_util.remove_training_nodes(input_graph_def)
        graph = tf.compat.v1.graph_util.remove_training_nodes(input_graph_def)
        #graph_frozen = graph_util.convert_variables_to_constants(sess, graph, output_names)
        graph_frozen = tf.compat.v1.graph_util.convert_variables_to_constants(sess, graph, output_names)
        tf.io.write_graph(graph_frozen, h5_dir, name='model.pb', as_text=False)
        tf.io.write_graph(graph_frozen, h5_dir, name='model.pbtxt', as_text=True)
    logging.info("save pb successfully！")


#create_pbtxt()
h5_dir = r"C:\work\ML_git\pb_pbtxt\1"
model = keras.models.load_model(h5_dir+ r"\unet_membrane.hdf5")
model = keras.models.convert_model(model)

frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
#tf.train.write_graph(frozen_graph, h5_dir, "model.pb", as_text=False)
#tf.train.write_graph(frozen_graph, h5_dir, "model.pbtxt", as_text=True)

#tf.compat.v1.graph_util.extract_sub_graph(h5_dir+r'\model.pb', h5_dir+r'\model.pbtxt')

cv2.dnn.readNetFromTensorflow(r"C:\Users\user\Documents\new_ML\pbtxt\2\model.pb",
                              r"C:\Users\user\Documents\new_ML\pbtxt\2\model.pbtxt")