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
    tf.compat.v1.disable_eager_execution() # could be removed ?
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


model = keras.models.load_model(r"C:\work\ML_git\clean_env_autoencoder\tmp\DEPTH_20200903-132536.model_new")
print(model.output.op.name)
frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, r"C:\work\ML_git\clean_env_autoencoder\tmp", "my_model.pb", as_text=False)
tf.train.write_graph(frozen_graph, r"C:\work\ML_git\clean_env_autoencoder\tmp", "my_model.pbtxt", as_text=True)

cv2.dnn.readNetFromTensorflow(r"C:\work\ML_git\clean_env_autoencoder\tmp\my_model.pb",
                              r"C:\work\ML_git\clean_env_autoencoder\tmp\my_model.pbtxt")
