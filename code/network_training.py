import os, sys, shutil
import time
from packaging import version
import tensorflow as tf
from tensorflow import keras
from keras import backend as kb
from datetime import datetime
from networks import Network

class NetworkTraining:
    def __init__(self, train_config, image_config):
        self.train_config = train_config
        self.image_config = image_config
        self.network = Network(train_config)

    def train(self):
        print("TensorFlow version: ", tf.__version__)
        assert version.parse(tf.__version__).release[0] >= 2, \
            "This training requires TensorFlow 2.0 or above."
        kb.clear_session()
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        print('Starting a training process ..')
        print('Preparing training data for CNN ..')

        time_path = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_config.models_path += r"/"+time_path
        self.train_config.logs_path += r"/" + time_path
        self.train_config.paths = [self.train_config.models_path, self.train_config.logs_path]
        self.train_config.create_folders()
        # save output to logs
        old_stdout = sys.stdout
        #timestr = time.strftime("%Y%m%d-%H%M%S")
        model_name = 'DEPTH_' + time_path + '.model'
        name = self.train_config.logs_path + r'/training_output_' + model_name + '.log'
        log_file = open(name, "w")
        sys.stdout = log_file
        print('Loss function output of model :', model_name, '..')

        # Create a basic model instance
        model = self.network.get()
        compiled_model = model.compile()

        save_model_name = self.train_config.models_path +'/' + model_name
        images_num_to_process = 1000
        all_cropped_num = len(
            os.listdir(self.train_config.train_cropped_images_path)) // 3  # this folder contains cropped images of pure, noisy and ir
        iterations = all_cropped_num // images_num_to_process
        if all_cropped_num % images_num_to_process > 0:
            iterations += 1
        for i in range(iterations):
            print('*************** Iteration : ', i, '****************')
            first_image = i * images_num_to_process
            if i == iterations - 1:
                images_num_to_process = all_cropped_num - i * images_num_to_process
            if self.train_config.LOAD_TRAINED_MODEL:
                # create a dir where we want to copy and rename
                save_model_name = self.train_config.load_model_name + '_new'
                if not os.path.isdir(save_model_name):
                    shutil.copytree(self.train_config.load_model_name, save_model_name)
                compiled_model = keras.models.load_model(save_model_name) # used to continue training old models

            noisy_input_train, pure_input_train  = self.image_config.image_to_array(first_image, images_num_to_process, self.train_config.train_cropped_images_path)
            #self.load_to_arrays(first_image, images_num_to_process)
            if self.train_config.OUTPUT_EQUALS_INPUT:
                pure_input_train = noisy_input_train

            model.train(compiled_model, noisy_input_train, pure_input_train,self.train_config.models_path)

            # save the model
            compiled_model.save(save_model_name) # check if using same name is ok
            compiled_model = keras.models.load_model(save_model_name)
            tf.io.write_graph(
                compiled_model.output.graph,
                self.train_config.models_path,
                'model.pbtxt',
                as_text=True,
            )
        sys.stdout = old_stdout
        log_file.close()