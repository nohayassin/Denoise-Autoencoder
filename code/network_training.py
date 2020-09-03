import os, sys, shutil
import keras
import time
import tensorflow as tf
from keras import backend as kb
from networks import Network

class NetworkTraining:
    def __init__(self, train_config, image_config):
        self.train_config = train_config
        self.image_config = image_config
        self.network = Network(train_config)

    def load_to_arrays(self, iteration = 0, images_num_to_process = 1):
        pure_input_train = self.image_config.image_to_array(iteration, images_num_to_process, self.train_config.get_image_to_array_train_input("pure"))
        noisy_input_train = self.image_config.image_to_array(iteration, images_num_to_process, self.train_config.get_image_to_array_train_input("noisy"))
        return pure_input_train, noisy_input_train

    def train(self):
        # Get the file paths
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
        # save output to logs
        old_stdout = sys.stdout
        timestr = time.strftime("%Y%m%d-%H%M%S")
        model_name = 'DEPTH_' + timestr + '.model'
        name = self.train_config.logs_path + r'/loss_output_' + model_name + '.log'
        log_file = open(name, "w")
        sys.stdout = log_file
        print('Loss function output of model :', model_name, '..')

        # Create a basic model instance
        model = self.network.get()
        compiled_model = model.compile()

        save_model_name = self.train_config.models_path +'/' + model_name
        images_num_to_process = 1000
        all_cropped_num = len(os.listdir(self.train_config.cropped_train_images_pure))
        iterations = all_cropped_num // images_num_to_process
        for i in range(iterations):
            print('*************** Iteration : ', i, '****************')
            if self.train_config.LOAD_TRAINED_MODEL:
                # create a dir where we want to copy and rename
                save_model_name = self.train_config.load_model_name + '_new'
                if not os.path.isdir(save_model_name):
                    shutil.copytree(self.train_config.load_model_name, save_model_name)
                compiled_model = keras.models.load_model(save_model_name) # used to continue training old models

            pure_input_train, noisy_input_train = self.load_to_arrays(i*images_num_to_process, images_num_to_process)
            if self.train_config.OUTPUT_EQUALS_INPUT:
                pure_input_train = noisy_input_train

            model.train(compiled_model, noisy_input_train, pure_input_train,self.train_config.models_path)

            # save the model
            compiled_model.save(save_model_name) # check if using same name is ok
            compiled_model = keras.models.load_model(save_model_name)
            #model = keras.models.load_model(save_model_name, custom_objects={'keras_custom_loss_function':keras_custom_loss_function})

        sys.stdout = old_stdout
        log_file.close()