from keras.applications.resnet_v2 import preprocess_input, decode_predictions
import numpy as np
import random
import tensorflow as tf
import time
from sklearn import metrics
print('--------------------------------------------------')
import os
import cv2
import numpy as np
import keras
from keras.optimizers import SGD
from keras.models import Model
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.mobilenet_v2 import mobilenet_v2
from keras.applications.vgg16 import vgg16
from keras.applications.resnet_v2 import preprocess_input
from keras.preprocessing import image
import scipy.io as sio
import pickle
import time
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from sklearn.metrics import roc_auc_score

def get_default_params(data_path, file_path):
    """
    description: parameters dictionary initialization
    return: dictionary of parameters
    """
    params = {'data_path': data_path,
              'file_path': file_path,
              'width': 224,
              'height': 224,
              'optimization_method': SGD,
              'decay': 0.1,
              'momentum': 0.9,
              'epochs': 12,
              'batch_size': 25
              }
    file_path = os.path.join(params["file_path"], "FlowerDataLabels")
    LabelsMat = sio.loadmat(file_path, mdict=None, appendmat=True)
    params['classification'] = np.transpose(LabelsMat['Labels']).tolist()
    return params


def load_data(params, train_Images_indices, test_Images_indices, data_aug=0):
    '''
    description: Performs preprocessing as well as division for training and validating and testing
    param params:  dict of all the pipeline parameters
    param train_Images_indices: set of train image indices
    param test_Images_indices: set of test image indices
    param data_aug: if to apply augmentation or not
    return train_data: train dictionary split to train and validation each split to processed images and labels
    return test_data: test dictionary of processed images with labels
    '''

    Images_train = len(train_Images_indices) - 50
    validation_train = 50
    print(Images_train, validation_train)
    test_Images_list = []
    train_Images_list = []
    validation_Images_list = []
    train_labels = []
    test_labels = []
    validation_labels = []
    counter = 0
    train_data = {'training': {'Images': []}, 'Validation': {'Images': []},'augmentedData':{'Images': [],'Labels': []}}
    test_data = {'Images': [], 'Labels': []}

    for i in train_Images_indices:
        image_dir_file = params['data_path'] + "/" + str(i) + ".jpeg"
        pic = image.load_img(image_dir_file, target_size=(params['width'], params['height']))  # load image
        pic = image.img_to_array(pic)
        pic = np.expand_dims(pic, axis=0)

        pic = preprocess_input(pic)
        if counter < Images_train:
            train_Images_list.append(pic)
            train_labels.append(params['classification'][i - 1][0])
        if counter >= Images_train and counter < (Images_train + validation_train):
            validation_Images_list.append(pic)
            validation_labels.append(
                params['classification'][i - 1][0])
        counter = counter + 1

    for i in test_Images_indices:
        image_dir_file = params['data_path'] + "/" + str(i) + ".jpeg"
        pic = image.load_img(image_dir_file, target_size=(params['width'], params['height']))  # image read
        pic = image.img_to_array(pic)
        pic = np.expand_dims(pic, axis=0)
        pic = preprocess_input(pic)
        test_Images_list.append(pic)
        test_labels.append(params['classification'][i - 1][0])

    train_Images_array = np.array(train_Images_list)
    train_Images_array = np.rollaxis(train_Images_array, 1, 0)
    train_data['training']['Images'] = train_Images_array[0]

    test_Images_array = np.array(test_Images_list)
    test_Images_array = np.rollaxis(test_Images_array, 1, 0)
    test_data['Images'] = test_Images_array[0]

    validation_Images_array = np.array(validation_Images_list)
    validation_Images_array = np.rollaxis(validation_Images_array, 1, 0)
    train_data['Validation']['Images'] = validation_Images_array[0]

    train_data['training']['Labels'] = np.array(train_labels)
    test_data['Labels'] = np.array(test_labels)
    train_data['Validation']['Labels'] = np.array(validation_labels)

    if data_aug==1:
        train_data = data_augmentation(params, train_Images_indices, train_data)


    # with open('pickleForReal.pickle', 'wb') as handle:
    #     pickle.dump((train_data, test_data), handle, protocol=pickle.HIGHEST_PROTOCOL)

    return train_data, test_data


def get_random_crop(image, params):
    '''
    description: Apply random crop  to a given image
    param image:  an image array
    param params:  dict of all the pipeline parameters
    return crop_image: random cropped image
    '''
    x1 = np.random.randint(params['width'] * 0.9, params['width'])
    y1 = np.random.randint(params['width'] * 0.9, params['height'])
    x2 = image.shape[1] - x1
    y2 = image.shape[0] - y1
    crop_image = image[x2:x1, y2:y1]
    crop_image = cv2.resize(crop_image, (params['width'], params['height']))

    return crop_image


def data_augmentation(params, train_Images_indices, train_data):
    '''
    description: Augment the training dataset by taking every image and apply 3 transformations: horizontal flip, clockwise rotation
    and random crop/
    param params:  dict of all the pipeline parameters
    param train_Images_indices:  set of train image indices
    return train_data: augmented train dataset
    '''

    augmented_images_list = []
    augmented_image_labels_list = []
    num_images = 472
    for i in range(1, len(train_Images_indices)):  # for filename in data_path folder which is flowerData
        image_dir_file = params['data_path'] + "/" + str(i) + ".jpeg"
        img = image.load_img(image_dir_file, target_size=(params['width'], params['height']))  # load image
        img = image.img_to_array(img)
        horizontally_flipped_image = cv2.flip(img, 1)
        horizontally_flipped_image = np.expand_dims(horizontally_flipped_image, axis=0)
        horizontally_flipped_image = preprocess_input(horizontally_flipped_image)


        clockwise_image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        clockwise_image = np.expand_dims(clockwise_image, axis=0)
        clockwise_image = preprocess_input(clockwise_image)

        anticlockwise_image = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        anticlockwise_image = np.expand_dims(anticlockwise_image, axis=0)
        anticlockwise_image = preprocess_input(anticlockwise_image)

        augmented_images_list.append(horizontally_flipped_image)
        augmented_images_list.append(clockwise_image)
        augmented_images_list.append(anticlockwise_image)

        augmented_image_labels_list += [params['classification'][i - 1][0]] * 3  #[train_labels[i]] * 3

    augmented_Images_array = np.array(augmented_images_list)
    augmented_Images_array = np.rollaxis(augmented_Images_array, 1, 0)

    train_data['augmentedData']['Images'] = augmented_Images_array[0]
    train_data['augmentedData']['Labels'] = np.array(augmented_image_labels_list)

    print(train_data['augmentedData']['Images'].shape)

    return train_data


def create_model(params, model='ResNet50V2', mode='advanced'):
    '''
    description: adjusting the resnet network to our mission
    param params: dictionary of all the pipeline parameters
    param model: number of layers to train
    param mode: train on base mode or advanced (means model with all the advances - decay, max pooling etc)
    return myNet: a new NCC with last layer as a 2 class classifier
    '''

    if mode =='advanced':
        if model=='ResNet50V2':
                imageinput = keras.layers.Input(shape=(224, 224, 3))
                model = keras.applications.resnet_v2.ResNet50V2(include_top=True, weights='imagenet', input_tensor=imageinput,
                                                                input_shape=None, pooling='max',
                                                                classes=1000)  # get the res net as required

        if model=='mobilenet_v2':
            imageinput = keras.layers.Input(shape=(224, 224, 3))
            model = keras.applications.mobilenet_v2.MobileNetV2(include_top=True, weights='imagenet', input_tensor=imageinput,
                                                            input_shape=None, pooling='max',
                                                            classes=1000)  # get the res net as required

        if model=='vgg16':
            imageinput = keras.layers.Input(shape=(224, 224, 3))
            model = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=imageinput,
                                                            input_shape=None, pooling='max',
                                                            classes=1000)

        model.layers.pop()
        last_layer = keras.layers.Dense(1, activation='sigmoid', name='output_layer')
        new_layer = last_layer(model.layers[-1].output)
        myNet = Model(inputs=model.input, outputs=new_layer)


        myNet.compile(loss='binary_crossentropy', optimizer=SGD(decay=params['decay'], nesterov=True, momentum=0.9),
                      metrics=['accuracy'])
        return (myNet)

    elif mode=='base':
        imageinput = keras.layers.Input(shape=(224, 224, 3))
        model = keras.applications.resnet_v2.ResNet50V2(include_top=True, weights='imagenet', input_tensor=imageinput,
                                                        input_shape=None, pooling='avg',
                                                        classes=1000)


        model.layers.pop()
        last_layer = keras.layers.Dense(1, activation='sigmoid', name='output_layer')
        new_layer = last_layer(model.layers[-1].output)
        myNet = Model(inputs=model.input, outputs=new_layer)
        myNet.compile(loss='binary_crossentropy', optimizer=SGD(),
                      metrics=['accuracy'])
        return (myNet)

def train_model(params, model, trainDic, mode='train_all_data_with_augmentation',with_early_stopping=True): # mode can be train with val or only train means on all the training data
    '''
    description: training the model
    param params: dictionary with pipeline  parameters
    param model: model architecture
    parm trainDic: dictionary of train data, images and labels
    param mode:'train_all_data_with_augmentation'/'train_all_data_with_augmentation'/'train_all_data_without_augmentation'
    return history: object containing the training data - train/val accuracy
    return model: trained model
    '''

    print("start training")

    if with_early_stopping:
        condition = keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.001, patience=1)
        if mode == 'train_with_validation':
            labVal = trainDic['Validation']['Labels']
            labVal = np.array(labVal)
            H = keras.callbacks.History()
            condition = keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.001, patience=1)
            merged_images = np.concatenate((trainDic['augmentedData']['Images'], trainDic['training']['Images']))
            merged_labels = np.concatenate((trainDic['augmentedData']['Labels'], trainDic['training']['Labels']))
            history = model.fit(merged_images, merged_labels, batch_size=params['batch_size'],
                                epochs=params['epochs'], verbose=1,
                                validation_data=(trainDic['Validation']['Images'], labVal),
                                callbacks=[H,condition])
            return model, history

        elif mode == 'train_all_data_with_augmentation':
            merged_images = np.concatenate((trainDic['augmentedData']['Images'], trainDic['training']['Images'], trainDic['Validation']['Images']))
            merged_labels = np.concatenate((trainDic['augmentedData']['Labels'], trainDic['training']['Labels'], trainDic['Validation']['Labels']))
            model.fit(merged_images, merged_labels, epochs=params['epochs'],
                      batch_size=params['batch_size'], verbose=1,
                      callbacks=[condition])
            return model

        elif mode == 'train_all_data_without_augmentation':
            merged_images = np.concatenate((trainDic['training']['Images'], trainDic['Validation']['Images']))
            merged_labels = np.concatenate((trainDic['training']['Labels'], trainDic['Validation']['Labels']))
            model.fit(merged_images, merged_labels, epochs=params['epochs'],
                      batch_size=params['batch_size'], verbose=1,
                      callbacks=[condition])

            return model
    else:
        merged_images = np.concatenate((trainDic['training']['Images'], trainDic['Validation']['Images']))
        merged_labels = np.concatenate((trainDic['training']['Labels'], trainDic['Validation']['Labels']))
        model.fit(merged_images, merged_labels, epochs=params['epochs'],
                  batch_size=params['batch_size'], verbose=1)

        return model


def predict(model, test_images):
    """
    predict label on test images
    :param model: trained model
    :param test_images:
    :return: model predictions
    """
    predictions = model.predict(np.asarray(test_images))
    print(predictions)
    return predictions


def calculate_error(predictions, test_labels, test_images_indices):
    """
    :param predictions: model predictions
    :param test_labels: actual labels
    :param test_images_indices:
    :return errors: dict type 1 and 2 errors on indices
    """
    errors = {'first': {'index': None, 'error': None}, 'second': {'index': None, 'error': None}}
    images_indices_1, images_indices_2, error_1, error_2 = [], [], [], []
    test_images_indices = list(test_images_indices)

    for i in range(len(test_labels)):
        if test_labels[i] == 0 and predictions[i] > 0.5:
            images_indices_1.append(test_images_indices[i])
            error_1.append(predictions[i])
        if test_labels[i] == 1 and predictions[i] <= 0.5:
            images_indices_2.append(test_images_indices[i])
            error_2.append(1 - predictions[i])

    errors['first'] = {'index': images_indices_1, 'error': error_1}
    errors['second'] = {'index': images_indices_2, 'error': error_2}

    return errors


def find_top_5_errors(errors):
    """
    find top 5 errors of each type
    :param errors: error dict
    :return max_five_first_error: dict with top 5 first error indices and and values
    :return max_five_second_error: dict with top 5 second error indices and and values
    """
    max_five_first_error = {'index': None, 'error': None}
    max_five_second_error = {'index': None, 'error': None}
    for mistake_type in errors.keys():
        if errors[mistake_type]['index'] is not None:
            zipped_lists = zip(errors[mistake_type]['error'], errors[mistake_type]['index'])
            sorted_pairs = sorted(zipped_lists)
            sorted_pairs.reverse()
            if mistake_type == 'first':
                max_five_first_error['index'] = [np.ceil(i[1]) for i in sorted_pairs[:5]]
                max_five_first_error['error'] = [i[0] for i in sorted_pairs[:5]]
            else:
                max_five_second_error['index'] = [np.ceil(i[1]) for i in sorted_pairs[:5]]
                max_five_second_error['error'] = [i[0] for i in sorted_pairs[:5]]

    print('top 5 five type 1 error indices: ', max_five_first_error['index'])
    print('top 5 five type 1 error values: ',  [i[0] for i in max_five_first_error['error']])
    print('top 5 five type 2 error indices: ', max_five_second_error['index'])
    print('top 5 five type 2 error values: ', [i[0] for i in max_five_second_error['error']])
    return max_five_first_error, max_five_second_error




def test(model, test_images, test_labels,params):
    '''
    testing function evaluates accuracy of the model and the model loss score
    param model: trained model
    param test_images:
    param test_labels:
    param params: pipeline parameters
    return: model accuracy
    '''
    print("start testing")
    (loss, accuracy) = model.evaluate(np.asarray(test_images), np.asarray(test_labels), batch_size=params['batch_size'], verbose=1)
    print("test results")
    print(" The loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100), "error :" + str(1 - accuracy),
          " Decay: " + str(params['decay']),  "pooling: " + 'avg')
    print("batch size :" + str(params['batch_size']), "epochs :" + str(params['epochs']))

    return accuracy



def tune_parameters(params ,trainDic,testDic, param_dict={'batch_size' : [25,32,64],
                                                  'decay' : [0.05,0.1,0.5]}):
    """
    :param params: pipeline parameters
    :param trainDic: dictionary containing training and validation images and labels
    :param testDic: dictionary containing test images and labels
    :param param_dict: set of parameters to tune
    :return: validation scores
    """
    results_dict = {}
    for batch_size_ in param_dict['batch_size']:
        for decay_ in param_dict['decay']:
            start = time.time()
            experiment_name = 'batch_size_' + str(batch_size_) + '_decay_' + str(decay_)
            print('started_experiment: ', experiment_name)
            params['batch_size'] = batch_size_
            params['decay'] = decay_
            model = create_model(params, 'ResNet50V2')
            model, history = train_model(params, model, trainDic, mode='train_with_validation')
            print(history.history)
            mean_val_loss = np.mean(history.history['val_accuracy'])
            end = time.time()
            print('run time: ', end - start)

            test_accuracy = test(model, testDic['Images'], testDic['Labels'],params)
            results_dict.update({experiment_name:{'validation_mean_score':mean_val_loss, 'test_accuracy':test_accuracy}})

            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.savefig(experiment_name+'_accuracy_plot.png')

    with open('parameter_tuning_results.pickle', 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return results_dict


def plot_confusion_matrix(predictions, testDic):
    '''
    creates confusion matrix plot and saves the image
    param predictions: predicted model probabilities
    param testDic: dictionary containing actual labels
    return:
    '''
    predictions = [np.round(i[0]) for i in predictions]
    array = confusion_matrix(testDic['Labels'], predictions)
    df_cm = pd.DataFrame(array, range(2), range(2))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
    #plt.savefig('confusion_matrix_res.png')
    print(df_cm)


def plot_precision_recall(model, testDic):
    '''
    plot precision-recall curve
    param: dictionary of test data, images and labels
    param: tested model
    '''
    precision, recall, thresholds = precision_recall_curve(testDic['Labels'], model.predict(testDic['Images']))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.1])
    plt.xlim([0.0, 1.05])
    #plt.savefig('precision_recall_plot.png')
    plt.show()

print('--------------------------------------------------')


if __name__ == "__main__":
    # in our pipeline the labels file should be separated from the flowers images folder
    start = time.time()
    random.seed(10)
    np.random.seed(10)
    tf.random.set_seed(10)

    data_path = "/home/thinkpad/Desktop/Masters courses/Computer Vision/task 2/FlowerData"
    mat_path = "/home/thinkpad/Desktop/Masters courses/Computer Vision/task 2"
    test_Images_indices = set(range(301, 473))

    train_Images_indices = set([i for i in range(1, 473) if not i in test_Images_indices])
    train_Images_indices = train_Images_indices - test_Images_indices

    params = get_default_params(data_path, mat_path)

    print("loading data")
    trainDic, testDic = load_data(params, train_Images_indices, test_Images_indices, 1)
    print('num true flower images in original training exaples: ', np.array(np.unique(np.concatenate((trainDic['Validation']['Labels'] ,trainDic['training']['Labels'])), return_counts=True)).T[0][1])
    #print('num true flower images in augmented training exaples: ', np.array(np.unique(trainDic['augmentedData']['Labels'], return_counts=True)).T[0][1])

    print("define network")
    model = create_model(params) #- achieved 88.9% accuracy
    #model = Functions.create_model(params, 'ResNet50V2','base')

    print('model structure')
    print(model.summary())

    model = train_model(params, model, trainDic, mode='train_all_data_without_augmentation')
    #model = Functions.train_model(params, model, trainDic, mode='train_all_data_without_augmentation',with_early_stopping=False)

    predictions = predict(model, testDic['Images'])

    errors = calculate_error(predictions, testDic['Labels'], test_Images_indices)

    max_five_first_error, max_five_second_error = find_top_5_errors(errors)

    accuracy = test(model, testDic['Images'], testDic['Labels'],params)


    #   results_dict = Functions.tune_parameters(params, trainDic, testDic, param_dict={'batch_size': [500],
    #                                                                                'decay': [0.1]})
    print('precision recall plot')
    plot_precision_recall(model, testDic)

    print("Confusion matrix")
    plot_confusion_matrix(predictions,testDic)

    print('AUC score: ',roc_auc_score(testDic['Labels'], predictions))

    end = time.time()
    print('run time: ', (end - start)/60, ' minutes')
