import os
import numpy as np
#import pickle
import cv2
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix


def getDefaultParam(class_indices,data_path):
    """"
    documentation:

    input class_indices: list of class indices

    output params: dictionary of experiment parameters
    """

    params = {}
    params['data_path'] = data_path
    params['class_indices'] = class_indices
    params['C'] = 1
    params['S'] = 100
    params['Degree'] = 3
    params['Kernel'] = 'poly'
    params['orientation'] = 7
    params['pixels_per_cell'] = (8, 8)
    params['cells_per_block'] = (4, 4)
    params['block_norm'] = 'L2'
    params['model_type'] = 'polynomial'# model type can be linear/polynomial
    return params

def load_data(params,data_path,tunning_mode=False):
    """"
    documentation: loads the images

    input params: dict with experiment parameters
    input data_path: path to the location of the dataset
    input tunning_mode(default is False): running mode (tuning or not)

    output image_collection: list of all the images loaded
    output image_locations: list of all the images paths

    """
    image_collection = []
    image_locations = {}
    for class_ in range(len(params['class_indices'])):
        image_locations[class_] = []

    map_dict = {}
    counter = 0
    for i in params['class_indices']:
        map_dict[i] = counter
        counter += 1

    classes = os.listdir(data_path)
    for i in params['class_indices']:
        class_path = os.path.join(data_path, classes[i])
        images = os.listdir(class_path)
        images = sorted(images)
        if len(images) >= 50:
            images = images[:50]
        class_images = []
        for image in images:
            image_locations[map_dict[i]].append(os.path.join(data_path, class_path, image))
            gray_img = cv2.imread(os.path.join(data_path, class_path, image), cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(gray_img, (params['S'], params['S']))
            class_images.append(resized_img)
        class_images = np.array(class_images)
        image_collection.append(class_images)

    if tunning_mode:
        return image_collection

    else:
        return image_collection, image_locations


def data_split(params,image_collection,image_locations=[],tunning_mode=False,add_augmentations=False):
    """"
    documentation: split data into train and test sets and saves image paths in a list

    input params: dict with experiment parameters
    input image_collection: list of all the images
    input image_locations: list of the image locations
    input tunning_mode(default is False): running mode (tuning or not)
    input add_augmentations(default is False): binary parameter if to use different data augmentation methods like rotation etc.

    output train_x: array of train images
    output train_y: array of train labels
    output test_x: array of test images
    output test_y: array of test labels
    output test_image_paths: list of the test image paths for the plotting at the end (images with the largest errors in each class)
    """
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    augmentations_x_train = []
    augmentations_y_train = []
    counter_train = 0
    counter_test = 0

    classes = []
    for ind in params['class_indices']:
        classes.append(os.listdir(params['data_path'])[ind])

    for i in range(len(image_collection)):
        print('class: ', classes[i])
        if image_collection[i].shape[0] == 50:
            print('train size: ', 25)
            print('test size: ', 25)
            train_x.append(image_collection[i][:25])
            test_x.append(image_collection[i][25:])
            if not tunning_mode:
                image_locations[i] = image_locations[i][25:]
            train_y.append([params['class_indices'][i]]*25)
            test_y.append([params['class_indices'][i]]*25)
            counter_train += 25
            counter_test += 25
        else:
            train_size = int(np.round(image_collection[i].shape[0]/2))
            print('train size: ', train_size)
            print('test size: ', image_collection[i].shape[0] - train_size)
            train_x.append(image_collection[i][:train_size])
            test_x.append(image_collection[i][train_size:])
            if not tunning_mode:
                image_locations[i] = image_locations[i][train_size:]
            train_y.append([params['class_indices'][i]]*train_size)
            test_y.append([params['class_indices'][i]]*(image_collection[i].shape[0] - train_size))
            counter_train += train_size
            counter_test += (image_collection[i].shape[0] - train_size)




    if not tunning_mode:
        test_image_paths = []
        for class_paths in image_locations.values():
            test_image_paths += class_paths

    # new part
    if add_augmentations:
        for i in range(len(train_x)):
            # rotation
            image = train_x[i]
            height, width = image.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 45, .5)
            rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
            augmentations_x_train.append(rotated_image)

            # Increase Brightness
            bright = np.ones(image.shape, dtype="uint8") * 70
            brightincrease = cv2.add(image, bright)
            augmentations_x_train.append(brightincrease)

            # Decrease Brightness
            brightdecrease = cv2.subtract(image, bright)
            augmentations_x_train.append(brightdecrease)
            

            # Fliping the Image
            flip = cv2.flip(image, 3)
            augmentations_x_train.append(flip)

            # Sharpening
            sharpening = np.array([[-1, -1, -1],
                                   [-1, 10, -1],
                                   [-1, -1, -1]])
            sharpened = cv2.filter2D(image, -1, sharpening)
            augmentations_x_train.append(sharpened)


            augmentations_y_train += [train_y[i]] * 5

        train_x += augmentations_x_train
        train_y += augmentations_y_train

    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)

    if not tunning_mode:
        return train_x, train_y, test_x, test_y, test_image_paths

    else:
        return train_x, train_y, test_x, test_y



def apply_HOG(params,train_x, test_x):
    """"
    documentation: gets HOG parameters and extracts HOG features for train and test images

    input params: dict with experiment parameters
    input train_x: array of train images
    input test_x: array of test images

    output train_images: list of extracted HOG features for the train examples
    output test_images: list of extracted HOG features for the test examples
    """

    train_images = []
    test_images = []
    for image in train_x:
        feat,HOG = hog(image, orientations=params['orientation'], pixels_per_cell=params['pixels_per_cell'],
                        cells_per_block=params['cells_per_block'], block_norm=params['block_norm'], visualize=True)
        train_images.append(feat)

    for image in test_x:
        feat,HOG = hog(image, orientations=params['orientation'], pixels_per_cell=params['pixels_per_cell'],
                        cells_per_block=params['cells_per_block'], block_norm=params['block_norm'], visualize=True)
        test_images.append(feat)

    return train_images, test_images



def m_classes_SVM_train(params,train_images,train_y):
    """"
    documentation: gets a dict of parameters , train images the labels and fit a model

    input params: dict with experiment parameters
    input train_images: list of train images
    input train_y: list of train labels

    output model: trained model
    """
    map_dict = {}
    counter = 0

    for i in set(train_y.tolist()):
        counter += 1
        map_dict[i] = counter

    y = []
    for i in train_y:
        y.append(map_dict[i])

    if params['model_type'] == 'linear':
        model = svm.LinearSVC(random_state=0,C=params['C']).fit(train_images, y)
        return model

    if params['model_type'] == 'polynomial':
        model = svm.SVC(kernel=params['Kernel'], degree=params['Degree'], C=params['C'], decision_function_shape='ovr', probability=True).fit(train_images,y)
        return model




def m_classes_SVM_predict(params,test_images,model):
    """"
    documentation: gets a model and training examples and predict the class

    input params: dict with parameters
    input test_images: list with test images
    input model: trained model to predict with

    output predictions: the predictions of the model
    output score_matrix: In a fitted classifier or outlier detector, predicts a “soft” score for each sample in relation to each class, rather than the “hard” categorical prediction produced by predict.
    Its input is usually only some observed data, X.
    """

    map_dict = {}
    for i in range(1,len(params['class_indices'])+1):
        map_dict[i] = params['class_indices'][i-1]


    predictions = model.predict(np.array(test_images))
    predictions = np.vectorize(map_dict.get)(predictions)
    score_matrix = model.decision_function(test_images)

    return predictions, score_matrix




def _create_grid(skf=StratifiedKFold(n_splits=10,random_state=1291295159),
                clf=svm.SVC(),
                param_grid={}):
    """
    Pipeline for the training phase

    parameters:
        rkf: cross validation object

    output grid: GridSearchCV object
    """

    pipe = Pipeline([
        ('clf', clf)
            ]
        )
    grid = GridSearchCV(pipe, param_grid, iid=False, scoring='accuracy', cv=skf, n_jobs=-1, verbose=3)
    return grid




def TrainWithTuning(data_path,class_indices,skf=StratifiedKFold(n_splits=5, random_state=1291295159),
                clf=svm.SVC(),
                param_grid={"clf__kernel": ['linear', 'poly', 'rbf'],
                            "clf__C": [1,2,3]},
                img_sizes=[50,100,150,200,250],
                Hog_orientations=[7, 8, 9,10, 11],
                Hog_pixels_per_cell= [(8,8)],#[(6, 6),(7, 7),(8, 8),(9, 9), (10,10)],
                Hog_cells_per_block=[(4,4)],#[(2, 2), (3, 3), (4, 4), (5,5), (6,6)],
                Hog_block_norm=['L2']):


    """"
    documentation: this function train the whole pipeline with the given parameters using cross validation method and finds
    the optimal parameters.

    input data_path: the path to the data
    input class_indices: classes to apply tuning process on
    input skf: stratified cross validation object
    input clf: the classifier object
    input param_grid: dictionary with the parameter values to tune for the classifier
    input img_sizes: list of image sizes
    input Hog_pixels_per_cell: list of image sizes
    input Hog_cells_per_block: list of Hog cells per blocks
    input Hog_block_norm: list of Hog_block_norm

    output res:  dictionary where keys are the image size and HOG parameters and the values are cross validation objects
    for the given experiment
    """

    params = {}
    params['class_indices'] = class_indices
    res = {}
    for S in img_sizes:
        params['S'] = S
        print('S:  ', S)
        image_collection = load_data(params,data_path,tunning_mode=True)

        train_x, train_y, test_x, test_y = data_split(params,image_collection,tunning_mode=True,add_augmentations=False)#data_split(image_collection, class_indices, add_augmentations=False)
        for orientation in Hog_orientations:
            for pixel in Hog_pixels_per_cell:
                for cell in Hog_cells_per_block:
                    for norm in Hog_block_norm:
                        print('cell ', cell)
                        experiment = str(S) + "_" + str(orientation) + "_" + str(pixel[0]) + "_" + str(cell[0]) + "_" + str(norm)

                        params['orientation'] = orientation
                        params['pixels_per_cell'] = pixel
                        params['cells_per_block'] = cell
                        params['block_norm'] = norm

                        train_images, test_images = apply_HOG(params, train_x, test_x)

                        pipeline = _create_grid(skf,clf,param_grid)
                        print('Training the model...', len(train_images), train_y.shape)
                        pipeline.fit(train_images, train_y)
                        print(pipeline.best_score_)
                        print('The model is ready...')
                        res[experiment] = pipeline

    return res



def getMargins(params,score_matrix,labels):
    '''documentation: this function calculates the margin of all the pictures
    margin is defined as the distance of this test labels from the linear plane built by the svm
    margin=(distance from plane)-(the biggest distance from all models)

    input params: dict with parameters
    input score_matrix: a matrix of the distances of each picture[i] to each class[j]
    input labels: the true labels of the pictures

    output margins: an array of all the mistakes each row represents a class'''


    map_dict = {}
    for i in range(len(params['class_indices'])):
        map_dict[params['class_indices'][i]] = i

    margins = np.zeros([10,len(labels)])

    for i in range(len(labels)):
        numbOfClass = map_dict[labels[i]]
        margins[numbOfClass][i] = score_matrix[i, numbOfClass]-np.amax(score_matrix[i, :])
    return margins


def BIG_mistake(params,Margins,test_image_paths):
  '''documentation: this function will find for each class the two biggest misclassifications and present them
  for each row in in margines will take the two biggest elements and their index
  this index is the index of the image path the i all-ready collected before

  input params: dict of parameters
  input Margins: matrix of margins (row=class, col=imageNumber)
  input test_image_paths: list of test image paths

  '''
  list= params['class_indices']

  for i in range(len(list)):
        classMistakes = Margins[i]
        max1 = np.min(classMistakes)
        index1 = np.argmin(classMistakes)
        classMistakes[index1] = 0
        max2 = np.min(classMistakes)
        index2 = np.argmin(classMistakes)
        classMistakes[index1] = max1
        if(max1<0):# a mistake
            plt.title("biggest error in classifying  " + str(list[i]))
            image = cv2.imread(test_image_paths[index1])
            plt.imshow(image)
            plt.show()
            if (max2 < 0):
                plt.title("2nd biggest error in classifying " + str(list[i]))                                                              #, "the error was: "+str(max2)
                image = cv2.imread(test_image_paths[index2])
                plt.imshow(image)
                plt.show()
            else:
                print("only one mistakes in class "+ str(list[i]) +":) ")
        else:
            print("Amazing! no mistakes in "+ str(list[i]) +":) ")



if __name__ == '__main__':
    data_path = 'C:/Users/oreng/PycharmProjects/VisionProject1/101_ObjectCategories'
    #class_indices = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    class_indices = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]

    params = getDefaultParam(class_indices,data_path)
    image_collection, image_locationas = load_data(params, data_path)

    train_x, train_y, test_x, test_y, test_image_paths = data_split(params, image_collection, image_locationas, add_augmentations=False)

    train_images, test_images = apply_HOG(params, train_x, test_x)

    trained_model = m_classes_SVM_train(params, train_images, train_y)
    predictions, score_matrix = m_classes_SVM_predict(params, test_images, trained_model)

    if params['model_type'] == 'polynomial':
        print('Error rate for non linear SVM: ',1-(accuracy_score(np.array(test_y), predictions)))
        print(confusion_matrix(np.array(test_y), predictions))

    if params['model_type'] == 'linear':
        print('Error rate for linear SVM: ',1-(accuracy_score(np.array(test_y), predictions)))
        print(confusion_matrix(np.array(test_y), predictions))

    margins = getMargins(params, score_matrix, test_y)

    BIG_mistake(params, margins, test_image_paths)

    '''
    res = TrainWithTuning(data_path,class_indices)

    with open('hyperparameter_results.pickle', 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)'''





