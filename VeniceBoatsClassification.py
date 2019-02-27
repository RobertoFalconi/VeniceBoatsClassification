import os
import re
#import tensorflow.python.platform
#from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC, LinearSVC
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import seaborn as sn

BARCHE = ['Alilaguna','Ambulanza','Barchino','Cacciapesca','Gondola','Lanciafino10m','Lanciafino10mBianca','Lanciafino10mMarrone','Lanciamaggioredi10mBianca','Lanciamaggioredi10mMarrone','Motobarca', 'Motopontonerettangolare', 'MotoscafoACTV', 'Mototopo','Patanella','Polizia','Raccoltarifiuti','Sandoloaremi','Sanpierota','Topa','VaporettoACTV','Vigilidelfuoco' ]

def extract_features(file_dir):
    
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)

    vgg19_feature_list = []
    classe = []
    list_files = [file_dir+f for f in os.listdir(file_dir)]
    for idx, dirname in enumerate(list_files):
        list_images = [dirname+'/'+f for f in os.listdir(dirname) if re.search('jpg|JPG', f)]
        class_name = list_files[idx]
        class_name = class_name[42:]
        for i, fname in enumerate(list_images):

            print('Processing image: '+list_images[i]+' '+ str(i+1) +'/'+str(len(list_images)))

            img = image.load_img(fname, target_size=(224, 224))
            img_data = image.img_to_array(img).reshape()
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)

            vgg19_feature = model.predict(img_data)
            vgg19_feature_np = np.array(vgg19_feature)
            vgg19_feature_list.append(vgg19_feature_np.flatten())
            classe.append(class_name)    

    se = pd.Series(classe)
    vgg19_feature_list_np = np.array(vgg19_feature_list)
    np_data = pd.DataFrame(vgg19_feature_list_np)
    np_data['Class'] = se.values
    return np_data
    

def extract_features_test(file_dir):
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)
    vgg19_feature_list = []
    classe_test= []
    file_dir_test = file_dir+'ground_truth.txt'
    Barche_considerate = BARCHE
    with open(file_dir_test, 'r') as f:
        x = f.readlines()
        for line in x:
            name_img = line[:21]
            tipo = line[26:]
            tipo = tipo.replace(' ','')
            tipo = tipo.replace(':','')
            tipo = tipo.strip()
            
            if tipo in Barche_considerate:
                
                fname = file_dir+name_img+'.jpg'
                img = image.load_img(fname, target_size=(224, 224))
                img_data = image.img_to_array(img)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)

                vgg19_feature = model.predict(img_data)
                vgg19_feature_np = np.array(vgg19_feature)
                vgg19_feature_list.append(vgg19_feature_np.flatten())
                classe_test.append(tipo)
                

    se = pd.Series(classe_test)
    vgg19_feature_list_np = np.array(vgg19_feature_list)
    np_data = pd.DataFrame(vgg19_feature_list_np)
    np_data['Class'] = se.values
    return np_data    

def print_results(nome_classificatore, y_true, y_pred):
    
    # CONFUSION MATRIX

    print(nome_classificatore + " Confusion Matrix: ", (confusion_matrix(y_true, y_pred)))

    # ACCURACY

    print(cross_val_score(nome_classificatore, y_true, y_pred, scoring='accuracy'))

    # BALANCED ACCURACY 

    print(cross_val_score(nome_classificatore, y_true, y_pred, scoring='balanced_accuracy'))

    # PRECISION SCORE

    print(cross_val_score(nome_classificatore, y_true, y_pred, scoring='precision'))

    # RECALL SCORE

    print(cross_val_score(nome_classificatore, y_true, y_pred, scoring='recall'))

    # F1 SCORE

    print(cross_val_score(nome_classificatore, y_true, y_pred, scoring='f1'))


file_dir_test = 'dataset\\test_set\\'
file_dir = 'dataset\\training_set\\'

pd_data_test = extract_features_test(file_dir_test)
y_true = pd_data_test['Class']
X_test = pd_data_test.drop('Class', 1)

pd_data = extract_features(file_dir)
y_train = pd_data['Class']
X_train = pd_data.drop('Class', 1)

rfclassifier = RandomForestClassifier()  
rfclassifier.fit(X_train, y_train)  
y_pred = rfclassifier.predict(X_test) 
con_mat = confusion_matrix(y_true,y_pred)
print_results("Random Forest", y_true, y_pred)

print(classification_report(y_true,y_pred))  
df_cm = pd.DataFrame(con_mat, index = BARCHE, columns =BARCHE)
plt.figure(figsize = (7,5))
sn.heatmap(df_cm, annot=True)
plt.ion()
plt.show(block=False)
input('press <ENTER> to continue')

svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)  
y_pred = svclassifier.predict(X_test) 
con_mat = confusion_matrix(y_true,y_pred)
print_results("SVM", y_true, y_pred)

print(classification_report(y_true,y_pred))  
df_cm = pd.DataFrame(con_mat, index = BARCHE, columns =BARCHE)
plt.figure(figsize = (7,5))
sn.heatmap(df_cm, annot=True)
plt.ion()
plt.show(block=False)
input('press <ENTER> to continue')