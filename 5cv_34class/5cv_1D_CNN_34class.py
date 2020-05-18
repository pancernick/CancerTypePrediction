'''
This code is written by Milad Mostavi, one of authors of
"Convolutional neural network models for cancer type prediction based on gene expression" paper.
Please cite this paper in the case it was useful in your research
'''
import sys
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten
from keras.callbacks import EarlyStopping

# import tensorflow.compat.v1 as tf
from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations

# tf.disable_v2_behavior()


batch_size = 128
epochs = 50
seed = 7
np.random.seed(seed)

file = sys.argv[1]
# out_path = sys.argv[2]
full_data = pd.read_csv(file, dtype=None, low_memory=False)
data = full_data.iloc[:, 1:]
# data_shuffled = pd.DataFrame.to_numpy(shuffle(data.T).T)

# values specific for GSE99095
healthy_cell_cut = 392


X_cancer_samples = data.iloc[:-4, :healthy_cell_cut].T.values
X_normal_samples = data.iloc[:-4, healthy_cell_cut:].T.values

name_cancer_samples = ['Bone Marrow'] * len(X_cancer_samples)
name_normal_samples = ['Normal Samples'] * len(X_normal_samples)

X_cancer_samples_34 = np.concatenate((X_cancer_samples, X_normal_samples))
X_names = np.concatenate((name_cancer_samples, name_normal_samples))

# padding by zeros
X_cancer_samples_mat = np.concatenate((X_cancer_samples_34,np.zeros((len(X_cancer_samples_34),42))),axis=1)
X_cancer_samples_mat = np.reshape(X_cancer_samples_mat, (-1, 173, 100))


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []

cv_yscores = []
Y_test = []

input_Xs = X_cancer_samples_mat
y_s = X_names

img_rows, img_cols = len(input_Xs[0][0]), len(input_Xs[0])
num_classes = len(set(y_s))

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y_s)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

i = 0
for train, test in kfold.split(X_cancer_samples_34, y_s):   # input_Xs in normal case and shuffled should be shuffled_Xs

    input_Xs = input_Xs.reshape(input_Xs.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    input_Xs = input_Xs.astype('float32')

    num_classes = len(onehot_encoded[0])

    model = Sequential()
    # *********** First layer Conv
    model.add(Conv2D(32, kernel_size=(1, 71), strides=(1, 1),
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(1, 2))
    # ********* Classification layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])
    callbacks = [EarlyStopping(monitor='categorical_accuracy', patience=3, verbose=0)]
    if i == 0:
        model.summary()
        i = i + 1
    history = model.fit(input_Xs[train], onehot_encoded[train],
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0, callbacks=callbacks, validation_data=(input_Xs[test], onehot_encoded[test]))
    scores = model.evaluate(input_Xs[test], onehot_encoded[test], verbose=0)
    y_score = model.predict(input_Xs[test])
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    cvscores.append(scores[1] * 100)
    Y_test.append(onehot_encoded[test])
    cv_yscores.append(y_score)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

cv_yscores = np.concatenate(cv_yscores)
Y_test = np.concatenate(Y_test)

conf_matrix = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(cv_yscores, axis=1),labels=None, sample_weight=None, normalize=None)
# CREATE SALIENCY HEATMAP

# Swap softmax with linear
layer_idx = -1
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

# ndices = np.where(Y_test[:, class_bonemarrow] == 1.)[0]
class_bonemarrow = 0
class_healthy = 1

heatmap_class_marrow = visualize_saliency(model, layer_idx, seed_input=input_Xs, filter_indices=class_bonemarrow, backprop_modifier='guided', grad_modifier="absolute", keepdims=False)
heatmap_class_healthy = visualize_saliency(
    model, layer_idx, seed_input=input_Xs, filter_indices=class_healthy,
    backprop_modifier='guided', grad_modifier="absolute", keepdims=False)

cancer_heatmap_class_marrow = heatmap_class_marrow.reshape(17300,)[:-42, ]
cancer_heatmap_class_healthy = heatmap_class_healthy.reshape(17300,)[:-42, ]
gene_names = full_data.iloc[:-4, 0]
heatmap_matrix = np.vstack(
    (gene_names, cancer_heatmap_class_marrow, cancer_heatmap_class_healthy)).T

import matplotlib.ticker as ticker

import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.pyplot as plt

fig = plt.figure()
fig, ax = plt.subplots(1,1, figsize=(12,12))
heatplot_df = pd.DataFrame.from_records(data=heatmap_matrix, columns=['genes','class_marrow', 'class_healthy']).set_index('genes').astype('float')

heatplot0_df_100 = heatplot_df.sort_values(by=['class_marrow'], ascending=False).iloc[:100,:]

heatplot1_df_100 = heatplot_df.sort_values(by=['class_healthy'], ascending=False).iloc[:100,:]

# heatplot0_df_100 =heatplot_df.loc[heatplot_df['class_marrow'] >= 0.4] - filterd by some saliency values
heatplot = ax.imshow(heatplot0_df_100, cmap='BuPu')
ax.set_xticklabels(heatplot0_df_100.columns)
ax.set_yticklabels(heatplot0_df_100.index)
# PLOT THE EHATMAP!
tick_spacing = 1
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.set_title("Heatmap of Gene Expression in each class")
ax.set_xlabel('class')
ax.set_ylabel('Month')

plt.show()
print(f"Confusin matrix: TP|FP||FN|TN : {conf_matrix}")
