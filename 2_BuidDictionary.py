import glob
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, regionprops_table
import cv2
import time
from skimage.color import rgb2gray
import pickle
import mahotas
import numpy.matlib
import pandas as pd
from sklearn.manifold import TSNE
### Folders De mask e Imagenes

# Este Codigo se usara para la extracción de caracteristicas de Zernike

DatasetDir ='/home/ricardo/Documents/Doctorado/DatasetMedNorm/'
MaskDir =  '/home/ricardo/SegmentacionMetodo/'
Clases = ['clase_1/','clase_2/','clase_3/']


FolderOutputZernike = '/home/ricardo/Documents/Doctorado/Pleomorphism/ZernikePython/'
FolderOutputDicImg = '/home/ricardo/Documents/Doctorado/Pleomorphism/DicPython/'
FolderCSV = '/home/ricardo/Documents/Doctorado/Pleomorphism/Dataset/CSVx40Atypia/'
### Hacer el KFOLD por pacientes
# probaremos con un 70/30 para probar el concepto
AllLabels=[]
AllLabelsSublabels=[]
AllZernike = []
AllDicNucleus = []

for clase in range(3):
    ### seleccionar Imagenes
    DirComplete = DatasetDir + Clases[clase] + '*'
    ImgFiles = glob.glob(DirComplete)

    Nimgs = range(len(ImgFiles))
    print('Total Imgs clase ', clase)
    print('Total ', Nimgs)

    ## Recorrer lista de imagenes
    for imn in range(15): # Nimgs:
        imPath = ImgFiles[imn]
        ZernikePath = FolderOutputZernike + imPath[len(imPath[0:-12]):-4] +'.pickle'
        DicPath = FolderOutputDicImg + imPath[len(imPath[0:-12]):-4] +'.pickle'
        csvOpen = pd.read_csv(FolderCSV+imPath[len(imPath[0:-12]):-4]+'_cna_criteria.csv',names=['0','1',2,3])
        ## Sort the CSV data
        OutDiag = []#concenso de los expertos
        for a in range(len(csvOpen)):
            Chist, V = np.histogram(np.array(csvOpen)[a, 1::], bins=3, range=(1, 3))
            if Chist[0]==Chist[1] & Chist[0]==Chist[2] & Chist[1]==Chist[2]:
                OutDiag.append(2)
            else:
                P = np.argmax(Chist)
                OutDiag.append(P+1)
        cars = {imPath[len(imPath[0:-12]):-4]: OutDiag}
        if imn==0: # and clase==2:
            df = pd.DataFrame(cars, index=csvOpen.iloc[:, 0])#df guarda el dataframe del diccionario o de los casos del for
        else:
            df[imPath[len(imPath[0:-12]):-4]]=OutDiag
        print(df)

        im = Image.open(imPath)

        with open(ZernikePath, 'rb') as f:
            ZernikeCoefficients = pickle.load(f)

        with open(DicPath, 'rb') as g:
            DictionaryOfTheImage = pickle.load(g)

        AllZernike.extend(ZernikeCoefficients)
        AllDicNucleus.extend(DictionaryOfTheImage)
        if clase==0:
            Label=1
        elif clase ==1:
            Label=2
        else:
            Label=3
        ArrayOfLabels = np.matlib.repmat(Label, len(ZernikeCoefficients), 1)
        ArrayOfLabels2 = np.matlib.repmat(np.hstack((Label,OutDiag)), len(ZernikeCoefficients), 1)
        AllLabels.extend(ArrayOfLabels)
        AllLabelsSublabels.extend(ArrayOfLabels2)
columnsAll = csvOpen.iloc[:, 0].to_list()
columnsAll.insert(0,'NPO Grade')
DF_AllFeatures=pd.DataFrame(AllLabelsSublabels,columns=columnsAll)
print('Aca Voy')

Indices = DF_AllFeatures[DF_AllFeatures['NPO Grade']==1].index.values
Indices2 = DF_AllFeatures[DF_AllFeatures['NPO Grade']==2].index.values
X = np.array(AllZernike)



from sklearn.mixture import BayesianGaussianMixture

#elegir anisonucleosis

#idx_d = DF_AllFeatures['anisonucleosis']==2
#X = np.array(AllZernike)[idx_d,:]

Mod_dpgmm = BayesianGaussianMixture(n_components=10)
Mod_dpgmm.fit(X)
#Y_ = Mod_dpgmm.predict(X)

#idx_d = DF_AllFeatures['anisonucleosis']==3
#X2 = np.array(AllZernike)[idx_d,:]
Y_ = Mod_dpgmm.predict(X)
#Y_2 = Y_>0.99
#Y_3 = Y_2.any(axis=1)
#Y_3 = np.logical_not(Y_3)
#AllDicNucleusSel = []
#Indices3 = Indices2[Y_3]
#for index in Indices3:
#    AllDicNucleusSel.append(AllDicNucleus[index])

# for index in range(len(idx_d)):
#     if idx_d[index]==True:
#         AllDicNucleusSel.append(AllDicNucleus[index])



# ######## BUNDO EL BIC
#
# import numpy as np
# import itertools
#
# from scipy import linalg
# import matplotlib.pyplot as plt
# import matplotlib as mpl
#
# from sklearn import mixture
#
# print(__doc__)
#
# # Number of samples per component
# n_samples = 500
#
# # Generate random sample, two components
# np.random.seed(0)
#
# lowest_bic = np.infty
# bic = []
# n_components_range = range(1, 10)
# cv_types = ['spherical', 'tied', 'diag', 'full']
# for cv_type in cv_types:
#     for n_components in n_components_range:
#         # Fit a Gaussian mixture with EM
#         gmm = mixture.GaussianMixture(n_components=n_components,
#                                       covariance_type=cv_type)
#         gmm.fit(X)
#         bic.append(gmm.bic(X))
#         if bic[-1] < lowest_bic:
#             lowest_bic = bic[-1]
#             best_gmm = gmm
#
# bic = np.array(bic)
# color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
#                               'darkorange'])
# clf = best_gmm
# bars = []
#
# # Plot the BIC scores
# plt.figure(figsize=(8, 6))
# spl = plt.subplot(2, 1, 1)
# for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
#     xpos = np.array(n_components_range) + .2 * (i - 2)
#     bars.append(plt.bar(xpos, bic[i * len(n_components_range):
#                                   (i + 1) * len(n_components_range)],
#                         width=.2, color=color))
# plt.xticks(n_components_range)
# plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
# plt.title('BIC score per model')
# xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
#     .2 * np.floor(bic.argmin() / len(n_components_range))
# plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
# spl.set_xlabel('Number of components')
# spl.legend([b[0] for b in bars], cv_types)
#
# # Plot the winner
# splot = plt.subplot(2, 1, 2)
# Y_ = clf.predict(X)
# for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
#                                            color_iter)):
#     v, w = linalg.eigh(cov)
#     if not np.any(Y_ == i):
#         continue
#     plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
#
#     # Plot an ellipse to show the Gaussian component
#     angle = np.arctan2(w[0][1], w[0][0])
#     angle = 180. * angle / np.pi  # convert to degrees
#     v = 2. * np.sqrt(2.) * np.sqrt(v)
#     ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
#     ell.set_clip_box(splot.bbox)
#     ell.set_alpha(.5)
#     splot.add_artist(ell)
#
# plt.xticks(())
# plt.yticks(())
# plt.title(f'Selected GMM: {best_gmm.covariance_type} model, '
#           f'{best_gmm.n_components} components')
# plt.subplots_adjust(hspace=.35, bottom=.02)
# plt.show()
#


import numpy as np
from sklearn.manifold import TSNE
X = np.array(AllZernike)
X_embedded = TSNE(n_components=2,metric='cosine',perplexity=10).fit_transform(X)
X_embedded.shape

import seaborn as sns
palette = sns.color_palette("bright", 3)
#sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=Y_, legend='full', palette=palette)
# for YY in range(3):
#     fig, axes = plt.subplots(50, figsize=(26,26)) # width, height in inches
#     c=0
#     condicion=np.array(Y_==YY)
#     for i in range(len(AllDicNucleus)):
#
#         if condicion[i]==True:
#
#             axes[c].imshow(AllDicNucleus[i])
#             if c < 49:
#                 c = c + 1
#             else:
#                 print('esto toca mejorarle')
#
#         else:
#             print('otra clase')
#     plt.show()
# print('uu')

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

# extract x and y coordinates representing the positions of the images on T-SNE plot
tx = X_embedded[:, 0]
ty = X_embedded[:, 1]

tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)

#cuando se ve una clase
tx = tx
ty = ty

# Compute the coordinates of the image on the plot
def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center,
    # compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y

colors_per_class = {
    0 : [254, 202, 87],
    1 : [255, 107, 107],
    2 : [10, 189, 227],
    3 : [255, 159, 243],
    4 : [16, 172, 132],
    5 : [128, 80, 128],
    6 : [87, 101, 116],
    7 : [52, 31, 151],
    8 : [0, 0, 0],
    9 : [100, 100, 255],
    10: [34,355,50]
}

def draw_rectangle_by_class(image, label):
    image_height, image_width, _ = image.shape

    # get the color corresponding to image class
    color = colors_per_class[label]
    #color = colors_per_class[2]
    image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=color, thickness=1)

    return image
# we'll put the image centers in the central area of the plot
# and use offsets to make sure the images fit the plot
offset = 100 // 2

plot_size=2000
image_centers_area_size = plot_size - 2 * offset
# init the plot as white canvas
tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)
from tqdm import tqdm
# now we'll put a small copy of every image to its corresponding T-SNE coordinate
labels=Y_
for image_path, label, x, y in tqdm(
        zip(AllDicNucleus, labels, tx, ty),
        desc='Building the T-SNE plot',
        total=len(AllDicNucleus)
):
    image = image_path

    # scale the image to put it to the plot
    image = image

    # draw a rectangle with a color corresponding to the image class
    image = draw_rectangle_by_class(image, label)

    # compute the coordinates of the image on the scaled plot visualization
    tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)

    # put the image to its t-SNE coordinates using numpy sub-array indices
    tsne_plot[tl_y:br_y, tl_x:br_x, :] = image
cv2.imwrite('allNP3ansinocleosis43.png',tsne_plot)#
#cv2.imshow('t-SNE', tsne_plot)
#cv2.waitKey()
print('mejor_????') 
### Sacar Caracteristicas del entrenamiento

### Sacar el Diccionario

#¿Cómo generar un diccionario confiable de nucleos de pleomorfismo?

#sacar un diccionario independiente con cada dataset. y mirar cuales son los clusters que permanecen sin cambio mirando los datos en cada cluster.
#sacar un diccionario unificado
# los nucleos que no se cruzan generan o no nuevos atomos






### Sacar el hiperplano sobre el Diccionario y encontrar los vectores de soporte con un svm lineas