####################################################
####################################################
### DICTIONARY BUILDING FUNCTION############
############################################
def openPleomorphismFeatures(X_names20X, DF_AllFeatures40X,Dictionary):
    AllHistograms=[]
    Temp_DF = DF_AllFeatures40X.set_index('Idx20XName')

    for name20x in X_names20X:

    # set idx of names in a data frame
        DirComplete = Temp_DF.loc[name20x]
        Nimgs = range(len(DirComplete))
        histo_img = np.zeros(Dictionary.n_components)
        for imn in Nimgs:
            ZernikePath = DirComplete['ZernikePath'].iloc[imn]  # path of the features
            DicPath = DirComplete['DicPath'].iloc[imn]  # path of the nuclei patches
            if os.path.exists(ZernikePath):
                with open(ZernikePath, 'rb') as f:  # open Zernike Coeficients
                    ZernikeCoefficients = pickle.load(f)
                with open(DicPath, 'rb') as g:
                    DictionaryOfTheImage = pickle.load(g)
            else:
                print('falta implementacion')
            #ToDo: To implement routine of feature extraction found in main.py
            bins_train_as = Dictionary.predict(ZernikeCoefficients)  # assigne a cluster for each feature on the image
            #Todo: To implemnent histogram build This is now

            for occ in range(Dictionary.n_components):
                freq = np.sum(bins_train_as == occ)
                histo_img[occ] = freq

        AllHistograms.append(histo_img)  # concatenate features
    return AllHistograms


def DictionaryBuilding(X_train,y_train, DF_AllFeatures40X,ClustersDictionary,plotDictionary):

    AllZernike = []
    AllDicNucleus = []

    for clase in range(3):

        idx = y_train==clase+1#revisar ## Classes indexs
        Names20X = X_train[idx] # names of the class
        Temp_DF = DF_AllFeatures40X.set_index('Idx20XName') # set idx of names in a data frame
        DirComplete = Temp_DF.loc[Names20X]# chose names of the class on the 40x data frame using 20x Names
        DirComplete = DirComplete.iloc[np.random.permutation(len(DirComplete))] # permute to randomize
        Nimgs = range(len(DirComplete)) # number of images
        print('Total Imgs class ', clase)
        print('Total ', Nimgs)

        ## Run on the images of each class
        for imn in range(10): # Nimgs:
            ZernikePath = DirComplete['ZernikePath'][imn] #path of the features
            DicPath = DirComplete['ZernikePath'][imn] # path of the nuclei patches
            if os.path.exists(ZernikePath):
                with open(ZernikePath, 'rb') as f: #open Zernike Coeficients
                    ZernikeCoefficients = pickle.load(f)
                with open(DicPath, 'rb') as g:
                    DictionaryOfTheImage = pickle.load(g)
            else:
                print('nada')
                #T o D o :   T o implement routine of feature extraction found in main.py
            AllZernike.extend(ZernikeCoefficients) #concatenate features
            AllDicNucleus.extend(DictionaryOfTheImage) #concatenate patche from images
    X = np.array(AllZernike)
    from sklearn.mixture import BayesianGaussianMixture
    Dictionary = BayesianGaussianMixture(n_components=ClustersDictionary)
    Dictionary.fit(X)
    #ToDo: To implement dictionary visualization on a latent space tsne, routine already in 2_buildictionary_visualizarDic.py
    return Dictionary


############################################3
################################################33




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

ClustersDictionary=20
DatasetDir ='/home/ricardo/Documents/Doctorado/DatasetMedNorm/'
MaskDir =  '/home/ricardo/SegmentacionMetodo/'
Clases = ['clase_1/','clase_2/','clase_3/']

FolderOutputZernike = '/home/ricardo/Documents/Doctorado/Pleomorphism/ZernikePython/'
FolderOutputDicImg = '/home/ricardo/Documents/Doctorado/Pleomorphism/DicPython/'
FolderCSV = '/home/ricardo/Documents/Doctorado/Pleomorphism/Dataset/CSVx40Atypia/'

### Output Files
DatasetFrame ='/home/ricardo/Documents/Doctorado/Pleomorphism/DatasetFrameMitos.pickle'
DictionaryFolder = '/home/ricardo/Documents/Doctorado/Pleomorphism/KfoldTest/'
### Hacer el KFOLD por pacientes
# probaremos con un 70/30 para probar el concepto
AllLabels=[]
AllImPaths=[]
AllZernikePaths = []
AllDicNucleusPaths = []
AllIDXImage=[]
########## Generate DatasetFrame - ORGANIZE DATA############
#############################################

if not os.path.exists(DatasetFrame):
    for clase in range(3):
        ### seleccionar Imagenes
        DirComplete = DatasetDir + Clases[clase] + '*'
        ImgFiles = glob.glob(DirComplete)
        Nimgs = range(len(ImgFiles))
        print('Total Imgs clase ', clase)
        print('Total ', Nimgs)

        ## Recorrer lista de imagenes
        for imn in Nimgs:
            imPath = ImgFiles[imn]
            ZernikePath = FolderOutputZernike + imPath[len(imPath[0:-12]):-4] +'.pickle'
            DicPath = FolderOutputDicImg + imPath[len(imPath[0:-12]):-4] +'.pickle'
            csvOpen = pd.read_csv(FolderCSV+imPath[len(imPath[0:-12]):-4]+'_cna_criteria.csv',names=['0','1',2,3])
            im = Image.open(imPath)

            if clase==0:
                Label=1
            elif clase ==1:
                Label=2
            else:
                Label=3

            ArrayOfLabels = Label
            AllLabels.extend([ArrayOfLabels])
            AllImPaths.extend( [imPath])
            AllZernikePaths.extend( [ZernikePath])
            AllDicNucleusPaths.extend([DicPath])
            AllIDXImage.extend([imPath[len(imPath[0:-12]):-5]])

    data={'NPGrade':AllLabels,'ImPath':AllImPaths,'ZernikePath':AllZernikePaths,'DicPath':AllDicNucleusPaths,'Idx20XName':AllIDXImage}
    Nombres20X,idx20x=np.unique(AllIDXImage, return_index=True)
    AllLabels20x = np.array(AllLabels)[idx20x.astype(int)]
    data20x = {'image20x':Nombres20X,'Npgrade':AllLabels20x}

    DF_AllFeatures40X=pd.DataFrame(data)

    DF_AllFeatures20X = pd.DataFrame(data20x)
    with open(DatasetFrame, 'wb') as f:
        pickle.dump([DF_AllFeatures40X,Nombres20X,DF_AllFeatures20X], f)
else:
    with open(DatasetFrame, 'rb') as g:
        DF_AllFeatures40X,Nombres20X,DF_AllFeatures20X = pickle.load(g)


############################################################
##### MAIN FUNCTION##########################
#########################################################
if __name__ == '__main__':
    from sklearn.model_selection import StratifiedKFold
    X_all = DF_AllFeatures20X['image20x']  #X contain name of images at 20x
    y = DF_AllFeatures20X['Npgrade']  #Y are nuclear grade
    skf = StratifiedKFold(n_splits=5) # define kfold function
    skf.get_n_splits(X_all, y) #split data
    CountK = 0 #counter of folds

    Y_predictions=np.array([])
    Y_ground=np.array([])

    for train_index, test_index in skf.split(X_all, y):

        DictionaryPathSave = DictionaryFolder +'Dictionary_'+ str(CountK) +'_fold.pickle' #name to save dictionary on each fold
        #print("TRAIN:", train_index, "TEST:", test_index) #
        X_train, X_test = X_all[train_index], X_all[test_index] # generate train and test for each fold - DATA
        y_train, y_test = y[train_index], y[test_index] # generate train and test for each fold - LABELS

        if not os.path.exists(DictionaryPathSave):
            ######### Call Dictionary building function and then save
            Dictionary = DictionaryBuilding(X_train,y_train, DF_AllFeatures40X,ClustersDictionary,plotDictionary=False)

            with open(DictionaryPathSave, 'wb') as f:
                pickle.dump(Dictionary, f)
        else:
            with open(DictionaryPathSave, 'rb') as f:
                Dictionary = pickle.load(f)

        CountK = CountK + 1

        #Generate Histograms For Train and Test
        X_histograms_train = openPleomorphismFeatures(X_train, DF_AllFeatures40X,Dictionary)#open features
        X_histograms_test = openPleomorphismFeatures(X_test, DF_AllFeatures40X,Dictionary)#open features
        import seaborn as sns

        sns.set(rc={'figure.figsize': (11.7, 8.27)})
        palette = sns.color_palette("bright", 3)
        from sklearn.manifold import TSNE
        X = np.array(X_histograms_train)
        X_embedded = TSNE(n_components=2, metric='cosine', perplexity=30, init='pca', early_exaggeration=10).fit_transform(
            X)
        sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y_train, legend='full', palette=palette)
        plt.show()
        #Train machine
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(max_depth=2, random_state=0)
        clf.fit(X, y_train)
        clf.predict(np.array(X_histograms_test))
        #test Machine
        Y_predictK = clf.predict(np.array(X_histograms_test))
        Y_predictions = np.concatenate((Y_predictions,Y_predictK ))
        Y_ground = np.concatenate((Y_ground,np.array(y_test)))
        #compute metrics save

        # get your fucking phD

    print('Aqui voy Faltan Metricas')













Y_ = Mod_dpgmm.predict(X)
import numpy as np
from sklearn.manifold import TSNE
X = np.array(AllZernike)
X_embedded = TSNE(n_components=2,metric='cosine',perplexity=30,init='pca',early_exaggeration=40).fit_transform(X)
X_embedded.shape

### Sacar Caracteristicas del entrenamiento

### Sacar el Diccionario

#¿Cómo generar un diccionario confiable de nucleos de pleomorfismo?
#¿Cómo generar un diccionario confiable de nucleos de pleomorfismo?

#sacar un diccionario independiente con cada dataset. y mirar cuales son los clusters que permanecen sin cambio mirando los datos en cada cluster.
#sacar un diccionario unificado
# los nucleos que no se cruzan generan o no nuevos atomos






### Sacar el hiperplano sobre el Diccionario y encontrar los vectores de soporte con un svm lineas