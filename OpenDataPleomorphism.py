import pandas as pd
import numpy as np
import pickle
from PIL import Image
from sklearn.mixture import BayesianGaussianMixture

def openDataPleomorphism (DictionaryIDX,listPaths,y_train,FolderOutputZernike,FolderOutputDicImg,FolderCSV):
    X_train = listPaths
    AllLabels = []
    AllLabels_nuclei_size=[]
    AllLabels_anisonucleosis=[]
    AllLabels_nucleoli_size=[]
    AllLabels_chromatin_density=[]
    AllLabels_membrane_thickness=[]
    AllLabels_nuclei_contour=[]
    AllLabelsSublabels = []
    AllZernike = []
    AllDicNucleus = []

    for imn in range(len(DictionaryIDX)):
        imgD = DictionaryIDX[imn]
        imPath = X_train.loc[imgD]
        ZernikePath = FolderOutputZernike + imPath[len(imPath[0:-12]):-4] +'.pickle'
        DicPath = FolderOutputDicImg + imPath[len(imPath[0:-12]):-4] +'.pickle'
        csvOpen = pd.read_csv(FolderCSV+imPath[len(imPath[0:-12]):-4]+'_cna_criteria.csv',names=['0','1','2','3'])
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

#        im = Image.open(imPath)

        with open(ZernikePath, 'rb') as f:
            ZernikeCoefficients = pickle.load(f)

        with open(DicPath, 'rb') as g:
            DictionaryOfTheImage = pickle.load(g)

        AllZernike.extend(ZernikeCoefficients)
        AllDicNucleus.extend(DictionaryOfTheImage)
        Label = y_train.loc[imgD]
        ArrayOfLabels = np.matlib.repmat(Label, len(ZernikeCoefficients),1)[:,0]
        ArrayOfLabels2 = np.matlib.repmat(np.hstack((Label,OutDiag)), len(ZernikeCoefficients),1)
        ArrayOfLabels_nuclei_size = np.matlib.repmat(OutDiag[0], len(ZernikeCoefficients),1)[:,0]
        ArrayOfLabels_anisonucleosis = np.matlib.repmat(OutDiag[1], len(ZernikeCoefficients),1)[:,0]
        ArrayOfLabels_nucleoli_size = np.matlib.repmat(OutDiag[2], len(ZernikeCoefficients),1)[:,0]
        ArrayOfLabels_chromatin_density = np.matlib.repmat(OutDiag[3], len(ZernikeCoefficients),1)[:,0]
        ArrayOfLabels_membrane_thickness = np.matlib.repmat(OutDiag[4], len(ZernikeCoefficients),1)[:,0]
        ArrayOfLabels_nuclei_contour = np.matlib.repmat(OutDiag[5], len(ZernikeCoefficients),1)[:,0]

        AllLabels.extend(ArrayOfLabels)
        AllLabels_anisonucleosis.extend(ArrayOfLabels_anisonucleosis)
        AllLabels_membrane_thickness.extend(ArrayOfLabels_membrane_thickness)
        AllLabels_nucleoli_size.extend(ArrayOfLabels_nucleoli_size)
        AllLabels_nuclei_contour.extend(ArrayOfLabels_nuclei_contour)
        AllLabels_chromatin_density.extend(ArrayOfLabels_chromatin_density)
        AllLabels_nuclei_size.extend(ArrayOfLabels_nuclei_size)

        AllLabelsSublabels.extend(ArrayOfLabels2)
    dfAllLabels = pd.DataFrame({'NPofImage':AllLabels,'anisonucleosis': AllLabels_anisonucleosis, 'membrane_thickness': AllLabels_membrane_thickness,'nucleoli_size':AllLabels_nucleoli_size,
                       'nuclei_contour':AllLabels_nuclei_contour,'chromatin_density':AllLabels_chromatin_density,'nuclei_size':AllLabels_nuclei_size})
    AllZernike = np.array(AllZernike)
    return AllZernike, AllDicNucleus, AllLabels, df,dfAllLabels

def GenerateHistogramsPN (listPaths,y_train,FolderOutputZernike,FolderOutputDicImg,FolderCSV,Mod_dpgmm):
    X_train = listPaths
    AllLabels = []
    AllLabelsSublabels = []
    AllZernike = []
    AllDicNucleus = []
    AllHistograms=[]
    for imn in range(len(listPaths)):
        imgD = imn
        imPath = X_train.iloc[imgD]
        ZernikePath = FolderOutputZernike + imPath[len(imPath[0:-12]):-4] +'.pickle'
        DicPath = FolderOutputDicImg + imPath[len(imPath[0:-12]):-4] +'.pickle'
        csvOpen = pd.read_csv(FolderCSV+imPath[len(imPath[0:-12]):-4]+'_cna_criteria.csv',names=['0','1','2','3'])
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

#        im = Image.open(imPath)

        with open(ZernikePath, 'rb') as f:
            ZernikeCoefficients = pickle.load(f)

        with open(DicPath, 'rb') as g:
            DictionaryOfTheImage = pickle.load(g)
        bins_ = Mod_dpgmm.predict(np.array(ZernikeCoefficients))
        histo_img = np.zeros(( Mod_dpgmm.n_components))
        for occ in range(Mod_dpgmm.n_components):
            freq = np.sum(bins_ == occ)
            histo_img[ occ] = freq

        AllHistograms.append(histo_img)


        ### depronto a futuro se usa esto para ver la correspondencia nucleo a diccionario

        AllZernike.extend(ZernikeCoefficients)
        AllDicNucleus.extend(DictionaryOfTheImage)
#        Label = y_train.loc[imgD]
       # ArrayOfLabels = np.matlib.repmat(Label, len(ZernikeCoefficients), 1)
      #  ArrayOfLabels2 = np.matlib.repmat(np.hstack((Label,OutDiag)), len(ZernikeCoefficients), 1)
        #AllLabels.extend(ArrayOfLabels)
        #AllLabelsSublabels.extend(ArrayOfLabels2)

    return np.array(AllHistograms)