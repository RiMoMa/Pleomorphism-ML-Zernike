### Folders De mask e Imagenes

# Este Codigo se usara para la extracción de caracteristicas de Zernike

DatasetDir ='/home/ricardo/Documents/Doctorado/DatasetMedNorm/'
MaskDir =  '/home/ricardo/Documents/Doctorado/Pleomorphism/NoiseletsImages/MaskNoiselet/'
Clases = ['clase_1/','clase_2/','clase_3/']
FolderOutputZernike = '/home/ricardo/Documents/Doctorado/Pleomorphism/ZernikePython/'
FolderOutputDicImg = '/home/ricardo/Documents/Doctorado/Pleomorphism/DicPython/'

SamplingClasses = 10
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
class ZernikeMoments:
    def __init__(self, radius, degree):
        # store the size of the radius that will be
        # used when computing moments
        self.radius = radius
        self.degree = degree

    def describe(self, image):
        # return the Zernike moments for the image
        return mahotas.features.zernike_moments(image, self.radius,self.degree)







RadiusZernike1 = 30
RadiusZernike2 = 60
RadiusZernike3 = 90
degreeZernike=27

iniciTi = time.time()

for clase in range(2,4):
    ### seleccionar Imagenes
    DirComplete = DatasetDir + Clases[clase] + '*'
    ImgFiles = glob.glob(DirComplete)
    
    Nimgs = range(len(ImgFiles))
    print('Total Imgs clase ', clase)
    print('Total ',Nimgs)

    ## Recorrer lista de imagenes
    for imn in Nimgs:
        # sacar rutas de la imagen y de la mascara
        imPath = ImgFiles[imn]
        ZernikePath = FolderOutputZernike + imPath[len(imPath[0:-12]):-4] +'.pickle'
        DicPath = FolderOutputDicImg + imPath[len(imPath[0:-12]):-4] +'.pickle'
        im = Image.open(imPath)
        maskPath = MaskDir+imPath[len(imPath[0:-12])::]
        if not os.path.exists(DicPath):
            # abrir mascaras
            if not os.path.exists(maskPath):
                print('Please implement a nuclei detection rutine for ',maskPath)
                exit()
            else: imMask = Image.open(maskPath)


            # sacar RigionProps de la imagen
            imMask = imMask.convert('L')
            threshold = 100
            imMask = imMask.point(lambda p: p > threshold and 255)
            imMask = np.array(imMask)
            label_img = label(imMask)
            regions = regionprops(label_img,np.array(im))

            #definir instancia de zernike con el radio determinado



            # Sacar Coeficientes de zernike de todas las regiones
            ImageDic=[]
            ImageDicZernike=[]
            i=0
            tic = time.time()

            ComputeZernike1 = ZernikeMoments(RadiusZernike1, degreeZernike)
            ComputeZernike2 = ZernikeMoments(RadiusZernike2, degreeZernike)
            ComputeZernike3 = ZernikeMoments(RadiusZernike3, degreeZernike)
            for imNuclei in regions:
                NucleusIm = imNuclei.intensity_image
                ImageDic.append(NucleusIm)
                NucleusIm = rgb2gray(NucleusIm)
                NucleusIm = cv2.copyMakeBorder(NucleusIm, 60, 60, 60, 60,
                                               cv2.BORDER_CONSTANT, value=0)

                ZernikeCoef1 = ComputeZernike1.describe(NucleusIm)
                ZernikeCoef2 = ComputeZernike2.describe(NucleusIm)
                ZernikeCoef3 = ComputeZernike3.describe(NucleusIm)

                ImageDicZernike.append(np.concatenate([ZernikeCoef1, ZernikeCoef2, ZernikeCoef3]))


            toc = time.time()
            with open(ZernikePath, 'wb') as f:
                pickle.dump(ImageDicZernike, f)
            with open(DicPath, 'wb') as f:
                pickle.dump(ImageDic, f)
            #  with open('mypickle.pickle', 'rb') as f:
            #    loaded_obj = pickle.load(f)

            print('Tiempo Caracterizacion ',toc-tic)
            print('Total nucleos caracterizados',len(ImageDicZernike))
        else:
            print('Procces already did for ',DicPath)





iniciTF = time.time()


print('Tiempo total de extraccion',iniciTF-iniciTi)


print('acabamos')







#abrir Imagen

#abrir mascara
#extraer nucleo a nucleo

#guardar diccionario de nucleos?
#extraer caracteristicas de los nucleos

#clusterizar nucleos de la imagenes de grado 1 y luego de grado 2

#entrenar un clasificador

#mirar los vectores de soporte y su distancia al hiperplano