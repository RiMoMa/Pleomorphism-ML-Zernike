### Folders De mask e Imagenes

# Este Codigo se usara para la extracción de caracteristicas de Zernike osea tiene obtencion de parches y extracción de parches de nucleos

# dataset folder
DatasetDir ='/home/ricardo/Documents/Doctorado/DatasetMedNorm/'
# dataset nuclei masks folder
MaskDir =  '/home/ricardo/Documents/Doctorado/Pleomorphism/NoiseletsImages/MaskNoiselet/'
# folders of each class will be combined with str folders
Clases = ['clase_1/','clase_2/','clase_3/']
# output folders
FolderOutputZernike = '/home/ricardo/Documents/Doctorado/Pleomorphism/ZernikePythonNo/' #Zernike output
FolderOutputDicImg = '/home/ricardo/Documents/Doctorado/Pleomorphism/DicPython/' #Dictionary output, this is a variable with all nucleus patches per image



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

#function of zernike moments
class ZernikeMoments:
    def __init__(self, radius, degree):
        # store the size of the radius that will be
        # used when computing moments
        self.radius = radius
        self.degree = degree

    def describe(self, image):
        # return the Zernike moments for the image
        return mahotas.features.zernike_moments_reconstruction(image, self.radius,self.degree)




## Radius used to extract zernike moments in each features
RadiusZernike1 = 30
RadiusZernike2 = 60
RadiusZernike3 = 90
degreeZernike=27 # level of decomposition
##


iniciTi = time.time() #time counter


for clase in range(1): #sacar las imagenes para el diccionario variando la clase
    ########################################################
    ###  To Build path of images each class at time ##############
    DirComplete = DatasetDir + Clases[clase] + '*'
    ImgFiles = glob.glob(DirComplete) #list the files
    Nimgs = range(len(ImgFiles)) #number of images
    print('Total Imgs per class ', clase)
    print('Total ',Nimgs)
################################################
    ############################################
    ## Open images and feature extraction ############3
    for imn in Nimgs:

        imPath = ImgFiles[imn] #image name
        ZernikePath = FolderOutputZernike + imPath[len(imPath[0:-12]):-4] +'.pickle' #file of zernike moments per image
        DicPath = FolderOutputDicImg + imPath[len(imPath[0:-12]):-4] +'.pickle' #
        im = Image.open(imPath) # Open image H&E
        maskPath = MaskDir+imPath[len(imPath[0:-12])::] #path of image nuclei mask
        if not os.path.exists(DicPath):
            # open Mask
            if not os.path.exists(maskPath):
                print('Please implement a nuclei detection rutine for ',maskPath)
                exit()
            else: imMask = Image.open(maskPath) ### Open mask when this exist

            # sacar RegionProps de la imagen
            #L = R * 299 / 1000 + G * 587 / 1000 + B * 114 / 1000
            imMask = imMask.convert('L')
            threshold = 100
            imMask = imMask.point(lambda p: p > threshold and 255) #binarize
            imMask = np.array(imMask)
            label_img = label(imMask)
            regions = regionprops(label_img,np.array(im)) #region props

            # Compute zernike Coefficients for each nucleus
            ImageDic=[]
            ImageDicZernike=[]
            i=0
            tic = time.time() #to compute zernike computational time per image
            #build zernike functions
            ComputeZernike1 = ZernikeMoments(RadiusZernike1, degreeZernike)
            ComputeZernike2 = ZernikeMoments(RadiusZernike2, degreeZernike)
            ComputeZernike3 = ZernikeMoments(RadiusZernike3, degreeZernike)
            for imNuclei in regions: # zernike feature extraction per nucleus
                NucleusIm = imNuclei.intensity_image #image of nucleus

                ImageDic.append(NucleusIm) # concatenate to an variable with all nuclei patches
                NucleusIm = rgb2gray(NucleusIm) # gray scale
                NucleusIm = cv2.copyMakeBorder(NucleusIm, 60, 60, 60, 60,
                                               cv2.BORDER_CONSTANT, value=0) # build a big patch to avoid small nucleus errors

                ZernikeCoef1 = ComputeZernike1.describe(NucleusIm) # level 1
                ZernikeCoef2 = ComputeZernike2.describe(NucleusIm) # level 2
                ZernikeCoef3 = ComputeZernike3.describe(NucleusIm) # level 3

                ImageDicZernike.append(np.concatenate([ZernikeCoef1, ZernikeCoef2, ZernikeCoef3]))


            toc = time.time()
            with open(ZernikePath, 'wb') as f:
                pickle.dump(ImageDicZernike, f)
            with open(DicPath, 'wb') as f:
                pickle.dump(ImageDic, f)


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