# Pleomorphism-ML-Zernike

Este repositorio contiene experimentos para caracterizar el pleomorfismo nuclear en imágenes histológicas mediante técnicas de visión por computador y aprendizaje automático.

## Estructura del proyecto
- **main.py**: extrae coeficientes de momentos de Zernike de cada núcleo y guarda los resultados en archivos `pickle` para su análisis posterior.
- **2_BuidDictionary.py**: reúne los coeficientes generados, calcula etiquetas a partir de archivos CSV, agrupa los núcleos con un modelo de mezcla gaussiana bayesiana y ofrece utilidades de visualización con t‑SNE.
- **OpenDataPleomorphism.py**: funciones auxiliares para cargar los coeficientes y generar histogramas de ocurrencias de los núcleos según el diccionario construido.
- **ExperimentsForSubClass.py**: ejecuta experimentos de *K-fold* para construir diccionarios por subconjuntos y obtener histogramas de entrenamiento y prueba.
- **VisualizarTSNE.py**: ejemplo de proyección de características en 2D con t‑SNE utilizando descriptores obtenidos mediante una red ResNet.

## Requisitos
Python 3.8 o superior y las siguientes bibliotecas: `numpy`, `pillow`, `matplotlib`, `scikit-image`, `opencv-python`, `mahotas`, `pandas`, `scikit-learn`, `tqdm`, `torch` y `torchvision` (estas dos últimas solo para `VisualizarTSNE.py`).

## Uso básico
1. Ajustar las rutas de directorios en cada script (`DatasetDir`, `MaskDir`, `FolderOutputZernike`, etc.).
2. Ejecutar `python main.py` para extraer y guardar los coeficientes de Zernike de los núcleos segmentados.
3. Ejecutar `python 2_BuidDictionary.py` o `python ExperimentsForSubClass.py` para construir el diccionario de núcleos, agrupar con *Gaussian Mixture* y generar histogramas.
4. (Opcional) Utilizar `VisualizarTSNE.py` para proyectar las características en 2D y visualizar la distribución de los núcleos.

## Técnicas empleadas
- Momentos de **Zernike** para describir la forma de cada núcleo en distintos radios.
- **Modelos de mezcla gaussiana bayesiana** para agrupar y construir un diccionario de formas nucleares.
- **t‑SNE** y arquitecturas **ResNet** para visualizar y explorar las características en espacios de menor dimensión.
- Procedimientos de validación como **K-fold** para evaluar modelos.
