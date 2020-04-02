# AI_HW_6

Mesterséges Intelligencia házi feladat #5

_Delta rule and binary classification_

The __readme__ is also available in English [here](#tasks).

## Feladatok
1. Két osztály megkülönböztetése kétrétegű neuronhálóval. Adathalmaz [innen](http://www.vision.caltech.edu/Image_Datasets/Caltech101). Krokodilok VS Sztegoszauruszok
2. A hibafüggvény a négyzetes hibafüggvényt legyen.
3. Jelenítse meg grafikusan a tesztelési halmaz egy részét az osztályozó kimenetével.
4. Jelenítse meg a konfúziós mátrixot.

### Felmerülő problémák
1. előfeldolgozás: képek átméretezése 64x64-es méretre, szürkeárnyalatos színcsatornává való konvertálás
2. adathalmaz mérete: nem pontos a tanítás, mert kicsi az adathalmaz (jelenleg a Pareto-elv szerint 80-20%-os felosztás)
3. lassú konvergálás 

A program Python nyelven lettek írva, 3.7-es verzióval. 
___
___

## Tasks
1. Classify images belonging to 2 categories with a bilayer neural network. The dataset can be found [here](http://www.vision.caltech.edu/Image_Datasets/Caltech101). Crocodiles VS Stegosauruses
2. The loss function should be the quadratic error function.
3. Graphically display a portion of the test set with the classifier's output.
4. Display the confusion matrix.

### Faced issues
1. preprocessing: resizing images into 64x64, converting to grayscale
2. the size of the dataset: not accurate teaching because the dataset is too small (currently 80-20% split according to Pareto principle)
3. slow convergence

It was written in Python, version 3.7.
