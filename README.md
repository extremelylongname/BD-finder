# BD-Finder

BD-finder is a ML-based T-Y brown dwarf detection package for brown dwarfs developed in Biswas, A. 2023, in preparation. The primary function is to use YJHK and WISE magnitudes to determine if an object is a T-Y brown dwarf and if so, to determine the object's properties. Details are described in Biswas, A. 2023, in preparation. 

The detection software requires you to have a Pandas source catalog with columns `'Ymag', 'Jmag', 'Hmag', 'Kmag', 'W1mag', 'W2mag', 'W3mag'`, with each column signifying the AB magnitude of a source. To run the classifier, simply run:
```
from BDfinder.src import classifier
detected_sources = classifier.EnsembleClassifier(df)
```
Note that due to the large amount of machine learning models employed, the classifier may take up to an hour to run. The classifier will output several columns, among them `otypeclassified` and `sptypeclassified`. If `otypeclassified` is "substellar", the classifier identified the source as a T/Y dwarf. Furthermore, `sptypeclassified` indicates the classified spectral type of the object in question. 

**Since this is a machine learning model, its results may not be entirely accurate. As done in the source paper, it is strongly recommended to run several post-processing steps to filter down the candidates.**
