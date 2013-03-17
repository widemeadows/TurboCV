TurboCV
=======

This is a project providing a toolkit for convenient image feature extraction.
However, it is not limited to image processing. In fact, it contains some supplementary APIs to simplify C++ development, such as I/O related APIs.  

TurboCV has functions as follow.

1. Image Processing.  
    (a) Several morphological algorithms, including cleaning, thinning and edge-linking.  
    (b) Several feature extraction algorithms, including HOG, HOOSC, SC, GIST, Gabor, CM, OCM, Hitmap and their variants.

2. I/O Processing.  
    The core of this module is operating directories and files. These APIs are organized like .Net Framework.  
    Of course, we can find similar functions in Boost. However, Boost is somewhat overkilled and it is not easy to debug if some problems occur. What's more, we wish TurboCV independent of Boost, thus avoiding code modifications due to update of Boost.

3. String Processing.  
    This module provides some string related APIs that are missing in the C++ Standard Library. For example, split and type conversion (int to string, string to int, etc).

This toolkit is written **for fun**. Its purpose is to simplify development of "demo programs", not "commercial softwares". Therefore, it won't contain many codes to process exceptions or some unimportant details. In addition, the efficiency is not a priority.

~~*Currently there is no code here. I will commit them after some code cleaning.*~~
