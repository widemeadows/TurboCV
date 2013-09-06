TurboCV
=======

This is a project providing a toolkit for convenient image feature extraction.
However, it is not limited to image processing. In fact, it contains some supplementary APIs to simplify C++ development, such as I/O related APIs.  

Currently TurboCV has modules as follow.  

1. **System**  
    The core module of TurboCV.  
    (a) Collection.h and Type.h define the data structures used in other modules, including Int, Double, Float, String, Group and ArrayList. Int, Double, Float and String provide some string related routines that are missing in std::string. For example, split and type conversion (int to string, string to int, etc). Group is also named as Tuple in other languages (in fact, it's same with std::tuple but we provide it here for non-C++11-supported complier). ArrayList is similar to std::vector except that the copy constructor and assigment operation in ArrayList is O(1), i.e. only the header is copied and the data are shared.  
    (b) FileSystem.h defines some routines for operating directories and files. These APIs are organized like .Net Framework. Currently these routines don't support Unicode paths, i.e. no Chinese characters are allowed.  
    (c) Math.h defines some basic math routines, including the calculation of Min, Max, Sum, Mean, STD, Gauss, Gauss Derivation, Norm-One Distance, Norm-Two Distance and RBF Distance.     
    (d) Util.h contains some rand-related routines. RandomPermuate is used to randomly pick up some number from [0, card_num - 1]. card_num is given by the parameters. Other routines work in a similar way.  

2. **System.Image**  
    This module provides routines for image processing.
    (a) BinaryImage.h provides several morphological algorithms, including cleaning and thinning.  
    (b) BOV.h provides the Bag-of-Visual-Words algorithm.  
    (c) EdgeLink.h provides a naive edge segmentation algorithm. This algorithm simply look up end points and junction points (together named as segpoints) in an edge map, and then it segments edges between each two segpoints.  
    (d) EdgeMatching.h provides the implementation for CM, OCM and HIT.  
    (e) Eigen.h provides two adapters for Eigen3 and OpenCV2, i.e. it allows using cv::Mat in Eigen3 to get generalized eigenvectors.  
    (f) Feature.h provides the implementation for WIND, SIGGRAPH-HOG, GHOG, HOG, HOOSC, SC, PSC, GIST and some variants (note: they're optimized for sketch recognition, so don't use them in general image recognition).  
    (g) Filter.h provides the Gauss-Deviation kernel and the Laplacian-of-Gaussian kernel. It also provides routines to split a sketch into several channels according to orientations.  
    (h) Geometry.h provides routines to calculate Euler Distance and Angle. Routines to sample points from sketches are provided in this file as well.  
    (i) Util.h provides some help functions, such as normalization.  

3. **System.ML**  
    This module provides the implementation for KNN, MQDF, SNE and t-SNE.  

4. **ExperimentCPP**  
    This module contains some experimental codes. The main entry is in this module as well. In order to run Cross-Validation on each algorithm, e.g. HIT, and so on, simply used the codes provided in the Batch function.  

5. **System.CS** and **ExperimentCS**  
    System.CS provides C# version of some codes mentioned in the previous modules. However, only a little C++ codes are changed into C# codes. ExperimentCS contains the main entry and some experimental codes.  

6. **ClrAdapter** and **Export**  
    These two modules export some C++ routines into C# interfaces. Different from System.CS and ExperimentCS, these two modules only invoke the native C++ DLL in managed C# codes, instead of providing the implementation directly. Export exports functions from native C++ into a DLL file named Export.dll. ClrAdapter invokes Export.dll and wrap them with C++/Cli, and then export the C++/Cli interfaces into a DLL file named ClrAdapter.dll. That is to say, if add ClrAdpater.dll and Export.dll into your project, you can use interfaces provided in ClrAdpater (with managed codes such as C#) though they will be executed by Export.dll (with native C++).  

This toolkit is written **for fun**. Its purpose is to simplify development of "demo programs", not "commercial softwares". Therefore, it won't contain many codes to process exceptions or some unimportant details. In addition, the efficiency is not a priority.

~~*Currently there is no code here. I will commit them after some code cleaning.*~~
