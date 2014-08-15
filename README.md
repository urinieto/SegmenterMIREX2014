SegmenterMIREX2014
==================

Segmentation algorithms for MIREX 2014.

It contains the following algorithms for boundary identification:

* Convex Non-negative Matrix Factorization (C-NMF)
* Foote
* Structural Features (SF)

And the following segment labeling algorithms:

* C-NMF
* 2D-Fourier Magnitude Coefficients (2D-FMC)


Version 1
---------

C-NMF for both boundaries and labels. To run:

    ./segmenter.py input.wav -o output.lab

Version 2
---------

Foote for boundaries and 2D-FMC for labels. To run:

    ./segmenter.py input.wav -o output.lab -s 2dfmc -b foote

Version 3
---------

SF for boundaries and 2D-FMC for labels. To run:

    ./segmenter.py input.wav -o output.lab -s 2dfmc -b sf

Version 4
---------

C-NMF for boundaries and 2D-FMC for labels. To run:

    ./segmenter.py input.wav -o output.lab -s 2dfmc


Requirements
------------

* Essentia 2.0.1
* cvxopt 1.1.7
* NumPy 1.8.2
* SciPy 0.14.0
* joblib 0.8.2 (for using the run_segmenter.py script to run segmenter on multiple files)
* mir_eval (for evaluating only)
* Pandas 0.14.1 (for evaluating only)

References
----------

Foote, J. (2000). Automatic Audio Segmentation Using a Measure Of Audio Novelty. 
In Proc. of the IEEE International Conference of Multimedia and Expo (pp. 452–455). 
New York City, NY, USA.

Nieto, O., & Bello, J. P. (2014). Music Segment Similarity Using 2D-Fourier Magnitude Coefficients. 
In Proc. of the 39th IEEE International Conference on Acoustics Speech and Signal Processing (pp. 664–668). 
Florence, Italy.

Nieto, O., & Jehan, T. (2013). Convex Non-Negative Matrix Factorization For Automatic Music Structure Identification. 
In Proc. of the 38th IEEE International Conference on Acoustics Speech and Signal Processing (pp. 236–240). 
Vancouver, Canada.

Serrà, J., Müller, M., Grosche, P., & Arcos, J. L. (2014). Unsupervised Music Structure Annotation by Time Series Structure Features and Segment Similarity. 
IEEE Transactions on Multimedia, Special Issue on Music Data Mining, 16(5), 1229 – 1240. 
doi:10.1109/TMM.2014.2310701

Author
------

[Oriol Nieto](https://files.nyu.edu/onc202/public/) (<oriol@nyu.edu>)
