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

Author
------

[Oriol Nieto](https://files.nyu.edu/onc202/public/) (<oriol@nyu.edu>)
