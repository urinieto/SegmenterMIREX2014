Convex Non-negative Matrix Factorization
========================================

To run:

    python segmenter.py input.wav -o output.lab

Requirements
------------

* Essentia 2.0.1
* PyMF (get it from here, otherwise it might not work: https://github.com/nils-werner/pymf)
* cvxopt 1.1.7
* NumPy 1.8.2
* SciPy 0.14.0
* joblib 0.8.2 (only for using the run_segmenter.py script to run multiple files)
* mir_eval (for evaluating only)
* Pandas 0.14.1 (for evaluating only)
