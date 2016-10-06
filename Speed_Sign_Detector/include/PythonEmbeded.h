/* 
 * File:   PythonCNN.h
 * Author: joseph
 *
 * Created on September 23, 2016, 8:15 PM
 */

#ifndef PYTHONCNN_H
#define	PYTHONCNN_H
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"                 // Needs to be first in the file
#include "numpy/arrayobject.h"      
#include <opencv2/core/mat.hpp>
#include <iostream>
#include <vector>

#define NUM_OUTPUT 11               // CNN output [25:5:75]
#define MIN_CONFI 0.98              // Minimum sign confidence
class PythonEmbeded {
public:
    PythonEmbeded();
    PythonEmbeded(const PythonEmbeded& orig);
    virtual ~PythonEmbeded();
    int setupPython();
    int RunCNN(cv::Mat &in);
    int CleanPython();
    int InterpretResults();
private:
    PyObject *pModule, *pFunc, *pName,*sys, *path;
    bool isInit;
    bool newReults;
    double results[NUM_OUTPUT];

};

#endif	/* PYTHONCNN_H */

