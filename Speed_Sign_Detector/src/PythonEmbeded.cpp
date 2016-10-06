/* 
 * File:   PythonCNN.cpp
 * Author: joseph
 * 
 * Created on September 23, 2016, 8:15 PM
 */

#include "PythonEmbeded.h"

void init_numpy()
{
    import_array();
}

PythonEmbeded::PythonEmbeded() {
    isInit = false;
    newReults = true;
}

PythonEmbeded::PythonEmbeded(const PythonEmbeded& orig) {
}

PythonEmbeded::~PythonEmbeded() {
}

int PythonEmbeded::CleanPython()
{
    if(!isInit)
        return 0;
    
    Py_DECREF(pFunc);
    Py_DECREF(pModule);
    Py_Finalize();      // Shut down python interpreter
    isInit = false;
    return 1;
}

int PythonEmbeded::InterpretResults()
{
    // Index: 25,30,35,40,45,50,55,60,65,70,75
    double MaxConf = 0;    
    int sign_num = -1;  // Default case of no sign found
    if(!newReults)
        return sign_num;
    
    //Search for the sign carrying the highest confidence
    for(int i =0; i< NUM_OUTPUT; i++)    
        if(results[i] >= MIN_CONFI)        
            if(results[i] > MaxConf)
            {
                MaxConf = results[i];
                sign_num = 5*i + 25;        // Convert index to sign number
            }
    newReults = false;
    return sign_num;
}

int PythonEmbeded::RunCNN(cv::Mat &in){
    // Extract data    
    uchar * ImgData = in.data;
    int len = in.rows*in.cols;
    
    // Prepare 1D python list
    npy_intp dims[1]{len};
    const int ND{ 1 };
    
    PyArrayObject * np_arg = reinterpret_cast<PyArrayObject*>
            (PyArray_SimpleNewFromData(ND, dims, NPY_UINT8, 
             reinterpret_cast<void*>(ImgData)));
    
    // Prepare passed argument
    PyObject * pArgs = PyTuple_New(1);
    // pArgs will steal reference from np_arg.
    PyTuple_SetItem(pArgs, 0, reinterpret_cast<PyObject*>(np_arg)); 
    PyObject * pReturn = PyObject_CallObject(pFunc, pArgs);
                  
    if (!PyList_Check(pReturn)){        
        std::cout << "Python module failed to return expected single list" << std::endl;
        
        Py_DECREF(np_arg);               // Dereference PyObjects
        Py_DECREF(pReturn);   
        return 0;
    }
    
    int count = (int) PyList_Size(pReturn);    
    if(count != NUM_OUTPUT)
    {
        std::cout << "Python module returned list of size "
                  << count<<" but expected "<<NUM_OUTPUT << std::endl;
        
        Py_DECREF(np_arg);               // Dereference PyObjects
        Py_DECREF(pReturn);   
        return 0;
    }
    
    // off load list items    
    for(int idx =0; idx < count ; idx++)
    {
        PyObject *pListItem = PyList_GetItem(pReturn, idx);
        results[idx] = PyFloat_AsDouble(pListItem);        
    }
    
    newReults = true;
    // Dereference PyObjects
    Py_DECREF(np_arg);
    Py_DECREF(pReturn);
    return 1;
}

int PythonEmbeded::setupPython()
{
    if(isInit)
        return 0;
    
    //Py_SetProgramName(program); /* optional but recommended */
    Py_Initialize();        //initialize python interpreter
    init_numpy();
    
    // Add CWD to python module search path
    sys = PyImport_ImportModule("sys");
    path = PyObject_GetAttrString(sys, "path");
    
    // Append does not steal it's reference
    PyList_Append(path, PyString_FromString("../python/"));

    // Import pythonToEmbed
    pName = PyString_FromString("EmbeddedCNN");
    pModule = PyImport_Import(pName);
    
    //When an object’s reference count becomes zero, the object is deallocated.
    // Otherwise it will remain creating memory leaks    
    Py_DECREF(pName);                       // name is no longer needed: free it
    Py_DECREF(path);
    Py_DECREF(sys);                         // is sys borrowed?
    if (!pModule){
        PyErr_Print();
        std::cout << "Python module can not be imported" << std::endl;        
        return 0;
    }
    
    //"pythonToEmbed.plotStdVectors"
    pFunc = PyObject_GetAttrString(pModule, "RunPyCNN");
    if (!pFunc || !PyCallable_Check(pFunc)){
        Py_DECREF(pModule);
        Py_XDECREF(pFunc);
        PyErr_Print();
        std::cout << "Python module attribut could not be retrieved" << std::endl;
        return 0;
    }
    
    isInit = true;
    return 1;
}
