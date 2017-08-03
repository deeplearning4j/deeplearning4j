#ifndef NDARRAY_H
#define NDARRAY_H

template <typename T> class NDArray 
{ 
    private: 
        T *buffer;                      // pointer on flattened data array in memory 
        int *shapeInfo;                 // pointer on array containing shape information about data array 

    public:
        // default constructor
        NDArray(buffer = nullptr, shapeInfo = nullptr);

};

#endif#