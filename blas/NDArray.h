#ifndef NDARRAY_H
#define NDARRAY_H

#include <shape.h>

template <typename T> class NDArray 
{ 
    private:
        bool allocated = false;

    public:
        T *buffer;                      // pointer on flattened data array in memory
        int *shapeInfo;                 // pointer on array containing shape information about data array

    // default constructor
        NDArray(T *buffer_ = nullptr, int* shapeInfo_ = nullptr);

        /**
         * This method creates new 2D NDArray
         * @param rows
         * @param columns
         * @param order
         */
        NDArray(int rows, int columns, char order);


    char ordering() {
        return shape::order(shapeInfo);
    }

    int *shapeOf() {
        return shape::shapeOf(shapeInfo);
    }

    int *stridesOf() {
        return shape::stride(shapeInfo);
    }

    int rankOf() {
        return shape::rank(shapeInfo);
    }

    int sizeOfT() {
        return sizeof(T);
    }

    NDArray<T>* dup(char newOrder);


    // default destructor
    ~NDArray() {
        if (allocated) {
            delete[] buffer;
            delete[] shapeInfo;
        }
    }

};

#endif#