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
        NDArray(T *buffer_ = nullptr, int *shapeInfo_ = nullptr);

        // This constructor creates new 2D NDArray
        NDArray(int rows, int columns, char order);

        NDArray(int length, char order);


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

    Nd4jIndex lengthOf() {
        return shape::length(shapeInfo);
    }

    int rows() {
        return shapeOf()[0];
    }

    int columns() {
        return shapeOf()[1];
    }

    int sizeOfT() {
        return sizeof(T);
    }

    NDArray<T>* dup(char newOrder);

    void assign(NDArray<T> *other);

    /**
     * This method returns true if two arrays are equal, with custom Eps value, false otherwise
     *
     * @param other
     * @param eps
     * @return
     */
    bool equalsTo(NDArray<T> *other, T eps);

    bool equalsTo(NDArray<T> *other) {
        return equalsTo(other, (T) 1e-5f);
    }

    /**
     * Return value from linear buffer
     *
     * @param i
     * @return
     */
    T getScalar(Nd4jIndex i);

    /**
     * Returns value from 2D matrix by coordinates
     * @param i
     * @param k
     * @return
     */
    T getScalar(int i, int k);

    /**
     * Returns value from 3D tensor by coordinates
     * @param i
     * @param k
     * @param j
     * @return
     */
    T getScalar(int i, int k, int j);

    /**
     * This method sets value in linear buffer to position i
     * @param i
     */
    void putScalar(Nd4jIndex i, T value);

    /**
     * This method sets value in 2D matrix to position i,k
     * @param i
     */
    void putScalar(int i, int k, T value);

    /**
     * This method sets value in 3D matrix to position i,k,j
     * @param i
     */
    void putScalar(int i, int k, int j, T value);


    // default destructor
    ~NDArray();

};

#endif