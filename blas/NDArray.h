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

        NDArray(char order, std::initializer_list<int> shape);

        /**
         * creates NEW NDarray with shape matching other array
         * @param other
         */
        NDArray(NDArray<T> *other);


    /**
     * This method replaces existing buffer/shapeinfo, AND releases original pointers (if releaseExisting TRUE)
     * @param _buffer
     * @param _shapeInfo
     */
    void replacePointers(T* _buffer, int* _shapeInfo, bool releaseExisting);

    /**
     * This method replaces existing buffer/shapeinfo, AND releases original pointers
     *
     * @param _buffer
     * @param _shapeInfo
     */
    void replacePointers(T* _buffer, int* _shapeInfo) {
        replacePointers(_buffer, _shapeInfo, true);
    }

    /**
     * This method returns order of this NDArray
     * @return
     */
    char ordering() {
        return shape::order(shapeInfo);
    }

    /**
     * This method returns shape portion of shapeInfo
     * @return
     */
    int *shapeOf() {
        return shape::shapeOf(shapeInfo);
    }

    /**
     * This method returns strides portion of shapeInfo
     * @return
     */
    int *stridesOf() {
        return shape::stride(shapeInfo);
    }

    /**
     * This method returns rank of this NDArray
     * @return
     */
    int rankOf() {
        return shape::rank(shapeInfo);
    }

    /**
     * This method returns length of this NDArray
     * @return
     */
    Nd4jIndex lengthOf() {
        return shape::length(shapeInfo);
    }

    /**
     * This method returns number of rows in this NDArray
     *
     * PLEASE NOTE: Applicable only to 2D NDArrays
     * @return
     */
    int rows() {
        return shapeOf()[0];
    }

    /**
     * This method returns number of columns in this NDArray
     *
     * PLEASE NOTE: Applicable only to 2D NDArrays
     * @return
     */
    int columns() {
        return shapeOf()[1];
    }

    /**
     * This method retuns sizeof(T) for this NDArray
     * @return
     */
    int sizeOfT() {
        return sizeof(T);
    }

    /**
     * This method returns new copy of this NDArray, optionally in different order
     *
     * @param newOrder
     * @return
     */
    NDArray<T>* dup(char newOrder);

    /**
     * This method assigns values of given NDArray to this one, wrt order
     *
     * @param other
     */
    void assign(NDArray<T> *other);

    /**
     * This method assigns given value to all elements in this NDArray
     *
     * @param value
     */
    void assign(T value);

    /**
     * This method returns sum of all elements of this NDArray
     * @return
     */
    T sumNumber();

    /**
     * This method returns mean number of this NDArray
     *
     * @return
     */
    T meanNumber();


    /**
     * This method applies transpose to this NDArray and returns new instance of NDArray
     *
     */
    NDArray<T> *transpose();

    /**
     * This method returns true if buffer && shapeInfo were defined
     * @return
     */
    bool nonNull();

    /**
     * This method applies inplace transpose to this NDArray
     *
     */
    void transposei();

    /**
     * This method returns true if two arrays are equal, with custom Eps value, false otherwise
     *
     * @param other
     * @param eps
     * @return
     */
    bool equalsTo(NDArray<T> *other, T eps);

    /**
     * This method returns true if two arrays are equal, with default Eps value of 1e-5, false otherwise
     *
     * @param other
     * @param eps
     * @return
     */
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

    /**
     * This method adds given row to all rows in this NDArray
     *
     * @param row
     */
    void addiRowVector(NDArray<T> *row);


    // default destructor
    ~NDArray();

};

#endif