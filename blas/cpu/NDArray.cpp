#include <NDArray.h>
#include <NativeOpExcutioner.h>

template <typename T> NDArray<T>::NDArray(T *buffer_, int* shapeInfo_ ) {
    this->buffer = buffer_;
    this->shapeInfo = shapeInfo_;
}

template <typename T> NDArray<T>::NDArray(int rows, int columns, char order){
    allocated = true;

    buffer = new T[rows * columns];
    int *shape = new int[2]{rows, columns};
    shapeInfo = shape::shapeBuffer(2, shape);

    if (order == 'f') {
        shapeInfo[7] = 102;
    } else {
        shapeInfo[7] = 99;
    }

    delete[] shape;
}


/**
     * Return value from linear buffer
     *
     * @param i
     * @return
     */
template <typename T> T NDArray<T>:: getScalar(Nd4jIndex i) {
    // throw something right here
    if (i >= shape::length(this->shapeInfo))
        return (T) 0.0f;

    return this->buffer[i];
}

template <typename T> void NDArray<T>:: putScalar(Nd4jIndex i, T value) {
    // throw something right here
    if (i >= shape::length(this->shapeInfo))
        return;

    this->buffer[i] = value;
}

/**
 * Returns value from 2D matrix by coordinates
 * @param i
 * @param k
 * @return
 */
template <typename T> T NDArray<T>::getScalar(int i, int k) {
    // throw something here
    if (this->rankOf() != 2)
        return 0.0f;

    int coords[2] = {i, k};
    Nd4jIndex xOffset = shape::getOffset(0, this->shapeOf(), this->stridesOf(), coords, this->rankOf());

    return this->buffer[xOffset];

}

template <typename T> void NDArray<T>::putScalar(int i, int k, T value) {
    // throw something here
    if (this->rankOf() != 2)
        return;

    int coords[2] = {i, k};
    Nd4jIndex xOffset = shape::getOffset(0, this->shapeOf(), this->stridesOf(), coords, this->rankOf());

    putScalar(xOffset, value);
}


/**
 * Returns value from 3D tensor by coordinates
 * @param i
 * @param k
 * @param j
 * @return
 */
template <typename T> T NDArray<T>:: getScalar(int i, int k, int j) {
    // throw something here
    if (this->rankOf() != 3)
        return 0.0f;

    int coords[3] = {i, k, j};

    Nd4jIndex xOffset = shape::getOffset(0, this->shapeOf(), this->stridesOf(), coords, this->rankOf());

    return this->buffer[xOffset];
}


template <typename T> void NDArray<T>::putScalar(int i, int k, int j, T value) {
    // throw something here
    if (this->rankOf() != 3)
        return;

    int coords[3] = {i, k, j};

    Nd4jIndex xOffset = shape::getOffset(0, this->shapeOf(), this->stridesOf(), coords, this->rankOf());

    putScalar(xOffset, value);
}

template <typename T> NDArray<T>* NDArray<T>::dup(char newOrder) {
    if (newOrder == ordering()) {
        // if ordering is the same as original one - we'll just go directly for memcpy, and we won't care about anything else
        Nd4jIndex newLength = shape::length(this->shapeInfo);
        T * newBuffer = new T[newLength];
        memcpy(newBuffer, this->buffer, newLength * this->sizeOfT());

        int shapeLen = this->rankOf() * 2 + 4;
        int *newShapeInfo = new int[shapeLen];
        memcpy(newShapeInfo, this->shapeInfo, shapeLen * sizeof(int));

        NDArray<T> *result = new NDArray<T>(newBuffer, newShapeInfo);
        // this value should be set, to avoid memleak
        result->allocated = true;
        return result;
    } else {
        // op
        Nd4jIndex newLength = shape::length(this->shapeInfo);
        T * newBuffer = new T[newLength];
        int *newShapeInfo;
        int *shape = this->shapeOf();

        if (newOrder == 'f')
            newShapeInfo = shape::shapeBufferFortran(this->rankOf(), this->shapeOf());
        else
            newShapeInfo = shape::shapeBuffer(this->rankOf(), this->shapeOf());

        // FIXME: we know that EWS is always 1 after dup() result
        newShapeInfo[rankOf() * 2 + 2] = 1;

        // now we construct shape info for new order

        // now we invoke dup pwt against target buffer
        NativeOpExcutioner<T>::execPairwiseTransform(1, newBuffer, newShapeInfo, this->buffer, this->shapeInfo, newBuffer, newShapeInfo, nullptr);

        NDArray<T> *result = new NDArray<T>(newBuffer, newShapeInfo);
        // this value should be set, to avoid memleak
        result->allocated = true;
        return result;
    }
}