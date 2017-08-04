#include <NDArray.h>
#include <NativeOpExcutioner.h>
#include <stdexcept>

template <typename T> NDArray<T>::NDArray(T *buffer_, int* shapeInfo_ ) {
    this->buffer = buffer_;
    this->shapeInfo = shapeInfo_;
}

template <typename T> NDArray<T>::NDArray(int rows, int columns, char order){
    allocated = true;

    buffer = new T[rows * columns];
    int *shape = new int[2]{rows, columns};


    if (order == 'f') {
        shapeInfo = shape::shapeBufferFortran(2, shape);
        shapeInfo[7] = 102;
    } else {
        shapeInfo = shape::shapeBuffer(2, shape);
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
        throw std::invalid_argument("Requested index above limit");

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
        throw std::invalid_argument("Requested index above limit");

    int coords[2] = {i, k};
    Nd4jIndex xOffset = shape::getOffset(0, this->shapeOf(), this->stridesOf(), coords, this->rankOf());

    return this->buffer[xOffset];

}

template <typename T> void NDArray<T>::putScalar(int i, int k, T value) {
    // throw something here
    if (this->rankOf() != 2)
        throw std::invalid_argument("Requested index above limit");

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
        throw std::invalid_argument("Requested index above limit");

    int coords[3] = {i, k, j};

    Nd4jIndex xOffset = shape::getOffset(0, this->shapeOf(), this->stridesOf(), coords, this->rankOf());

    return this->buffer[xOffset];
}


template <typename T> void NDArray<T>::putScalar(int i, int k, int j, T value) {
    // throw something here
    if (this->rankOf() != 3)
        throw std::invalid_argument("Requested index above limit");

    int coords[3] = {i, k, j};

    Nd4jIndex xOffset = shape::getOffset(0, this->shapeOf(), this->stridesOf(), coords, this->rankOf());

    putScalar(xOffset, value);
}

template <typename T>
void NDArray<T>::assign(NDArray<T> *other) {
    if (other->lengthOf() != this->lengthOf())
        throw std::invalid_argument("Lengths of arrays are mismatching");

    if (this->ordering() == other->ordering()) {

        memcpy(this->buffer, other->buffer, this->lengthOf() * this->sizeOfT());
    } else {
        // now we invoke dup pwt against target buffer
        NativeOpExcutioner<T>::execPairwiseTransform(1, this->buffer, this->shapeInfo, other->buffer, other->shapeInfo, this->buffer, this->shapeInfo, nullptr);
    }
}

template <typename T> NDArray<T>* NDArray<T>::dup(char newOrder) {
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

        NDArray<T> *result = new NDArray<T>(newBuffer, newShapeInfo);
        // this value should be set, to avoid memleak
        result->allocated = true;

        result->assign(this);

        return result;
}