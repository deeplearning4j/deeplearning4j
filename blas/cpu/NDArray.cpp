#include <NDArray.h>
#include <NativeOpExcutioner.h>
#include <stdexcept>

template <typename T> NDArray<T>::NDArray(T *buffer_, int *shapeInfo_ ) {
    buffer    = buffer_;
    shapeInfo = shapeInfo_;
}



template <typename T> NDArray<T>::NDArray(int rows, int columns, char order){
    allocated = true;

    buffer = new T[rows * columns];
    memset(buffer, 0, rows * columns * sizeOfT());

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

template <typename T> NDArray<T>::NDArray(int length, char order) {
    allocated = true;

    buffer = new T[length];
    memset(buffer, 0, length * sizeOfT());

    int *shape = new int[2]{1, length};


    if (order == 'f') {
        shapeInfo = shape::shapeBufferFortran(2, shape);
        shapeInfo[7] = 102;
    } else {
        shapeInfo = shape::shapeBuffer(2, shape);
        shapeInfo[7] = 99;
    }

    delete[] shape;
}

// default destructor
template <typename T> ~NDArray<T>::NDArray() {
    if (allocated) {
        delete[] buffer;
        delete[] shapeInfo;
    }
}

// Return value from linear buffer
template <typename T> T NDArray<T>::getScalar(Nd4jIndex i) {
    // throw something right here
    if (i >= shape::length(shapeInfo))
        throw std::invalid_argument("Requested index above limit");

    return buffer[i];
}

template <typename T> void NDArray<T>:: putScalar(Nd4jIndex i, T value) {
    // throw something right here
    if (i >= shape::length(shapeInfo))
        return;

    buffer[i] = value;
}

// Returns value from 2D matrix by coordinates 
template <typename T> T NDArray<T>::getScalar(int i, int k) {
    // throw something here
    if (rankOf() != 2)
        throw std::invalid_argument("Requested index above limit");

    int coords[2] = {i, k};
    Nd4jIndex xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());

    return buffer[xOffset];

}

template <typename T> void NDArray<T>::putScalar(int i, int k, T value) {
    // throw something here
    if (rankOf() != 2)
        throw std::invalid_argument("Requested index above limit");

    int coords[2] = {i, k};
    Nd4jIndex xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
    putScalar(xOffset, value);
}


// Returns value from 3D tensor by coordinates
template <typename T> T NDArray<T>:: getScalar(int i, int k, int j) {
    // throw something here
    if (rankOf() != 3)
        throw std::invalid_argument("Requested index above limit");

    int coords[3] = {i, k, j};

    Nd4jIndex xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());

    return buffer[xOffset];
}


template <typename T> void NDArray<T>::putScalar(int i, int k, int j, T value) {
    // throw something here
    if (rankOf() != 3)
        throw std::invalid_argument("Requested index above limit");

    int coords[3] = {i, k, j};

    Nd4jIndex xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());

    putScalar(xOffset, value);
}


template <typename T> void NDArray<T>::assign(NDArray<T> *other) {
    if (other->lengthOf() != lengthOf())
        throw std::invalid_argument("Lengths of arrays are mismatched");

    if (ordering() == other->ordering()) {

        memcpy(buffer, other->buffer, lengthOf() * sizeOfT());
    } else {
        // now we invoke dup pwt against target buffer
        NativeOpExcutioner<T>::execPairwiseTransform(1, buffer, shapeInfo, other->buffer, other->shapeInfo, buffer, shapeInfo, nullptr);
    }
}

template <typename T> NDArray<T>* NDArray<T>::dup(char newOrder) {
        // op
        Nd4jIndex newLength = shape::length(shapeInfo);
        T * newBuffer = new T[newLength];
        int *newShapeInfo;
        int *shape = shapeOf();

        if (newOrder == 'f')
            newShapeInfo = shape::shapeBufferFortran(rankOf(), shapeOf());
        else
            newShapeInfo = shape::shapeBuffer(rankOf(), shapeOf());

        // FIXME: we know that EWS is always 1 after dup() result
        newShapeInfo[rankOf() * 2 + 2] = 1;

        NDArray<T> *result = new NDArray<T>(newBuffer, newShapeInfo);
        // this value should be set, to avoid memleak
        result->allocated = true;

        result->assign(this);

        return result;
}

template <typename T>
bool NDArray<T>::equalsTo(NDArray<T> *other, T eps) {
    if (this->lengthOf() != other->lengthOf())
        return false;

    if (!shape::equalsSoft(this->shapeInfo, other->shapeInfo))
        return false;

    // we don't need extraparams for this op
    T val = NativeOpExcutioner<T>::execReduce3Scalar(4, this->buffer, this->shapeInfo, nullptr, other->buffer, other->shapeInfo);

    if (val > 0)
        return false;

    return true;
}