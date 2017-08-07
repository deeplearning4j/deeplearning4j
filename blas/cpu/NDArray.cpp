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

template <typename T> NDArray<T>::NDArray(char order, std::initializer_list<int> shape) {
    int rank = (int) shape.size();

    if (rank > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    int *shapeOf = new int[rank];
    int cnt = 0;

    for (auto v: shape)
        shapeOf[cnt++] = v;

    if (order == 'f') {
        shapeInfo = shape::shapeBufferFortran(rank, shapeOf);
    } else {
        shapeInfo = shape::shapeBuffer(rank, shapeOf);
    }

    this->allocated = true;
    this->buffer = new T[shape::length(shapeInfo)];
    memset(this->buffer, 0, sizeOfT() * shape::length(shapeInfo));

    delete[] shapeOf;
}

template <typename T>
void NDArray<T>::replacePointers(T* _buffer, int* _shapeInfo, bool releaseExisting) {
    if (this->allocated && releaseExisting) {
        printf("Deleting original memory\n");
        delete[] this->buffer;
        delete[] this->shapeInfo;
    }

    this->allocated = false;
    this->buffer = _buffer;
    this->shapeInfo = _shapeInfo;
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
template <typename T> NDArray<T>::~NDArray() {
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

template <typename T> T NDArray<T>::sumNumber() {
    return NativeOpExcutioner<T>::execReduceScalar(1, this->buffer, this->shapeInfo, nullptr);
}


template <typename T> T NDArray<T>::meanNumber() {
    return NativeOpExcutioner<T>::execReduceScalar(0, this->buffer, this->shapeInfo, nullptr);
}

template <typename T> void NDArray<T>::assign(T value) {

    // just fire scalar
    NativeOpExcutioner<T>::execScalar(13, this->buffer, this->shapeInfo, this->buffer, this->shapeInfo, value, nullptr);
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

template <typename T>
bool NDArray<T>::nonNull() {
    return this->buffer != nullptr && this->shapeInfo != nullptr;
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
NDArray<T>* NDArray<T>::transpose() {
    int *rearrange = new int[this->rankOf()];
    int cnt = 0;
    for (int d = this->rankOf() - 1; d >= 0; d--) {
        rearrange[cnt++] = d;
    }

    int sLen = this->rankOf() * 2 + 4;
    int *newShapeBuffer = new int[sLen];
    memcpy(newShapeBuffer, this->shapeInfo, sizeof(int) * sLen);

    shape::doPermuteShapeBuffer(newShapeBuffer, rearrange);

    // fixme: this is bad
    newShapeBuffer[sLen - 2] = 1;

    T *newBuffer = new T[this->lengthOf()];
    memcpy(newBuffer, this->buffer, this->sizeOfT() * this->lengthOf());

    NDArray<T> *result = new NDArray(newBuffer, newShapeBuffer);
    result->allocated = true;

    delete[] rearrange;

    return result;
}

template <typename T>
void NDArray<T>::transposei() {
    int *rearrange = new int[this->rankOf()];
    int cnt = 0;
    for (int d = this->rankOf() - 1; d >= 0; d--) {
        rearrange[cnt++] = d;
    }

    int *newShapeBuffer;
    int sLen = this->rankOf() * 2 + 4;
    if (!this->allocated) {
        // if we're going for transpose - we'll have to detach this array from original one
        this->allocated = true;

        T *newBuffer = new T[this->lengthOf()];
        memcpy(newBuffer, this->buffer, this->sizeOfT() * this->lengthOf());

        this->buffer = newBuffer;
        newShapeBuffer = new int[sLen];
        memcpy(newShapeBuffer, this->shapeInfo, sizeof(int) * sLen);
    } else {
        newShapeBuffer = this->shapeInfo;
    }

    shape::doPermuteShapeBuffer(newShapeBuffer, rearrange);

    // fixme: this is bad
    newShapeBuffer[sLen - 2] = 1;
}

template <typename T>
void NDArray<T>::addiRowVector(NDArray<T> *row) {
    if (this->rankOf() != 2)
        throw std::invalid_argument("addiRowVector can be called only on Matrix");

    if (!shape::isRowVector(row->shapeInfo))
        throw std::invalid_argument("Argument should be row vector");

    int *dimension = new int[1] {1};

    shape::TAD *tad = new shape::TAD(this->shapeInfo, dimension, 1);
    tad->createTadOnlyShapeInfo();
    tad->createOffsets();

    NativeOpExcutioner<T>::execBroadcast(0, this->buffer, this->shapeInfo, row->buffer, row->shapeInfo, this->buffer, this->shapeInfo, dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets, tad->tadOnlyShapeInfo, tad->tadOffsets);

    delete[] dimension;
    delete tad;
}

template <typename T>
bool NDArray<T>::equalsTo(NDArray<T> *other, T eps) {
    if (this->lengthOf() != other->lengthOf())
        return false;

    if (!shape::equalsSoft(this->shapeInfo, other->shapeInfo))
        return false;

    T *extras = new T[1] {eps};

    // we don't need extraparams for this op
    T val = NativeOpExcutioner<T>::execReduce3Scalar(4, this->buffer, this->shapeInfo, extras, other->buffer, other->shapeInfo);

    delete[] extras;

    if (val > 0)
        return false;

    return true;
}