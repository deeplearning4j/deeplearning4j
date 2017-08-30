#ifndef NDARRAY_CPP
#define NDARRAY_CPP

#include "../NDArray.h"
#include <stdexcept>
#include <memory>


namespace nd4j {

// default constructor, do not allocate memory, memory for array is passed from outside 
    template<typename T>
    NDArray<T>::NDArray(T *buffer, int *shapeInfo) {

        _buffer = buffer;
        _shapeInfo = shapeInfo;
        _allocated = false;                                  // indicate that memory for array is passed from outside
    }


// this constructor creates 2D NDArray, memory for array is allocated in this constructor 
    template<typename T>
    NDArray<T>::NDArray(const int rows, const int columns, const char order) {

        _buffer = new T[rows * columns];
        memset(_buffer, 0, rows * columns * sizeOfT());              // set all elements in new array to be zeros

        int *shape = new int[2]{rows, columns};

        if (order == 'f') {
            _shapeInfo = shape::shapeBufferFortran(2, shape);
            _shapeInfo[7] = 102;
        } else {
            _shapeInfo = shape::shapeBuffer(2, shape);
            _shapeInfo[7] = 99;
        }

        _shapeInfo[6] = 1;
        _allocated = true;

        delete[] shape;
    }


// this constructor creates NDArray as single row (dimension is 1xlength), memory for array is allocated in constructor 
    template<typename T>
    NDArray<T>::NDArray(const int length, const char order) {

        _buffer = new T[length];
        memset(_buffer, 0, length * sizeOfT());

        int *shape = new int[2]{1, length};

        if (order == 'f') {
            _shapeInfo = shape::shapeBufferFortran(2, shape);
            _shapeInfo[7] = 102;
        } else {
            _shapeInfo = shape::shapeBuffer(2, shape);
            _shapeInfo[7] = 99;
        }

        _allocated = true;
        delete[] shape;
    }

    template<typename T>
    std::vector<T> NDArray<T>::getBufferAsVector() {
        std::vector<T> vector;

#pragma omp simd
        for (int e = 0; e < this->lengthOf(); e++) {
            vector.push_back(this->getScalar(e));
        }

        return vector;
    }

    template<typename T>
    std::vector<int32_t> NDArray<T>::getShapeAsVector() {
        std::vector<int32_t> vector;

        int magicNumber = this->rankOf() * 2 + 4;
#pragma omp simd
        for (int e = 0; e < magicNumber; e++) {
            vector.push_back(this->_shapeInfo[e]);
        }

        return vector;
    }

// creates new NDArray with shape matching "other" array, do not copy "other" elements into new array
    template<typename T>
    NDArray<T>::NDArray(const NDArray<T> *other) {

        _buffer = new T[other->lengthOf()];
        memset(_buffer, 0, other->lengthOf() * sizeOfT());          // set all elements in new array to be zeros

        _shapeInfo = new int[other->rankOf() * 2 + 4];
        memcpy(_shapeInfo, other->_shapeInfo,
               (other->rankOf() * 2 + 4) * sizeof(int));     // copy shape information into new array
        _allocated = true;
    }


// this constructor creates new array using rank information contained in initializer_list argument
    template<typename T>
    NDArray<T>::NDArray(const char order, const std::initializer_list<int> &shape) {

        int rank = (int) shape.size();

        if (rank > MAX_RANK)
            throw std::invalid_argument("Rank of NDArray can't exceed 32");

        std::unique_ptr<int> shapeOf(new int[rank]);
        int cnt = 0;

        for (auto &item: shape)
            shapeOf.get()[cnt++] = item;

        if (order == 'f') {
            _shapeInfo = shape::shapeBufferFortran(rank, shapeOf.get());
        } else {
            _shapeInfo = shape::shapeBuffer(rank, shapeOf.get());
        }

        _buffer = new T[shape::length(_shapeInfo)];
        memset(_buffer, 0, sizeOfT() * shape::length(_shapeInfo));
        _allocated = true;
    }


    template<typename T>
    NDArray<T>::NDArray(const char order, const std::vector<int> &shape) {

        int rank = (int) shape.size();

        if (rank > MAX_RANK)
            throw std::invalid_argument("Rank of NDArray can't exceed 32");

        std::unique_ptr<int> shapeOf(new int[rank]);
        int cnt = 0;

        for (auto &item: shape)
            shapeOf.get()[cnt++] = item;

        if (order == 'f') {
            _shapeInfo = shape::shapeBufferFortran(rank, shapeOf.get());
        } else {
            _shapeInfo = shape::shapeBuffer(rank, shapeOf.get());
        }

        _buffer = new T[shape::length(_shapeInfo)];
        memset(_buffer, 0, sizeOfT() * shape::length(_shapeInfo));
        _allocated = true;
    }


// This method replaces existing buffer/shapeinfo, AND releases original pointers (if releaseExisting TRUE)
    template<typename T>
    void NDArray<T>::replacePointers(T *buffer, int *shapeInfo, const bool releaseExisting) {

        if (_allocated && releaseExisting) {
            printf("Deleting original memory\n");
            delete[] _buffer;
            delete[] _shapeInfo;
        }
        _allocated = false;
        _buffer = buffer;
        _shapeInfo = shapeInfo;
    }


// This method assigns values of given NDArray to this one, wrt order
    template<typename T>
    void NDArray<T>::assign(NDArray<T> *other) {

        if (other->lengthOf() != lengthOf())
            throw std::invalid_argument("Lengths of arrays are mismatched");

        if (ordering() == other->ordering()) {

            memcpy(_buffer, other->_buffer, lengthOf() * sizeOfT());
        } else {
            // now we invoke dup pwt against target buffer
            NativeOpExcutioner<T>::execPairwiseTransform(1, _buffer, _shapeInfo, other->_buffer, other->_shapeInfo,
                                                         _buffer, _shapeInfo, nullptr);
        }
    }

// This method assigns given value to all elements in this NDArray
    template<typename T>
    void NDArray<T>::assign(const T value) {

        // just fire scalar
        NativeOpExcutioner<T>::execScalar(13, _buffer, _shapeInfo, _buffer, _shapeInfo, value, nullptr);
    }


// This method returns new copy of this NDArray, optionally in different order
    template<typename T>
    NDArray<T> *NDArray<T>::dup(const char newOrder) {
        // op
        Nd4jIndex newLength = shape::length(_shapeInfo);
        T *newBuffer = new T[newLength];
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
        result->_allocated = true;

        result->assign(this);

        return result;
    }


// This method returns sum of all elements of this NDArray
    template<typename T>
    T NDArray<T>::sumNumber() const {
        return NativeOpExcutioner<T>::execReduceScalar(1, _buffer, _shapeInfo, nullptr);
    }


// This method returns mean number of this NDArray
    template<typename T>
    T NDArray<T>::meanNumber() const {
        return NativeOpExcutioner<T>::execReduceScalar(0, _buffer, _shapeInfo, nullptr);
    }


// method calculates sum along dimension(s) in this array and save it to row: as new NDArray with dimensions 1xN
    template<typename T>
    NDArray<T> *NDArray<T>::sum(const std::initializer_list<int> &dimensions) const {
        return reduceAlongDimension<simdOps::Sum<T>>(dimensions);
//    NativeOpExcutioner<T>::execReduce(1, _buffer, _shapeInfo, nullptr, result->_buffer, result->_shapeInfo, dims, dimensions.size(), tad->tadOnlyShapeInfo, tad->tadOffsets);

    }


// eventually this method reduces this array to 1xN row 
    template<typename T>
    template<typename OpName>
    NDArray<T> *NDArray<T>::reduceAlongDimension(const std::initializer_list<int> &dimensions) const {

        int *dims = new int[dimensions.size()];
        int cnt = 0;
        for (auto &d : dimensions)
            dims[cnt++] = d;

        // FIXME: we need inplace sort on dims here (!!!)
        shape::TAD *tad = new shape::TAD(_shapeInfo, dims, dimensions.size());
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        auto *result = new NDArray<T>(1, tad->numTads, 'c');

        functions::reduce::ReduceFunction<T>::template exec<OpName>(_buffer, _shapeInfo, nullptr, result->_buffer,
                                                                    result->_shapeInfo, dims, dimensions.size(),
                                                                    tad->tadOnlyShapeInfo, tad->tadOffsets);

        delete tad;
        delete dims;

        return result;
    }

//
    template<typename T>
    template<typename OpName>
    T NDArray<T>::reduceNumber(T *extraParams) {
        return functions::reduce::ReduceFunction<T>::template execScalar<OpName>(_buffer, _shapeInfo, extraParams);
    }

// perform array transformation
    template<typename T>
    template<typename OpName>
    void NDArray<T>::applyTransform(NDArray<T> *target, T *extraParams) {
        functions::transform::Transform<T>::template exec<OpName>(this->_buffer, this->_shapeInfo, target->_buffer,
                                                                  target->_shapeInfo, extraParams, nullptr, nullptr);
    }

// perform array transformation
    template<typename T>
    template<typename OpName>
    void NDArray<T>::applyTransform(T *extraParams) {
        applyTransform<OpName>(this, extraParams);
    }

// perform pairwise transformation
    template<typename T>
    template<typename OpName>
    void NDArray<T>::applyPairwiseTransform(NDArray<T> *other, T *extraParams) {
        applyPairwiseTransform<OpName>(other, this, extraParams);
    }

// perform pairwise transformation
    template<typename T>
    template<typename OpName>
    void NDArray<T>::applyPairwiseTransform(NDArray<T> *other, NDArray<T> *target, T *extraParams) {
        functions::pairwise_transforms::PairWiseTransform<T>::template exec<OpName>(this->_buffer, this->_shapeInfo,
                                                                                    other->_buffer, other->_shapeInfo,
                                                                                    target->_buffer, target->_shapeInfo,
                                                                                    extraParams);
    }


// method makes copy of this array and applies to the copy the transpose operation, that is this array remains unaffected 
    template<typename T>
    NDArray<T> *NDArray<T>::transpose() const {
        int *rearrange = new int[rankOf()];
        int cnt = 0;
        for (int d = rankOf() - 1; d >= 0; d--) {
            rearrange[cnt++] = d;
        }

        int sLen = rankOf() * 2 + 4;
        int *newShapeBuffer = new int[sLen];
        memcpy(newShapeBuffer, _shapeInfo, sizeof(int) * sLen);

        shape::doPermuteShapeBuffer(newShapeBuffer, rearrange);

        // fixme: this is bad
        newShapeBuffer[sLen - 2] = 1;

        T *newBuffer = new T[lengthOf()];
        memcpy(newBuffer, _buffer, sizeOfT() * lengthOf());

        NDArray<T> *result = new NDArray(newBuffer, newShapeBuffer);
        result->_allocated = true;

        delete[] rearrange;

        return result;
    }


// This method applies in-place transpose to this array, so this array becomes transposed 
    template<typename T>
    void NDArray<T>::transposei() {

        int *rearrange = new int[rankOf()];
        int cnt = 0;
        for (int d = rankOf() - 1; d >= 0; d--) {
            rearrange[cnt++] = d;
        }

        int *newShapeBuffer;
        int sLen = rankOf() * 2 + 4;
        if (!_allocated) {
            // if we're going for transpose - we'll have to detach this array from original one
            _allocated = true;

            T *newBuffer = new T[lengthOf()];
            memcpy(newBuffer, _buffer, sizeOfT() * lengthOf());

            _buffer = newBuffer;
            newShapeBuffer = new int[sLen];
            memcpy(newShapeBuffer, _shapeInfo, sizeof(int) * sLen);
        } else {
            newShapeBuffer = _shapeInfo;
        }

        shape::doPermuteShapeBuffer(newShapeBuffer, rearrange);

        // fixme: this is bad
        newShapeBuffer[sLen - 2] = 1;
        _shapeInfo = newShapeBuffer;
    }


// This method returns true if two arrays are equal, with custom or default Eps value of 1e-5, false otherwise
    template<typename T>
    bool NDArray<T>::equalsTo(const NDArray<T> *other, T eps) const {

        if (lengthOf() != other->lengthOf())
            return false;

        if (!shape::equalsSoft(_shapeInfo, other->_shapeInfo))
            return false;

        T *extras = new T[1]{eps};

        // we don't need extraparams for this op
        T val = NativeOpExcutioner<T>::execReduce3Scalar(4, _buffer, _shapeInfo, extras, other->_buffer,
                                                         other->_shapeInfo);

        delete[] extras;

        if (val > 0)
            return false;

        return true;
    }


// Return value from linear buffer
    template<typename T>
    T NDArray<T>::getScalar(const Nd4jIndex i) const {

        // throw something right here
        if (i >= shape::length(_shapeInfo))
            throw std::invalid_argument("Requested index above limit");

        return _buffer[i];
    }


// Returns value from 2D matrix by coordinates/indexes 
    template<typename T>
    T NDArray<T>::getScalar(const int i, const int j) const {
        // throw something here
        if (rankOf() != 2)
            throw std::invalid_argument("Requested index above limit");

        int coords[2] = {i, j};
        Nd4jIndex xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());

        return _buffer[xOffset];
    }


// Returns value from 3D tensor by coordinates
    template<typename T>
    T NDArray<T>::getScalar(const int i, const int j, const int k) const {
        // throw something here
        if (rankOf() != 3)
            throw std::invalid_argument("Requested index above limit");

        int coords[3] = {i, j, k};

        Nd4jIndex xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());

        return _buffer[xOffset];
    }


// This method sets value in linear buffer to position i
    template<typename T>
    void NDArray<T>::putScalar(const Nd4jIndex i, const T value) {
        // throw something right here
        if (i >= shape::length(_shapeInfo))
            return;

        _buffer[i] = value;
    }


// This method sets value in 2D matrix to position i, j 
    template<typename T>
    void NDArray<T>::putScalar(const int i, const int j, const T value) {
        // throw something here
        if (rankOf() != 2)
            throw std::invalid_argument("Requested index above limit");

        int coords[2] = {i, j};
        Nd4jIndex xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        putScalar(xOffset, value);
    }


// This method sets value in 3D matrix to position i,j,k
    template<typename T>
    void NDArray<T>::putScalar(const int i, const int j, const int k, const T value) {
        // throw something here
        if (rankOf() != 3)
            throw std::invalid_argument("Requested index above limit");

        int coords[3] = {i, j, k};

        Nd4jIndex xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());

        putScalar(xOffset, value);
    }


// This method adds given row to all rows in this NDArray, that is this array becomes affected
    template<typename T>
    void NDArray<T>::addiRowVector(const NDArray<T> *row) {
        if (rankOf() != 2)
            throw std::invalid_argument("addiRowVector can be called only on Matrix");

        if (!shape::isRowVector(row->_shapeInfo))
            throw std::invalid_argument("Argument should be row vector");

        int *dimension = new int[1]{1};

        shape::TAD *tad = new shape::TAD(_shapeInfo, dimension, 1);
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExcutioner<T>::execBroadcast(0, _buffer, _shapeInfo, row->_buffer, row->_shapeInfo, _buffer, _shapeInfo,
                                             dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets,
                                             tad->tadOnlyShapeInfo, tad->tadOffsets);

        delete[] dimension;
        delete tad;
    }


    template<typename T>
    template<typename OpName>
    void NDArray<T>::applyScalar(T scalar, NDArray<T>* target, T *extraParams) {

        if (target == nullptr)
            functions::scalar::ScalarTransform<T>::template transform(this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, scalar, extraParams);
        else
            functions::scalar::ScalarTransform<T>::template transform(this->_buffer, this->_shapeInfo, target->_buffer, target->_shapeInfo, scalar, extraParams);
    }

    template<typename T>
    template<typename OpName>
    void NDArray<T>::applyScalar(NDArray<T>& scalar, NDArray<T>* target, T *extraParams) {
        if (!scalar.isScalar()) {
            throw "Operand is not a scalar!";
        }

        applyScalar<OpName>(scalar, target, extraParams);
    }


// default destructor
    template<typename T>
    NDArray<T>::~NDArray() {

        if (_allocated) {
            delete[] _buffer;
            delete[] _shapeInfo;
        }
    }
}

#endif

