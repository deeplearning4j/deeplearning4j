#ifndef NDARRAY_CPP
#define NDARRAY_CPP

#include "../NDArray.h"
#include "../NativeOpExcutioner.h"
#include "../NDArrayFactory.h"
#include <memory/Workspace.h>
#include <ops/gemm.h>
#include <pointercast.h>
#include <stdexcept>
#include <memory>
#include <helpers/logger.h>
#include <loops/broadcasting.h>

namespace nd4j {

////////////////////////////////////////////////////////////////////////
// default constructor, do not allocate memory, memory for array is passed from outside 
template <typename T> NDArray<T>::NDArray(T *buffer, int *shapeInfo, nd4j::memory::Workspace* workspace) {
    
    _buffer    = buffer;
    _shapeInfo = shapeInfo;
    _isBuffAlloc = false;                                  // indicate that memory for array is passed from outside
    _isShapeAlloc = false;

    _workspace = workspace;
}

template <typename T> NDArray<T>::NDArray(const Nd4jIndex length, const char order, nd4j::memory::Workspace* workspace) {
    if (length < 1)
        throw "Can't allocate non-positive number of elements";
    _workspace = workspace;

    if (workspace == nullptr) {
        _buffer =  new T[length];
    } else {
        _buffer = (T*) _workspace->allocateBytes(length * sizeOfT());
    }

    // todo make this optional
    memset(_buffer, 0, length * sizeOfT());              // set all elements in new array to be zeros

    std::unique_ptr<int> shape(new int[2] {1, length});

    if (order == 'f') {
        _shapeInfo = shape::shapeBufferFortran(2, shape.get());
        _shapeInfo[7] = 102;
    } else {
        _shapeInfo = shape::shapeBuffer(2, shape.get());
        _shapeInfo[7] = 99;
    }

    _shapeInfo[6] = 1;
    _isBuffAlloc = true;
    _isShapeAlloc = true;
}

////////////////////////////////////////////////////////////////////////
// this constructor creates 2D NDArray, memory for array is allocated in this constructor 
template <typename T> NDArray<T>::NDArray(const int rows, const int columns, const char order, nd4j::memory::Workspace* workspace) {

    Nd4jIndex length = rows * columns;
    int rank = 2;

    std::unique_ptr<int> shapeOf(new int[2] {rows, columns});

    _workspace = workspace;
    if (workspace == nullptr) {
        if (order == 'f')
            _shapeInfo = shape::shapeBufferFortran(rank, shapeOf.get());
        else
            _shapeInfo = shape::shapeBuffer(rank, shapeOf.get());

        _buffer =  new T[shape::length(_shapeInfo)];
    } else {
        std::unique_ptr<int> shapeInfo( order == 'f' ? shape::shapeBufferFortran(rank, shapeOf.get()) : shape::shapeBuffer(rank, shapeOf.get()));

        _shapeInfo = (int*) _workspace->allocateBytes(shape::shapeInfoByteLength(rank));
        memcpy(_shapeInfo, shapeInfo.get(), shape::shapeInfoByteLength(rank));

        _buffer = (T*) _workspace->allocateBytes(shape::length(_shapeInfo) * sizeOfT());
    }

    memset(_buffer, 0, length * sizeOfT());              // set all elements in new array to be zeros

    if (order == 'f') {
        _shapeInfo[7] = 102;
    } else {
        _shapeInfo[7] = 99;
    }

    _shapeInfo[6] = 1;
    _isBuffAlloc = true; 
    _isShapeAlloc = true;
}

////////////////////////////////////////////////////////////////////////
// creates new NDArray using shape information from "shape" array, set all elements in new array to be zeros
template <typename T> NDArray<T>::NDArray(const int* shapeInfo, nd4j::memory::Workspace* workspace) {
   
    int arrLength = shape::length(const_cast<int*>(shapeInfo));
    int shapeLength = shape::rank(const_cast<int*>(shapeInfo))*2 + 4;

    _workspace = workspace;
    if (workspace == nullptr) {
        _buffer =  new T[arrLength];
        _shapeInfo = new int[shapeLength];
    } else {
        _buffer = (T*) _workspace->allocateBytes(arrLength * sizeOfT());
        _shapeInfo = (int*) _workspace->allocateBytes(shapeLength * 4);
    }

    memset(_buffer, 0, arrLength*sizeOfT());          // set all elements in new array to be zeros

    memcpy(_shapeInfo, shapeInfo, shapeLength*sizeof(int));     // copy shape information into new array

    _isBuffAlloc = true;
    _isShapeAlloc = true;
}

////////////////////////////////////////////////////////////////////////
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

template <typename T>
NDArray<T>::NDArray(const NDArray<T> *other, nd4j::memory::Workspace* workspace) {
    int arrLength = shape::length(other->_shapeInfo);
    int shapeLength = shape::rank(other->_shapeInfo)*2 + 4;

    _workspace = workspace;
    if (workspace == nullptr) {
        _buffer =  new T[arrLength];
        _shapeInfo = new int[shapeLength];
    } else {
        _buffer = (T*) _workspace->allocateBytes(arrLength * sizeOfT());
        _shapeInfo = (int*) _workspace->allocateBytes(shapeLength * 4);
    }

    memcpy(_buffer, other->_buffer, arrLength*sizeOfT());      // copy other._buffer information into new array

    memcpy(_shapeInfo, other->_shapeInfo, shapeLength*sizeof(int));     // copy shape information into new array

    _isBuffAlloc = true;
    _isShapeAlloc = true;
}

////////////////////////////////////////////////////////////////////////
// copy constructor
template <typename T> NDArray<T>::NDArray(const NDArray<T>& other, nd4j::memory::Workspace* workspace)
{
    int arrLength = shape::length(other._shapeInfo);
    int shapeLength = shape::rank(other._shapeInfo)*2 + 4;

    _workspace = workspace;
    if (workspace == nullptr) {
        _buffer =  new T[arrLength];
        _shapeInfo = new int[shapeLength];
    } else {
        _buffer = (T*) _workspace->allocateBytes(arrLength * sizeOfT());
        _shapeInfo = (int*) _workspace->allocateBytes(shapeLength * 4);
    }

    memcpy(_buffer, other._buffer, arrLength*sizeOfT());      // copy other._buffer information into new array

    memcpy(_shapeInfo, other._shapeInfo, shapeLength*sizeof(int));     // copy shape information into new array
    
    _isBuffAlloc = true; 
    _isShapeAlloc = true;
}

////////////////////////////////////////////////////////////////////////
// this constructor creates new array using rank information contained in initializer_list argument
template <typename T> NDArray<T>::NDArray(const char order, const std::initializer_list<int>& shape, nd4j::memory::Workspace* workspace) {
    
    int rank = (int) shape.size();

    if (rank > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    std::unique_ptr<int> shapeOf(new int[rank]);
    int cnt = 0;

    for (auto& item: shape)
        shapeOf.get()[cnt++] = item;

    _workspace = workspace;
    if (workspace == nullptr) {
        if (order == 'f')
            _shapeInfo = shape::shapeBufferFortran(rank, shapeOf.get());
        else
            _shapeInfo = shape::shapeBuffer(rank, shapeOf.get());

        _buffer =  new T[shape::length(_shapeInfo)];
    } else {
        std::unique_ptr<int> shapeInfo( order == 'f' ? shape::shapeBufferFortran(rank, shapeOf.get()) : shape::shapeBuffer(rank, shapeOf.get()));

        _shapeInfo = (int*) _workspace->allocateBytes(shape::shapeInfoByteLength(rank));
        memcpy(_shapeInfo, shapeInfo.get(), shape::shapeInfoByteLength(rank));

        _buffer = (T*) _workspace->allocateBytes(shape::length(_shapeInfo) * sizeOfT());
    }

    memset(_buffer, 0, sizeOfT() * shape::length(_shapeInfo));
    
    _isBuffAlloc = true; 
    _isShapeAlloc = true;
}

    template<typename T>
    T* NDArray<T>::getBuffer() {
        return _buffer;
    }

    template<typename T>
    int* NDArray<T>::getShapeInfo() {
        return _shapeInfo;
    }

////////////////////////////////////////////////////////////////////////
// assignment operator
template<typename T> NDArray<T>& NDArray<T>::operator=(const NDArray<T>& other) {
	if (this == &other) return *this;

    if (shape::equalsStrict(_shapeInfo, other._shapeInfo))
        memcpy(_buffer, other._buffer, lengthOf()*sizeOfT());
    else {
        if(_isBuffAlloc)
            delete []_buffer;
        if(_isShapeAlloc)
            delete []_shapeInfo;

		int arrLength = other.lengthOf();
		int shapeLength = other.rankOf()*2 + 4;

        _buffer = new T[arrLength];
        memcpy(_buffer, other._buffer, arrLength*sizeOfT());               // copy elements of other current array

        _shapeInfo = new int[shapeLength];
        memcpy(_shapeInfo, other._shapeInfo, shapeLength*sizeof(int));     // copy shape information into new array

        _isBuffAlloc = true;
        _isShapeAlloc = true;
    }

    return *this;
}

template <typename T>
void NDArray<T>::replacePointers(T *buffer, int *shapeInfo, const bool releaseExisting ) {
    this->_buffer = buffer;
    this->_shapeInfo = shapeInfo;

    if (releaseExisting) {
        if (_isShapeAlloc)
            delete[] _shapeInfo;

        if (_isBuffAlloc)
            delete[] _buffer;
    }
}



    template<typename T>
    NDArray<T>::NDArray(const char order, const std::vector<int> &shape, nd4j::memory::Workspace* workspace) {

        int rank = (int) shape.size();

        if (rank > MAX_RANK)
            throw std::invalid_argument("Rank of NDArray can't exceed 32");

        std::unique_ptr<int> shapeOf(new int[rank]);
        int cnt = 0;

        for (auto &item: shape)
            shapeOf.get()[cnt++] = item;

        _workspace = workspace;
        if (workspace == nullptr) {
            if (order == 'f')
                _shapeInfo = shape::shapeBufferFortran(rank, shapeOf.get());
            else
                _shapeInfo = shape::shapeBuffer(rank, shapeOf.get());

            _buffer =  new T[shape::length(_shapeInfo)];
        } else {
            std::unique_ptr<int> shapeInfo( order == 'f' ? shape::shapeBufferFortran(rank, shapeOf.get()) : shape::shapeBuffer(rank, shapeOf.get()));

            _shapeInfo = (int*) _workspace->allocateBytes(shape::shapeInfoByteLength(rank));
            memcpy(_shapeInfo, shapeInfo.get(), shape::shapeInfoByteLength(rank));

            _buffer = (T*) _workspace->allocateBytes(shape::length(_shapeInfo) * sizeOfT());
        }

        memset(_buffer, 0, sizeOfT() * shape::length(_shapeInfo));
        
		_isBuffAlloc = true; 
		_isShapeAlloc = true;
	
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


////////////////////////////////////////////////////////////////////////
// This method returns new copy of this NDArray, optionally in different order
template <typename T> NDArray<T>* NDArray<T>::dup(const char newOrder) {
    // op
    Nd4jIndex newLength = shape::length(_shapeInfo);
    T * newBuffer = new T[newLength];
    int *newShapeInfo;

    if (newOrder == 'f')
        newShapeInfo = shape::shapeBufferFortran(rankOf(), shapeOf());
    else
        newShapeInfo = shape::shapeBuffer(rankOf(), shapeOf());

    // FIXME: we know that EWS is always 1 after dup() result
    newShapeInfo[rankOf() * 2 + 2] = 1;

    NDArray<T> *result = new NDArray<T>(newBuffer, newShapeInfo);
    // this value should be set, to avoid memleak
    result->_isBuffAlloc = true;
    result->_isShapeAlloc = true;

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


    template <typename T>
    Nd4jIndex NDArray<T>::tensorsAlongDimension(std::initializer_list<int> dimensions) {
        std::vector<int> vector(dimensions);
        return tensorsAlongDimension(vector);
    }

    template <typename T>
    Nd4jIndex NDArray<T>::tensorsAlongDimension(std::vector<int>& dimensions) {
        if ((int) dimensions.size() > this->rankOf())
            throw "TAD can't have dimensions higher then original array";

        std::vector<int> copy(dimensions);

        // we need to sort dimensions (?)
        if (dimensions.size() > 1)
            std::sort (copy.begin(), copy.end());

        Nd4jIndex tadLength = shape::tadLength(this->_shapeInfo, copy.data(), copy.size());
        Nd4jIndex numTads = this->lengthOf() / tadLength;

        return numTads;
    }

    template <typename T>
    NDArray<T>* NDArray<T>::tensorAlongDimension(int index, std::initializer_list<int> dimensions) {
        std::vector<int> vector(dimensions);
        return tensorAlongDimension(index, vector);
    }



    template <typename T>
    void NDArray<T>::printBuffer(const char* msg) {
        if (msg != nullptr)
            printf("%s [", msg);
        else
            printf("[");
        for (Nd4jIndex e = 0; e < lengthOf(); e++) {
            printf("%f", this->getScalar(e));
            if (e < lengthOf() - 1)
                printf(", ");
        }
        printf("]\n");
        fflush(stdout);
    }

    template <typename T>
    NDArray<T>* NDArray<T>::tensorAlongDimension(int index, std::vector<int>& dimensions) {
        if ((int) dimensions.size() > this->rankOf())
            throw "TAD can't have dimensions higher then original array";

        std::vector<int> copy(dimensions);

        // we need to sort dimensions (?)
        if (dimensions.size() > 1)
            std::sort (copy.begin(), copy.end());

        Nd4jIndex tadLength = shape::tadLength(this->_shapeInfo, copy.data(), copy.size());
        Nd4jIndex numTads = this->lengthOf() / tadLength;

        if (index >= numTads)
            throw "Can't get index higher than total number of TADs";

        std::unique_ptr<shape::TAD> tad(new shape::TAD(this->_shapeInfo, copy.data(), copy.size()));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        nd4j_verbose("Applying offset [%i] for index [%i]\n", tad->tadOffsets[index], index);

        T* buffer = this->_buffer + tad->tadOffsets[index];
        int* shapeInfo = new int[shape::shapeInfoLength(tad->tadOnlyShapeInfo[0])];
        std::memcpy(shapeInfo, tad->tadOnlyShapeInfo, shape::shapeInfoByteLength(tad->tadOnlyShapeInfo));

        auto array = new NDArray<T>(buffer, shapeInfo);
        array->_isBuffAlloc = false;
        array->_isShapeAlloc = true;
        array->_isView = true;


        return array;
    }

// method makes copy of this array and applies to the copy the transpose operation, that is this array remains unaffected 
template <typename T> NDArray<T>* NDArray<T>::transpose() const {
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
    result->_isBuffAlloc = true;
    result->_isShapeAlloc = true;

    delete[] rearrange;

    return result;
}

////////////////////////////////////////////////////////////////////////
// This method applies in-place transpose to this array, so this array becomes transposed 
template <typename T> void NDArray<T>::transposei() {
    
    int *rearrange = new int[rankOf()];
    int cnt = 0;
    for (int d = rankOf() - 1; d >= 0; d--) {
        rearrange[cnt++] = d;
    }

    int *newShapeBuffer;
    int sLen = rankOf() * 2 + 4;  
    if (!_isBuffAlloc) {
        // if we're going for transpose - we'll have to detach this array from original one
        _isBuffAlloc = true;
        T *newBuffer = new T[lengthOf()];
        memcpy(newBuffer, _buffer, sizeOfT() * lengthOf());
        _buffer = newBuffer;
    }
    else if(!_isShapeAlloc) {
        _isShapeAlloc = true;
        newShapeBuffer = new int[sLen];
        memcpy(newShapeBuffer, _shapeInfo, sizeof(int) * sLen);
    }
    else {
        newShapeBuffer = _shapeInfo;
    }

    shape::doPermuteShapeBuffer(newShapeBuffer, rearrange);

    // fixme: this is bad
    newShapeBuffer[sLen - 2] = 1;
    _shapeInfo = newShapeBuffer;
    delete []rearrange;
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

    template<typename T>
    T NDArray<T>::getIndexedScalar(const Nd4jIndex i)  {
        // throw something right here
        if (i >= shape::length(_shapeInfo))
            throw std::invalid_argument("Requested index above limit");

        int idx[MAX_RANK];
        shape::ind2subC(this->rankOf(), this->shapeOf(), i, idx);

        Nd4jIndex offset = shape::getOffset(0, this->shapeOf(), this->stridesOf(), idx, this->rankOf());

        return _buffer[offset];
    }

    template<typename T>
    void NDArray<T>::putIndexedScalar(const Nd4jIndex i, const T value)  {
        // throw something right here
        if (i >= shape::length(_shapeInfo))
            throw std::invalid_argument("Requested index above limit");

        int idx[MAX_RANK];
        shape::ind2subC(this->rankOf(), this->shapeOf(), i, idx);

        Nd4jIndex offset = shape::getOffset(0, this->shapeOf(), this->stridesOf(), idx, this->rankOf());
        _buffer[offset] = value;
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

//////////////////////////////////////////////////////////////////////////
// accessing operator for 2D matrix, i - row, j - column
// be careful this method doesn't check the rank of array
template<typename T>
T NDArray<T>::operator()(const int i, const int j) const {

    int coords[2] = {i, j};
    Nd4jIndex xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
    return _buffer[xOffset];
}

//////////////////////////////////////////////////////////////////////////
// modifying operator for 2D matrix, i - row, j - column
// be careful this method doesn't check the rank of array
template<typename T>
T& NDArray<T>::operator()(const int i, const int j) {

	int coords[2] = {i, j};
    Nd4jIndex xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
    return _buffer[xOffset];
}


// This method adds given row to all rows in this NDArray, that is this array becomes affected
    template<typename T>
    void NDArray<T>::addiRowVector(const NDArray<T> *row) {
        if (rankOf() != 2)
            throw std::invalid_argument("addiRowVector can be called only on Matrix");

        if (!shape::isRowVector(row->_shapeInfo))
            throw std::invalid_argument("Argument should be row vector");

        int dimension[1] = {1};

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, dimension, 1));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExcutioner<T>::execBroadcast(0, _buffer, _shapeInfo, row->_buffer, row->_shapeInfo, _buffer, _shapeInfo,
                                             dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets,
                                             tad->tadOnlyShapeInfo, tad->tadOffsets);
    }


    template<typename T>
    template<typename OpName>
    void NDArray<T>::applyScalar(T scalar, NDArray<T>* target, T *extraParams) {

        if (target == nullptr)
            functions::scalar::ScalarTransform<T>::template transform<OpName>(this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, scalar, extraParams);
        else
            functions::scalar::ScalarTransform<T>::template transform<OpName>(this->_buffer, this->_shapeInfo, target->_buffer, target->_shapeInfo, scalar, extraParams);
    }

    template<typename T>
    template<typename OpName>
    void NDArray<T>::applyScalar(NDArray<T>& scalar, NDArray<T>* target, T *extraParams) {
        if (!scalar.isScalar()) {
            throw "Operand is not a scalar!";
        }

        applyScalar<OpName>(scalar.getScalar(0), target, extraParams);
    }


//////////////////////////////////////////////////////////////////////////
// calculate strides 
template <typename T> void NDArray<T>::updateStrides(const char order) {
	
	int rank = rankOf();	
	int doubleRank = 2*rank;
	if(order == 'c') {
        _shapeInfo[doubleRank] = 1;          // set unity as last stride for c order
        for(int j=1; j<rank; ++j)
            _shapeInfo[doubleRank-j] = _shapeInfo[doubleRank-j+1]*_shapeInfo[rank+1-j];
    }
    else {
        _shapeInfo[rank+1] = 1;             // set unity as first stride for f order
        for(int j=rank+1; j<doubleRank; ++j)
            _shapeInfo[j+1] = _shapeInfo[j]*_shapeInfo[j-rank];
    }
	// set last 3 elements in _shapeInfo
	_shapeInfo[doubleRank + 1] = 0;                  
    _shapeInfo[doubleRank + 2] = 1;
    _shapeInfo[doubleRank + 3] = (int)order;
}

//////////////////////////////////////////////////////////////////////////
// set new order and shape in case of suitable array length 
template <typename T> bool NDArray<T>::reshape(const char order, const std::initializer_list<int>& shape) {
        std::vector<int> vShape(shape);
        return reshape(order, vShape);
/*
    int rank = shape.size();
    int arrLength = 1;
    for(const auto& item : shape)
        arrLength *= item;

    if(_buffer==nullptr || arrLength != lengthOf())
        return false;

    int shapeLength = rank*2 + 4;
    // remember old values

    int elemWiseStride = _shapeInfo[rankOf()*2 + 2];
    // if rank is different then delete and resize _shapeInfo appropriately
    // also check if current object is _shapeInfo owner
    if(rank != rankOf() || !_isShapeAlloc) {
        if(_isShapeAlloc)
            delete []_shapeInfo;
        _shapeInfo = new int[shapeLength];
        _shapeInfo[0] = rank;
        _isShapeAlloc = true;
    }
    // copy new dimensions to _shapeInfo
    int i = 1;
    for(const auto& item : shape)
        _shapeInfo[i++] = item;                 // exclude first element -> rank
    // set strides in correspondence to dimensions and order
	updateStrides(order);

    return true;
        */
}


//////////////////////////////////////////////////////////////////////////
// set new order and shape in case of suitable array length 
template <typename T> bool NDArray<T>::reshape(const char order, const std::vector<int>& cshape) {

    std::vector<int> shape(cshape);
    int rank = shape.size();

    // looking for negative in shape

    int numberNegativesOnes = 0;

    int* shape_ = shape.data();
    for (int i = 0; i < (int) shape.size(); i++) {
        if (shape[i] < 0) {
            if (numberNegativesOnes >= 1)
                throw "Only one dimension can be negative ones";

            numberNegativesOnes++;

            int shapeLength = 1;
            for (int j = 0; j < (int) shape.size(); j++)
                if (shape_[j] >= 1)
                    shapeLength *= shape_[j];

            int realShape = nd4j::math::nd4j_abs<int>(lengthOf() / shapeLength);
            int* thisNewShape = new int[shape.size()];

            for (int j = 0; j < (int) shape.size(); j++) {
                if (i != j) {
                    thisNewShape[j] = shape_[j];
                } else
                    thisNewShape[j] = realShape;
            }

            shape_ = thisNewShape;
            break;
        }
    }

    for (int e = 0; e < (int) shape.size(); e++) {
        shape[e] = shape_[e];
    }

    if (numberNegativesOnes > 0)
        delete[] shape_;

    int arrLength = 1;
    for(const auto& item : shape)
        arrLength *= item;

    if(_buffer==nullptr || arrLength != this->lengthOf()) {
        this->printShapeInfo("Mismatched shape");
        nd4j_verbose("ArrLength: %i\n", arrLength);
        throw "Bad shape!";
    }

    int shapeLength = shape::shapeInfoLength(rank);
    // remember old values

    //int elemWiseStride = _shapeInfo[rankOf()*2 + 2];
    // if rank is different then delete and resize _shapeInfo appropriately
    // also check if current object is _shapeInfo owner
    if(rank != rankOf() || !_isShapeAlloc) {
        if(_isShapeAlloc)
            delete []_shapeInfo;
        _shapeInfo = new int[shapeLength];
        _shapeInfo[0] = rank;
        _isShapeAlloc = true;
    }
    // copy new dimensions to _shapeInfo
    int i = 1;
    for(const auto& item : shape)
        _shapeInfo[i++] = item;                 // exclude first element -> rank
    // set strides in correspondence to dimensions and order
    updateStrides(order);

    return true;
}

//////////////////////////////////////////////////////////////////////////
// change an array by repeating it the number of times given by reps.
template <typename T> void NDArray<T>::tilei(const std::vector<int>& reps) {
	// check whether reps contains at least one zero (then throw exception) or whether all elements in reps are unities (then simply reshape or do nothing)
	int dim = reps.size();	
	int product = 1;
	for(const auto& item : reps)
		product *= item;
	if(product == 0)
		throw "Tile method: one of the elements in reps array is zero !";
	int rankOld = rankOf();
	int diff = rankOld - dim;
	if(product==1) {	    // in this case 2 possibilities are present: just reshape or nothing to do
		if(diff < 0) {		// reshape to higher dimension			
			std::vector<int> shapeNew = reps;				// need to have unities at first "diff" positions of new shape
			memcpy(&shapeNew[-diff], _shapeInfo+1, rankOld*sizeof(int));   // put old shape numbers at rest of positions
			reshape(ordering(), shapeNew);
		}		
		return;				// nothing to do, if diff >= 0 -> identity tile 
	}	
	
	// evaluate new shapeInfo
	int* newShapeInfo = nullptr;	
	if(diff < 0) {
		newShapeInfo = new int[dim*2 + 4];
		newShapeInfo[0] = dim;					// set new rank
		for(int i=1; i <= -diff; ++i)
			newShapeInfo[i] = 1;				// set unities to be new dimensions at left-hand side of newShapeInfo shape place
		memcpy(newShapeInfo + 1 - diff, _shapeInfo + 1, rankOld*sizeof(int));		// copy old dimensions to the right-hand side of newShapeInfo shape place
		for(int i=1; i <= dim; ++i)
			newShapeInfo[i] *= reps[i - 1];		// set new shape by multiplying old dimensions by corresponding numbers from reps 
	}
	else {
		newShapeInfo = new int[rankOld*2 + 4];
		memcpy(newShapeInfo, _shapeInfo, (rankOld*2 + 4)*sizeof(int));		// copy all elements of _shapeInfo to newShapeInfo
		for(int i=1; i <= dim; ++i)
			newShapeInfo[rankOld + 1 - i] *= reps[dim - i];		// set new shape by multiplying old dimensions by corresponding numbers from reps 
	}
	int rankNew = newShapeInfo[0];
	// create new buffer, in any case the memory amount new buffer points to is bigger then those for old _buffer
	T* newBuff = new T[shape::length(newShapeInfo)];
	char order = ordering();
	int arrLengthNew = shape::length(newShapeInfo);		
	int arrLengthOld = shape::length(_shapeInfo);		
	int stepNew, stepOld, bitStepOld, numCopies;
	if(order == 'c') {
		stepNew = newShapeInfo[rankNew];
		stepOld = _shapeInfo[rankOld];
		bitStepOld = sizeOfT()*stepOld;
		numCopies = reps[dim-1]*stepOld;
	}
	else {
		stepNew = newShapeInfo[1];
		stepOld = _shapeInfo[1];
		bitStepOld = sizeOfT()*stepOld;
		if(diff <= 0)
			numCopies = reps[0]*stepOld;
		else	
			numCopies = stepOld;
	}	
	// fill newBuff, loop through all dimensions of newBuff except elementary dimension
	// looping through _buffer goes automatically by means of "position" index 
	int position = 0;
	for(int i=0;  i<arrLengthNew; i+=stepNew) {		
		for(int j=0; j<numCopies; j+=stepOld)
				memcpy(newBuff + i + j, _buffer + position, bitStepOld);
		position += stepOld;
		if(position == arrLengthOld)		// if loop through _buffer has come to end then start it again from beginning
			position = 0;
	}
	
	// assign new shape to "this" array
    // also check if current object is _shapeInfo and _buffer owner
    if(_isShapeAlloc)       
		delete []_shapeInfo;
	else	
		_isShapeAlloc = true;
	_shapeInfo = newShapeInfo;    
    // assign new buffer to "this" array
    if(_isBuffAlloc)       
		delete []_buffer;
	else
		_isBuffAlloc = true;
	_buffer = newBuff;        

	updateStrides(order);
	
}

    template<typename T>
    int NDArray<T>::sizeAt(int dim) {
        if (dim > this->rankOf())
            throw "Bad size index requested";

        return this->_shapeInfo[1+dim];
    }


//////////////////////////////////////////////////////////////////////////
// change an array by repeating it the number of times given by reps
template<typename T> NDArray<T>* NDArray<T>::repeat(int dimension, const std::vector<int>& repeats) {

    if (dimension < 0)
        dimension += this->rankOf();

    std::vector<int> reps;

    if ((int) reps.size() < this->rankOf()) {
        if (dimension > 0) {
            for (int e = 0; e < this->rankOf() - (int) repeats.size(); e++)
                reps.push_back(1);

            for (auto r: repeats)
                reps.push_back(r);
        } else {
            for (auto r: repeats)
                reps.push_back(r);

            for (int e = 0; e < this->rankOf() - (int) repeats.size(); e++)
                reps.push_back(1);
        }
    }/* else {
        for (auto r: repeats)
            reps.push_back(r);
    }*/

    std::unique_ptr<int> newShape(new int[this->rankOf()]);
    std::vector<int> rShape;

    for (int i = 0; i < this->rankOf(); i++) {
        newShape.get()[i] = this->sizeAt(i) * reps.at(i);
        rShape.push_back(newShape.get()[i]);
    }

    auto ret = new NDArray<T>('c', rShape);

    auto repeatDelta = shape::prodLong(newShape.get(), this->rankOf()) / this->lengthOf();
    auto numTads = this->tensorsAlongDimension({dimension});
    for (int i = 0; i < numTads; i++) {
        auto thisTensor = this->tensorAlongDimension(i, {dimension});
        auto retTensor = ret->tensorAlongDimension(i, {dimension});
        int retIdx = 0;
        for (int k = 0; k < thisTensor->lengthOf(); k++) {
            T s = thisTensor->getIndexedScalar(k);
            for (int j = 0; j < repeatDelta; j++) {
                retTensor->putIndexedScalar(retIdx++, s);
            }
        }

        delete thisTensor;
        delete retTensor;
    }

    return ret;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
bool NDArray<T>::permutei(const int* dimensions, const int rank) {

    if(_buffer==nullptr || rank != rankOf())
        return false;

    // check if current object is _shapeInfo owner
    if(!_isShapeAlloc) {             // if _shapeInfo is not its own
        int* shapeInfoNew = new int[rank*2+4];
		memcpy(shapeInfoNew, _shapeInfo, (rank*2+4)*sizeof(int));	        
		shape::doPermuteShapeBuffer(rank, shapeInfoNew, const_cast<int*>(dimensions));	
        _shapeInfo = shapeInfoNew;
        _isShapeAlloc = true;
    }
    else        
		shape::doPermuteShapeBuffer(rank, _shapeInfo, const_cast<int*>(dimensions));	

    return true;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
bool NDArray<T>::permutei(const std::initializer_list<int>& dimensions) {
    std::vector<int> vec(dimensions);
    return permutei(vec);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
bool NDArray<T>::permutei(const std::vector<int>& dimensions) {
    return permutei(dimensions.data(), dimensions.size());
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T>* NDArray<T>::permute(const int* dimensions, const int rank) {

    if(_buffer==nullptr || rank != rankOf())
        throw "Wrong arguments in permute method: either array is nullptr or rank is not suitable!";
	int buffLength = lengthOf();
	int shapeInfoLength = rankOf()*2 + 4;
	// allocate memory for new array - buffer and shapeInfo
	T* bufferNew = new T[buffLength];
	int* shapeInfoNew = new int[shapeInfoLength];
	// copy this arrays _buffer and _shapeInfo into new array	
	memcpy(bufferNew, _buffer, buffLength*sizeOfT());	
	memcpy(shapeInfoNew, _shapeInfo, shapeInfoLength*sizeof(int));	
	// perform buffer permutation	
	shape::doPermuteShapeBuffer(rank, shapeInfoNew, const_cast<int*>(dimensions));	
	// create array to be returned
    NDArray<T>* ret = new NDArray<T>(bufferNew, shapeInfoNew);	
	// don't forget to indicate that memory for new array was allocated
    ret->_isBuffAlloc = true;
    ret->_isShapeAlloc = true;

    return ret;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T>* NDArray<T>::permute(const std::vector<int>& dimensions) {
    return permute(dimensions.data(), dimensions.size());
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T>* NDArray<T>::permute(const std::initializer_list<int>& dimensions) {
    std::vector<int> vec(dimensions);
    return permute(vec);
}


//////////////////////////////////////////////////////////////////////////
// tile an array by repeating it the number of times given by reps.
template<typename T> NDArray<T>* NDArray<T>::tile(const std::vector<int>& reps) {
	// check whether reps contains at least one zero (then throw exception) or whether all elements in reps are unities (then simply reshape or do nothing)
	if(std::find(reps.begin(), reps.end(), 0) != reps.end())
	    throw "Tile method: one of the elements in reps array is zero !";

    bool wasReshaped = false;
    std::vector<int> origShape(this->_shapeInfo+1, this->_shapeInfo + 1 + this->rankOf());
    char origOrder = this->ordering();

    if (reps.size() > this->rankOf()) {
        wasReshaped = true;
        std::vector<int> newS;
        for (int e = 0; e < (reps.size() - this->rankOf()); e++)
            newS.push_back(1);

        for (auto v: origShape)
            newS.push_back(v);

        this->reshape('c', newS);
    }

    // evaluate new shape
	int dim = reps.size();
	int rank = this->rankOf();
	int diff = rank - dim;
	std::vector<int> shapeNew;
	if(diff < 0) {
	    shapeNew = reps;
	    for(int i=0; i<rank; ++i)
		    shapeNew[dim-1-i] *= this->_shapeInfo[rank-i];
	} else {
	    shapeNew = std::vector<int>(this->_shapeInfo + 1, this->_shapeInfo + 1 + rank);
	    for(int i=1; i<=dim; ++i)
		    shapeNew[rank-i] *= reps[dim-i];
	}

	// create empty array with new shape

	// ret = this->repeat()

    int d = reps.size();
    int *shape = new int[this->rankOf()];
    std::memcpy(shape, this->shapeOf(), this->rankOf() * sizeof(int));

    int l = (int) this->lengthOf();
    int n = nd4j::math::nd4j_max<int>(l, 1);
    std::vector<int> repeats;
    if (d < this->rankOf()) {
        for (int e = 0; e < this->rankOf() - reps.size(); e++)
            repeats.push_back(1);
    }
    for (auto r: reps)
        repeats.push_back(r);

    int* repeat = repeats.data();
    NDArray<T>* result = this;
    for (int i = 0; i < rank; i++) {
        if (repeat[i] != 1) {
            result->reshape('c', {-1, n});
            NDArray<T> *tmp = result->repeat(0, {repeat[i]});

            if (result->_shapeInfo != this->_shapeInfo)
                delete result;

            result = tmp;
        }

        //int in = shape[i];
        //n /= nd4j::math::nd4j_max<int>(in, 1);
        int in = shape[i];
        int nOut = in * repeat[i];
        shape[i] = nOut;
        n /= nd4j::math::nd4j_max<int>(in, 1);
    }
    delete[] shape;

    result->reshape('c', shapeNew);

    if (wasReshaped)
        this->reshape(origOrder, origShape);

    return result;
}


    template<typename T>
    template<typename OpName>
    void NDArray<T>::applyBroadcast(std::initializer_list<int> dimensions, NDArray<T>* tadArray, NDArray<T>* target, T* extraArgs) {
        std::vector<int> vec(dimensions);
        applyBroadcast<OpName>(vec, tadArray, target, extraArgs);
    }


    template<typename T>
    template<typename OpName>
    void NDArray<T>::applyBroadcast(std::vector<int>& dimensions, NDArray<T>* tadArray, NDArray<T>* target, T* extraArgs) {
        if (dimensions.size() == 0)
            return;

        std::vector<int> copy(dimensions);

        if (dimensions.size() > 1)
            std::sort(copy.begin(), copy.end());

        Nd4jIndex tadLength = shape::tadLength(this->_shapeInfo, copy.data(), (int) copy.size());
        if (tadLength != tadArray->lengthOf())
            throw "Tad length mismatch";

        shape::TAD tad(this->_shapeInfo, copy.data(), copy.size());
        tad.createTadOnlyShapeInfo();
        tad.createOffsets();

        NDArray<T>* result = target == nullptr ? this : target;

        // TODO: eventually we want separate tads here
        functions::broadcast::Broadcast<T>::template exec<OpName>(this->_buffer, this->_shapeInfo, tadArray->_buffer, tadArray->_shapeInfo, result->_buffer, result->_shapeInfo, copy.data(), (int) copy.size(), tad.tadOnlyShapeInfo, tad.tadOffsets, tad.tadOnlyShapeInfo, tad.tadOffsets);
    }

//////////////////////////////////////////////////////////////////////////
// return array which is broadcasted from this and argument array  
template<typename T>
NDArray<T>* NDArray<T>::broadcast(const NDArray<T>& other) {	
	// the orders must be the same
	char order = ordering();
	if(order != other.ordering())
		throw "Broadcast method: arrays have different orders!";
	// recognize shapes with smaller and bigger rank
	int* biggerShapeInfo = nullptr;
	int* smallerShapeInfo = nullptr;
	int smallerRank, biggerRank;
	if (rankOf() > other.rankOf()) {
		biggerShapeInfo = _shapeInfo;
		biggerRank = _shapeInfo[0];
		smallerShapeInfo = other._shapeInfo;
		smallerRank = other._shapeInfo[0];
	}
	else {
		biggerShapeInfo = other._shapeInfo;
		biggerRank = other._shapeInfo[0];
		smallerShapeInfo = _shapeInfo;
		smallerRank = _shapeInfo[0];
	}
	// check shapes on consistency	
	int diff = biggerRank - smallerRank;
	for (int i = smallerRank; i<=1; --i)
		if(biggerShapeInfo[diff+i] != smallerShapeInfo[i] && biggerShapeInfo[i] != 1 && smallerShapeInfo[i] != 1)
			throw "Broadcast method: arrays have incompatible shapes !";
	// create and fill ret shapeInfo
	int* shapeInfoNew = new int[biggerRank*2 + 4];
	memcpy(shapeInfoNew, biggerShapeInfo, (biggerRank*2 + 4)*sizeof(int));
	for (int i = smallerRank; i>=1; --i) 
		if(shapeInfoNew[diff+i] == 1 || smallerShapeInfo[i] == 1) 
			shapeInfoNew[diff+i] *= smallerShapeInfo[i];

	NDArray<T>* ret = new NDArray<T>(shapeInfoNew);
	ret->updateStrides(order);
	delete []shapeInfoNew;

	return ret;
}


//////////////////////////////////////////////////////////////////////////
// check whether array's rows (arg=0) or columns create orthogonal basis
template<typename T>
bool NDArray<T>::hasOrthonormalBasis(const int arg) {

	if(rankOf() !=2 )
		throw "hasOrthBasis method: rank of ndarray is not equal 2 !";

	if(arg!=0  && arg!=1)
		throw "hasOrthBasis method: input argument is not equal to 0 or 1 !";

	const T eps = 1e-5f;
	T dot = 0.f;
	if(arg) {					// check whether columns create orthogonal basis
		for(int j=0; j<columns()-1; ++j)
			for(int k=j+1; k<columns(); ++k) {
				for(int i=0; i<rows(); ++i)
					dot += getScalar(i,j)*getScalar(i,k);
				if(nd4j::math::nd4j_abs(dot) > eps )
					return false;
				dot = 0.f;
			}
		for(int j=0; j<columns(); ++j)	{	// check whether norm of column vector = 1
			for(int i=0; i<rows(); ++i)
				dot += getScalar(i,j)*getScalar(i,j);
			if(dot!=0.f && nd4j::math::nd4j_abs(nd4j::math::nd4j_sqrt<T>(dot) - 1.f) > eps)
				return false;
			dot = 0.f;
		}
	}
	else {						// check whether rows create orthogonal basis
		for(int i=0; i<rows()-1; ++i)
			for(int k=i+1; k<rows(); ++k) {
				for(int j=0; j<columns(); ++j)
					dot += getScalar(i,j)*getScalar(k,j);
				if(nd4j::math::nd4j_abs(dot) > eps )
					return false;
				dot = 0.f;
			}
		for(int i=0; i<rows(); ++i) {		// check whether norm of row vector = 1
			for(int j=0; j<columns(); ++j)
					dot += getScalar(i,j)*getScalar(i,j);
			if(dot!=0.f && nd4j::math::nd4j_abs(nd4j::math::nd4j_sqrt<T>(dot) - 1.f) > eps)
				return false;
			dot = 0.f;
		}
	}
	return true;
}

//////////////////////////////////////////////////////////////////////////
// check whether array is identity matrix
template<typename T>
bool NDArray<T>::isIdentityMatrix() {
	if(rankOf() !=2 || rows() != columns())
		throw "isIdentityMatrix method: matrix must be square and have rank = 2 !";

	const T eps = 1e-5f;
	for(int i=0; i<rows(); ++i)
		if(nd4j::math::nd4j_abs(getScalar(i,i) - 1.f) > eps)
			return false;

	for(int i=0; i<rows(); ++i)
		for(int j=0; j!=i && j<columns(); ++j)
			if(nd4j::math::nd4j_abs(getScalar(i,j)) > eps)
				return false;
	return true;
}

//////////////////////////////////////////////////////////////////////////
// check whether array is unitary matrix
template<typename T>
bool NDArray<T>::isUnitary() {

	if(rankOf() !=2 || rows() != columns())
		throw "isUnitary method: matrix must be square and have rank = 2 !";

	NDArray<T> tr = *(this->transpose());
	tr = *nd4j::NDArrayFactory::mmulHelper<T>(this, &tr, &tr, 1.f, 0.f);

	return tr.isIdentityMatrix();
}

//////////////////////////////////////////////////////////////////////////
// Singular value decomposition program from "Numerical Recipes, The Art of Scientific Computing, 3d edition"
// (Cambridge Univ. Press) by W.H. Press, S.A. Teukolsky, W.T. Vetterling, and B.P. Flannery
/*******************************************************************************
Given a matrix a[m][n], this method computes its singular value
decomposition, *this = U.W.VT.  The matrix U replaces *this on output.  The diagonal
matrix of singular values W is output as a vector w[n].  The matrix vt is output as vt[n][n]
*******************************************************************************/
// compute (a2 + b2)^1/2 without destructive underflow or overflow
template<typename T>
T pythag (T a, T b) {
    T absa, absb;
    absa = fabs(a);
    absb = fabs(b);
    if (absa > absb) return absa*nd4j::math::nd4j_sqrt<T>(1.f + (absb/absa)*(absb/absa));
    else return (absb == 0.f ? 0.f : absb*nd4j::math::nd4j_sqrt<T>(1.f + (absa/absb)*(absa/absb)));
};


template<typename T>
void NDArray<T>::svd(NDArray<T>& u, NDArray<T>& w, NDArray<T>& vt)
{
    if(rankOf() !=2 || w.rankOf() !=2 || vt.rankOf() !=2)
        throw "SVD operation: rank of some of input matrices is not equal 2 !";

    int m = rows();
    int n = columns();

    if(w.rows() !=1 || w.columns() !=n || vt.rows() !=n || vt.columns() !=n)
        throw "SVD operation: shape of some of input matrices is wrong !";

    u = *this;
	const T eps = 1e-10f;
	bool flag;
	int i,its,j,jj,k,l,nm;
	T anorm,c,f,g,h,s,scale,x,y,z;
	std::vector<T> rv1(n);
	// Householder reduction to bidiagonal form
	g = scale = anorm = 0.f;
	for (i=0;i<n;i++) {
		l=i+2;
		rv1[i]=scale*g;
		g=s=scale=0.f;
		if (i < m) {
			for (k=i;k<m;k++) scale += nd4j::math::nd4j_abs<T>(u(k,i));
			if (scale != 0.f) {
				for (k=i;k<m;k++) {
					u(k,i) /= scale;
					s += u(k,i)*u(k,i);
				}
				f=u(i,i);
				g = - nd4j::math::nd4j_copysign<T>(nd4j::math::nd4j_sqrt<T>(s),f);
				h=f*g-s;
				u(i,i)=f-g;
				for (j=l-1;j<n;j++) {
					for (s=0.f,k=i;k<m;k++) s += u(k,i)*u(k,j);
					f=s/h;
					for (k=i;k<m;k++) u(k,j) += f*u(k,i);
				}
				for (k=i;k<m;k++) u(k,i) *= scale;
			}
		}
		w(0,i)=scale*g;
		g=s=scale=0.f;
		if (i+1 <= m && i+1 != n) {
			for (k=l-1;k<n;k++) scale += nd4j::math::nd4j_abs<T>(u(i,k));
			if (scale != 0.f) {
				for (k=l-1;k<n;k++) {
					u(i,k) /= scale;
					s += u(i,k)*u(i,k);
				}
				f=u(i,l-1);
				g = -nd4j::math::nd4j_copysign<T>(nd4j::math::nd4j_sqrt<T>(s),f);
				h=f*g-s;
				u(i,l-1)=f-g;
				for (k=l-1;k<n;k++) rv1[k]=u(i,k)/h;
				for (j=l-1;j<m;j++) {
					for (s=0.f,k=l-1;k<n;k++) s += u(j,k)*u(i,k);
					for (k=l-1;k<n;k++) u(j,k) += s*rv1[k];
				}
				for (k=l-1;k<n;k++) u(i,k) *= scale;
			}
		}
		anorm = nd4j::math::nd4j_max<T>(anorm,(nd4j::math::nd4j_abs<T>(w(0,i)) + nd4j::math::nd4j_abs<T>(rv1[i])));
	}
	// accumulation of right-hand transformations
	for (i=n-1;i>=0;i--) {
		if (i < n-1) {
			if (g != 0.f) {
				for (j=l;j<n;j++)
					vt(j,i)=(u(i,j)/u(i,l))/g;
				for (j=l;j<n;j++) {
					for (s=0.f,k=l;k<n;k++) s += u(i,k)*vt(k,j);
					for (k=l;k<n;k++) vt(k,j) += s*vt(k,i);
				}
			}
			for (j=l;j<n;j++) vt(i,j)=vt(j,i)=0.f;
		}
		vt(i,i)=1.f;
		g=rv1[i];
		l=i;
	}
	// accumulation of left-hand transformations
	for (i=nd4j::math::nd4j_min<T>(m,n)-1;i>=0;i--) {
		l=i+1;
		g=w(0,i);
		for (j=l;j<n;j++) u(i,j)=0.f;
		if (g != 0.f) {
			g=1.0/g;
			for (j=l;j<n;j++) {
				for (s=0.f,k=l;k<m;k++) s += u(k,i)*u(k,j);
				f=(s/u(i,i))*g;
				for (k=i;k<m;k++) u(k,j) += f*u(k,i);
			}
			for (j=i;j<m;j++) u(j,i) *= g;
		}
		else
			for (j=i;j<m;j++) u(j,i)=0.f;
		++u(i,i);
	}
	// diagonalization of the bidiagonal form
	for (k=n-1;k>=0;k--) {						// loop over singular values
		for (its=0;its<30;its++) {				// loop over allowed iterations
			flag=true;
			for (l=k;l>=0;l--) {				// test for splitting
				nm=l-1;							// note that rv1[1] is always zero
				if (l == 0 || nd4j::math::nd4j_abs<T>(rv1[l]) <= eps*anorm) {
					flag=false;
					break;
				}
				if (nd4j::math::nd4j_abs<T>(w(0,nm)) <= eps*anorm) break;
			}
			if (flag) {							// Cancellation of rv1[l], if l > 1
				c=0.f;
				s=1.0;
				for (i=l;i<k+1;i++) {
					f=s*rv1[i];
					rv1[i] *=c;
					if (nd4j::math::nd4j_abs<T>(f) <= eps*anorm) break;
					g=w(0,i);
					h=pythag<T>(f,g);
					w(i,0)=h;
					h=1.f/h;
					c=g*h;
					s = -f*h;
					for (j=0;j<m;j++) {
						y=u(j,nm);
						z=u(j,i);
						u(j,nm)=y*c+z*s;
						u(j,i)=z*c-y*s;
					}
				}
			}
			z=w(0,k);
			if (l == k) {						// convergence
				if (z < 0.f) {					// singular value is made nonnegative
					w(0,k) = -z;
					for (j=0;j<n;j++) vt(j,k) = -vt(j,k);
				}
				break;
			}
			if (its == 29) throw("no convergence in 30 svdcmp iterations");
			x=w(0,l);
			nm=k-1;
			y=w(0,nm);
			g=rv1[nm];
			h=rv1[k];
			f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
			g=pythag<T>(f,1.f);
			f=((x-z)*(x+z)+h*((y/(f+nd4j::math::nd4j_copysign<T>(g,f)))-h))/x;
			c=s=1.0;
			for (j=l;j<=nm;j++) {
				i=j+1;
				g=rv1[i];
				y=w(0,i);
				h=s*g;
				g=c*g;
				z=pythag<T>(f,h);
				rv1[j]=z;
				c=f/z;
				s=h/z;
				f=x*c+g*s;
				g=g*c-x*s;
				h=y*s;
				y *= c;
				for (jj=0;jj<n;jj++) {
					x=vt(jj,j);
					z=vt(jj,i);
					vt(jj,j)=x*c+z*s;
					vt(jj,i)=z*c-x*s;
				}
				z=pythag<T>(f,h);
				w(0,j)=z;							// rotation can be arbitrary if z = 0
				if (z) {
					z=1.0/z;
					c=f*z;
					s=h*z;
				}
				f=c*g+s*y;
				x=c*y-s*g;
				for (jj=0;jj<m;jj++) {
					y=u(jj,j);
					z=u(jj,i);
					u(jj,j)=y*c+z*s;
					u(jj,i)=z*c-y*s;
				}
			}
			rv1[l]=0.f;
			rv1[k]=f;
			w(0,k)=x;
		}
	}
   // transpose vt
    vt.transposei();
}

    // default destructor
    template<typename T>
    NDArray<T>::~NDArray() {

        if (_isBuffAlloc && _workspace == nullptr)
            delete[] _buffer;

        if (_isShapeAlloc  && _workspace == nullptr)
            delete[] _shapeInfo;
    }
}

#endif

