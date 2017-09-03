#ifndef NDARRAY_CPP
#define NDARRAY_CPP

#include "../NDArray.h"
#include <stdexcept>
#include <memory>


namespace nd4j {

////////////////////////////////////////////////////////////////////////
// default constructor, do not allocate memory, memory for array is passed from outside 
template <typename T> NDArray<T>::NDArray(T *buffer, int *shapeInfo ) {        
    
    _buffer    = buffer;
    _shapeInfo = shapeInfo;
    _isBuffAlloc = false;                                  // indicate that memory for array is passed from outside
    _isShapeAlloc = false;
}

    template <typename T> NDArray<T>::NDArray(const Nd4jIndex length, const char order) {
        if (length < 1)
            throw "Can't allocate non-positive number of elements";

        _buffer = new T[length];
        memset(_buffer, 0, length * sizeOfT());              // set all elements in new array to be zeros

        int *shape = new int[2] {1, length};

        if (order == 'f') {
            _shapeInfo = shape::shapeBufferFortran(2, shape);
            _shapeInfo[7] = 102;
        } else {
            _shapeInfo = shape::shapeBuffer(2, shape);
            _shapeInfo[7] = 99;
        }

        _shapeInfo[6] = 1;
        _isBuffAlloc = true;
        _isShapeAlloc = true;

        delete[] shape;
    }

////////////////////////////////////////////////////////////////////////
// this constructor creates 2D NDArray, memory for array is allocated in this constructor 
template <typename T> NDArray<T>::NDArray(const int rows, const int columns, const char order) {
    
    _buffer = new T[rows * columns];
    memset(_buffer, 0, rows * columns * sizeOfT());              // set all elements in new array to be zeros

    int *shape = new int[2] {rows, columns};

    if (order == 'f') {
        _shapeInfo = shape::shapeBufferFortran(2, shape);
        _shapeInfo[7] = 102;
    } else {
        _shapeInfo = shape::shapeBuffer(2, shape);
        _shapeInfo[7] = 99;
    }

    _shapeInfo[6] = 1;
    _isBuffAlloc = true; 
    _isShapeAlloc = true;
    
    delete[] shape;    
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
NDArray<T>::NDArray(const NDArray<T> *other) {
    int arrLength = shape::length(other->_shapeInfo);
    int shapeLength = shape::rank(other->_shapeInfo)*2 + 4;

    _buffer = new T[arrLength];
    memcpy(_buffer, other->_buffer, arrLength*sizeOfT());      // copy other._buffer information into new array

    _shapeInfo = new int[shapeLength];
    memcpy(_shapeInfo, other->_shapeInfo, shapeLength*sizeof(int));     // copy shape information into new array

    _isBuffAlloc = true;
    _isShapeAlloc = true;
}

////////////////////////////////////////////////////////////////////////
// copy constructor
template <typename T> NDArray<T>::NDArray(const NDArray<T>& other)
{
    int arrLength = shape::length(other._shapeInfo);
    int shapeLength = shape::rank(other._shapeInfo)*2 + 4;
    
    _buffer = new T[arrLength];
    memcpy(_buffer, other._buffer, arrLength*sizeOfT());      // copy other._buffer information into new array
 
    _shapeInfo = new int[shapeLength];             
    memcpy(_shapeInfo, other._shapeInfo, shapeLength*sizeof(int));     // copy shape information into new array
    
    _isBuffAlloc = true; 
    _isShapeAlloc = true;
}

////////////////////////////////////////////////////////////////////////
// this constructor creates new array using rank information contained in initializer_list argument
template <typename T> NDArray<T>::NDArray(const char order, const std::initializer_list<int>& shape) {
    
    int rank = (int) shape.size();

    if (rank > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    int *shapeOf = new int[rank];
    int cnt = 0;

    for (auto& item: shape)
        shapeOf[cnt++] = item;

    if (order == 'f')
        _shapeInfo = shape::shapeBufferFortran(rank, shapeOf);
    else 
        _shapeInfo = shape::shapeBuffer(rank, shapeOf);

    _buffer = new T[shape::length(_shapeInfo)];
    memset(_buffer, 0, sizeOfT() * shape::length(_shapeInfo));
    
    _isBuffAlloc = true; 
    _isShapeAlloc = true;
    
    delete[] shapeOf;
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
    NDArray<T>* NDArray<T>::tensorAlongDimension(int index, std::initializer_list<int> dimensions) {
        std::vector<int> vector(dimensions);
        return tensorAlongDimension(index, vector);
    }

    template <typename T>
    void NDArray<T>::printBuffer() {
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
        if (dimensions.size() > this->rankOf())
            throw "TAD can't have dimensions higher then original array";

        std::vector<int> copy(dimensions);

        // we need to sort dimensions (?)
        if (dimensions.size() > 1)
            std::sort (copy.begin(), copy.end());

        int *dims = copy.data();

        Nd4jIndex tadLength = shape::tadLength(this->_shapeInfo, dims, copy.size());
        Nd4jIndex numTads = this->lengthOf() / tadLength;

        if (index >= numTads)
            throw "Can't get index higher than total number of TADs";

        std::unique_ptr<shape::TAD> tad(new shape::TAD(this->_shapeInfo, dims, copy.size()));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        T* buffer = this->_buffer + tad->tadOffsets[index];
        int* shapeInfo = new int[shape::shapeInfoLength(tad->tadOnlyShapeInfo[0])];
        std::memcpy(shapeInfo, tad->tadOnlyShapeInfo, shape::shapeInfoByteLength(tad->tadOnlyShapeInfo));

        auto array = new NDArray<T>(buffer, shapeInfo);
        array->_isBuffAlloc = false;
        array->_isShapeAlloc = true;
        array->_isView = true;

		array->printShapeInfo();
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
template <typename T> void NDArray<T>::updateStrides() {
	
	int rank = rankOf();	
	int doubleRank = 2*rank;
	if(ordering() == 'c') {
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
    _shapeInfo[doubleRank + 3] = (int)ordering();
}

//////////////////////////////////////////////////////////////////////////
// set new order and shape in case of suitable array length 
template <typename T> bool NDArray<T>::reshape(char order, const std::initializer_list<int>& shape) {

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
	updateStrides();

    return true;
}


//////////////////////////////////////////////////////////////////////////
// set new order and shape in case of suitable array length 
template <typename T> bool NDArray<T>::reshape(char order, const std::vector<int>& shape) {

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
    updateStrides();

    return true;
}

//////////////////////////////////////////////////////////////////////////
// change an array by repeating it the number of times given by reps.
template <typename T> void NDArray<T>::tile(const std::vector<int>& reps) {
	// check whether reps contains at least one zero (then throw exception) or whether all elements in reps are unities (then simply reshape or do nothing)
	int dim = reps.size();	
	int product = 1;
	for(const auto& item : reps)
		product *= item;
	if(product == 0)
		throw "Tile method: one of the elements in reps array is zero !";
	int rank = rankOf();
	int diff = rank - dim;
	if(product==1) {	    // in this case 2 possibilities are present: just reshape or nothing to do
		if(diff < 0) {		// reshape to higher dimension			
			std::vector<int> shapeNew = reps;				// need to have unities at first "diff" positions of new shape
			memcpy(&shapeNew[-diff], _shapeInfo+1, rank*sizeof(int));   // put old shape numbers at rest of positions
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
		memcpy(newShapeInfo + 1 - diff, _shapeInfo + 1, rank*sizeof(int));		// copy old dimensions to the right-hand side of newShapeInfo shape place
		for(int i=1; i <= dim; ++i)
			newShapeInfo[i] *= reps[i - 1];		// set new shape by multiplying old dimensions by corresponding numbers from reps 
	}
	else {
		newShapeInfo = new int[rank*2 + 4];
		memcpy(newShapeInfo, _shapeInfo, (rank*2 + 4)*sizeof(int));		// copy all elements of _shapeInfo to newShapeInfo
		for(int i=1; i <= dim; ++i)
			newShapeInfo[rank + 1 - i] *= reps[dim - i];		// set new shape by multiplying old dimensions by corresponding numbers from reps 
	}
	
	// create new buffer, in any case the memory amount new buffer points to is bigger then those for old _buffer
	T* newBuff = new T[shape::length(newShapeInfo)];
	char order = ordering();
	int arrLengthInBitsNew = sizeOfT()*shape::length(newShapeInfo);		
	int arrLengthInBitsOld = sizeOfT()*shape::length(_shapeInfo);		
	int bitStepNew, bitStepOld, numCopies;
	if(order == 'c') {
		bitStepNew = sizeOfT()*newShapeInfo[rank];
		bitStepOld = sizeOfT()*_shapeInfo[rank];
		numCopies = reps[dim-1];
	}
	else {
		bitStepNew = sizeOfT()*newShapeInfo[1];
		bitStepOld = sizeOfT()*_shapeInfo[1];
		if(diff <= 0)
			numCopies = reps[0];
		else	
			numCopies = 1;
	}	
	// fill newBuff, loop through all dimensions of newBuff except elementary dimension
	// looping through _buffer goes automatically by means of "position" index 
	int position = 0;
	for(int i=0;  i<arrLengthInBitsNew; i+=bitStepNew) {		
		for(int j=0; j<numCopies; j+=bitStepOld)
				memcpy(newBuff + i + j, _buffer + position, bitStepOld);
		position += bitStepOld;
		if(position == arrLengthInBitsOld)		// if loop through _buffer has come to end then start it again from beginning
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

	updateStrides();
	
}

// default destructor
    template<typename T>
    NDArray<T>::~NDArray() {

        if (_isBuffAlloc)
            delete[] _buffer;

        if (_isShapeAlloc)
            delete[] _shapeInfo;
    }
}

#endif

