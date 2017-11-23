#ifndef NDARRAY_H
#define NDARRAY_H

#include <initializer_list>
#include <functional>
#include "NativeOps.h"
#include <shape.h>
#include "NativeOpExcutioner.h"
#include <memory/Workspace.h>
#include <indexing/NDIndex.h>
#include <indexing/IndicesList.h>
#include <graph/Intervals.h>
#include <array/DataType.h>

namespace nd4j {

    template<typename T> class ND4J_EXPORT NDArray;
    template<typename T> NDArray<T> operator+(const T, const NDArray<T>&);
    // template<typename T> NDArray<T> operator-(const T, const NDArray<T>&);
    ND4J_EXPORT NDArray<float> operator-(const float, const NDArray<float>&);
    ND4J_EXPORT NDArray<float16> operator-(const float16, const NDArray<float16>&);
    ND4J_EXPORT NDArray<double> operator-(const double, const NDArray<double>&);
    template<typename T> NDArray<T> mmul(const NDArray<T>&, const NDArray<T>&);


    template<typename T>
    class NDArray {
    protected:
        bool _isView = false;

        T    *_buffer = nullptr;                          // pointer on flattened data array in memory
        int  *_shapeInfo = nullptr;                       // contains shape info:  matrix rank, numbers of elements per each dimension, dimensions strides, c-like or fortan-like order, element-wise-stride

        nd4j::memory::Workspace* _workspace = nullptr;

#ifdef __CUDACC__
        T* _bufferD = nullptr;
        int* _shapeInfoD = nullptr;
#endif

        bool _isShapeAlloc = false;                    // indicates whether user allocates memory for _shapeInfo by himself, in opposite case the memory must be allocated from outside
        bool _isBuffAlloc = false; 						// indicates whether user allocates memory for _buffer by himself, in opposite case the memory must be allocated from outside

        DataType _dataType = DataType_FLOAT;

    public:
        void* operator new(size_t i);
        void operator delete(void* p);

		// forbid assignment operator
		NDArray<T>& operator=(const NDArray<T>& other);
        
        // accessing operator for matrix, i - absolute index
        // be careful this method doesn't check the boundaries of array
        FORCEINLINE T operator()(const Nd4jIndex i) const;

        // modifying operator for matrix, i - absolute index
        // be careful this method doesn't check the boundaries of array
        FORCEINLINE T& operator()(const Nd4jIndex i);

        // accessing operator for 2D array, i - row, j - column
        FORCEINLINE T operator()(const int i, const int j) const;        

        // modifying operator for 2D array, i - row, j - column
        FORCEINLINE T& operator()(const int i, const int j);

        // accessing operator for 3D array
        FORCEINLINE T operator()(const int i, const int j, const int k) const;        

        // modifying operator for 3D array
        FORCEINLINE T& operator()(const int i, const int j, const int k);

        // default constructor, do not allocate memory, memory for array is passed from outside 
        NDArray(T *buffer = nullptr, int *shapeInfo = nullptr, nd4j::memory::Workspace* workspace = nullptr);

        //constructor, create empty array with at workspace
        NDArray(nd4j::memory::Workspace* workspace);

        // this constructor creates 2D NDArray, memory for array is allocated in constructor
        NDArray(const int rows, const int columns, const char order, nd4j::memory::Workspace* workspace = nullptr);

        // this constructor creates NDArray as single row (dimension is 1xlength), memory for array is allocated in constructor 
        NDArray(const Nd4jIndex length, const char order, nd4j::memory::Workspace* workspace = nullptr);

        // this constructor creates new NDArray with shape matching "other" array, do not copy "other" elements into new array
        NDArray(const NDArray<T> *other, const bool copyStrides = false, nd4j::memory::Workspace* workspace = nullptr);
		
		// copy constructor
        NDArray(const NDArray<T>& other);

		// constructor new NDArray using shape information from "shape" array, set all elements in new array to be zeros
		NDArray(const int* shapeInfo, const bool copyStrides = false, nd4j::memory::Workspace* workspace = nullptr);

        // this constructor creates new array using shape information contained in initializer_list/vector argument
        //NDArray(const char order, const std::initializer_list<int> shape, nd4j::memory::Workspace* workspace = nullptr);
        NDArray(const char order, const std::vector<int> &shape , nd4j::memory::Workspace* workspace = nullptr);

        NDArray(const char order, const std::vector<int> &shape, const std::vector<T> &data, nd4j::memory::Workspace* workspace = nullptr);

        NDArray(T *buffer, const char order, const std::vector<int> &shape , nd4j::memory::Workspace* workspace = nullptr);

        // This method replaces existing buffer/shapeinfo, AND releases original pointers (if releaseExisting TRUE)
        void replacePointers(T *buffer, int *shapeInfo, const bool releaseExisting = true);
 
        NDArray<T>* repeat(int dimension, const std::vector<int>& reps) const;

        void repeat(int dimension, NDArray<T>& target) const;

        DataType dataType() const;

        NDArray<T>* getView();

        NDArray<T> *subarray(IndicesList& indices) const;

        NDArray<T> *subarray(IndicesList& indices, std::vector<int>& strides) const;

        NDArray<T>* subarray(const std::initializer_list<NDIndex*>& idx) const;

        NDArray<T>* subarray(const Intervals& idx) const;

        NDArray<T>* cast(DataType dtype);

        void cast(NDArray<T>* target, DataType dtype);

        nd4j::memory::Workspace* getWorkspace() const {
            return _workspace;
        }

        T* getBuffer();
        T* buffer();


        int* shapeInfo();
        int* getShapeInfo() const;

        void setShapeInfo(int *shapeInfo) {
            if(_isShapeAlloc && _workspace == nullptr)
                delete []_shapeInfo;

            _shapeInfo = shapeInfo;
            _isShapeAlloc = false;
        }

        void setBuffer(T* buffer) {
            if(_isBuffAlloc && _workspace == nullptr)
                delete []_buffer;

            _buffer = buffer;
            _isBuffAlloc = false;
        }

        void triggerAllocationFlag(bool bufferAllocated, bool shapeAllocated) {
            _isBuffAlloc = bufferAllocated;
            _isShapeAlloc = shapeAllocated;
        }

        int sizeAt(int dim) const;

        // This method returns order of this NDArray
        char ordering() const {
            return shape::order(_shapeInfo);
        }

        bool isView() {
            return _isView;
        }

        // This method returns shape portion of shapeInfo
        int *shapeOf() const {
            return shape::shapeOf(_shapeInfo);
        }

        // This method returns strides portion of shapeInfo
        int *stridesOf() const {
            return shape::stride(_shapeInfo);
        }

        // This method returns rank of this NDArray
        int rankOf() const {
            return shape::rank(_shapeInfo);
        }

        // This method returns length of this NDArray
        Nd4jIndex lengthOf() const {
            return shape::length(_shapeInfo);
        }

        void svd(NDArray<T>& u, NDArray<T>& w, NDArray<T>& vt);

        bool permutei(const std::initializer_list<int>& dimensions);
        bool permutei(const std::vector<int>& dimensions);
        bool permutei(const int* dimensions, const int rank);

		NDArray<T>* permute(const std::initializer_list<int>& dimensions) const;
        NDArray<T>* permute(const std::vector<int>& dimensions) const;
        NDArray<T>* permute(const int* dimensions, const int rank) const;

        void permute(const int* dimensions, const int rank, NDArray<T>& target) const;
        void permute(const std::vector<int>& dimensions, NDArray<T>& target) const;

        // This method returns number of rows in this NDArray
        int rows() const {
            return shapeOf()[0];
        }

        // This method returns number of columns in this NDArray
        int columns() const {
            return shapeOf()[1];
        }

        // This method returns sizeof(T) for this NDArray
        int sizeOfT() const {
            return sizeof(T);
        }

        bool isContiguous();

        // print information about array shape
        void printShapeInfo(const char * msg = nullptr) const;

        void printBuffer(const char* msg = nullptr, int limit = -1);

        void printIndexedBuffer(const char* msg = nullptr, int limit = -1);

        // This method assigns values of given NDArray to this one, wrt order
        void assign(const NDArray<T> *other);

        // This method assigns given value to all elements in this NDArray
        void assign(const T value);

        // This method returns new copy of this NDArray, optionally in different order
        NDArray<T> *dup(const char newOrder = 'a');

        // Returns true if these two NDArrays have same shape
        FORCEINLINE bool isSameShape(const NDArray<T> *other) const;
        FORCEINLINE bool isSameShape(std::initializer_list<int> shape);
        FORCEINLINE bool isSameShape(std::initializer_list<Nd4jIndex> shape);
        FORCEINLINE bool isSameShape(std::vector<int>& shape);
        FORCEINLINE bool isSameShape(std::vector<Nd4jIndex >& shape);

		// Returns true if these two NDArrays have same shape
        FORCEINLINE bool isSameShapeStrict(const NDArray<T> *other) const;

        // This method returns sum of all elements of this NDArray
        T sumNumber() const;

        // This method returns mean number of this NDArray
        T meanNumber() const;

        int ews();

        // method calculates sum along dimension(s) in this array and save it reduced array
        NDArray<T> *sum(const std::initializer_list<int> &dimensions, const bool keepDims = false) const;

		// method reduces array by excluding its shapes along axes present in dimensions vector
        template<typename OpName>
        NDArray<T>* reduceAlongDimension(const std::vector<int>& dimensions, const bool keepDims = false) const;
		
        // method reduces array by excluding its shapes along axes present in dimensions vector
        template<typename OpName>
        NDArray<T>* reduceAlongDimension(const std::initializer_list<int>& dimensions, const bool keepDims = false) const;

        // method reduces array by excluding its shapes along axes present in dimensions vector
        template<typename OpName>
        void reduceAlongDimension(NDArray<T>* target, const std::vector<int>& dimensions, const bool keepDims = false) const;

        // method reduces array by excluding its shapes along axes present in dimensions vector
        template<typename OpName>
        NDArray<T> reduceAlongDims(const std::vector<int>& dimensions, const bool keepDims = false) const;

        template<typename OpName>
        T varianceNumber(bool biasCorrected = true);

        // 
        template<typename OpName>
        T reduceNumber(T *extraParams = nullptr);

        Nd4jIndex argMax(std::initializer_list<int> dimensions = {});

        // perform array transformation
        template<typename OpName>
        void applyTransform(T *extraParams = nullptr);

        // perform array transformation
        template<typename OpName>
        void applyTransform(NDArray<T> *target, T *extraParams = nullptr);

        // perform array transformation
        template<typename OpName>
        NDArray<T> transform(T *extraParams = nullptr);

        // perform pairwise transformation
        template<typename OpName>
        void applyPairwiseTransform(NDArray<T> *other, T *extraParams);

        // perform pairwise transformation
        template<typename OpName>
        void applyPairwiseTransform(NDArray<T> *other, NDArray<T> *target, T *extraParams);

        template<typename OpName>
        void applyBroadcast(std::initializer_list<int> dimensions, const NDArray<T>* tad, NDArray<T>* target = nullptr, T* extraArgs = nullptr);

        template <typename OpName>
        void applyBroadcast(std::vector<int> &dimensions, const NDArray<T> *tad, NDArray<T> *target = nullptr, T *extraArgs = nullptr);
        
        template <typename OpName>
        NDArray<T> applyTrueBroadcast(const NDArray<T>& tad, T *extraArgs = nullptr) const;

        template<typename OpName>
        void applyScalar(T scalar, NDArray<T>* target = nullptr, T *extraParams = nullptr);

        template<typename OpName>
        void applyScalar(NDArray<T>& scalar, NDArray<T>* target = nullptr, T *extraParams = nullptr);

#ifndef __JAVACPP_HACK__
        void applyLambda(const std::function<T(T)>& func, NDArray<T>* target = nullptr);

        void applyPairwiseLambda(NDArray<T>* other, const std::function<T(T, T)>& func, NDArray<T>* target = nullptr);
#endif

        // method makes copy of this array and applies to the copy the transpose operation, that is this array remains unaffected 
        NDArray<T> *transpose() const;

        // method performs transpose operation based on this array and store result in target, this array remains unaffected 
        void transpose(NDArray<T>& target) const;

        // This method applies in-place transpose to this array, so this array becomes transposed 
        void transposei();

        NDArray<T>* tensorAlongDimension(int index, const std::initializer_list<int>& dimensions) const;

        NDArray<T>* tensorAlongDimension(int index, const std::vector<int>& dimensions) const;

        // this method returns number of tensors along specified dimension(s)
        Nd4jIndex tensorsAlongDimension(const std::initializer_list<int> dimensions) const ;
        Nd4jIndex tensorsAlongDimension(const std::vector<int>& dimensions) const ;

        // This method returns true if buffer && shapeInfo were defined
        bool nonNull() const {
            return this->_buffer != nullptr && this->_shapeInfo != nullptr;
        }

        // This method returns true if two arrays are equal, with custom or default Eps value of 1e-5, false otherwise
        bool equalsTo(const NDArray<T> *other, T eps = (T) 1e-5f) const;

        // Return value from linear buffer
        FORCEINLINE T getScalar(const Nd4jIndex i) const;
        
        FORCEINLINE T getIndexedScalar(const Nd4jIndex i) const;

        // Returns value from 2D matrix by coordinates/indexes         
        FORCEINLINE T getScalar(const int i, const int j) const;

        // returns value from 3D tensor by coordinates        
        FORCEINLINE T getScalar(const int i, const int j, const int k) const;        
        
        FORCEINLINE void putIndexedScalar(const Nd4jIndex i, const T value);        

        // This method sets value in linear buffer to position i        
        FORCEINLINE void putScalar(const Nd4jIndex i, const T value);        

        // This method sets value in 2D matrix to position i, j         
        FORCEINLINE void putScalar(const int i, const int j, const T value);        

        // This method sets value in 3D matrix to position i,j,k        
        FORCEINLINE void putScalar(const int i, const int j, const int k, const T value);
        
        // This method adds given row to all rows in this NDArray, that is this array becomes affected
        void addiRowVector(const NDArray<T> *row);

        void addRowVector(const NDArray<T> *row, NDArray<T>* target) const;
        
        void subRowVector(const NDArray<T> *row, NDArray<T>* target) const;
        
        void mulRowVector(const NDArray<T> *row, NDArray<T>* target) const;

        void divRowVector(const NDArray<T> *row, NDArray<T>* target) const;

        void addColumnVector(const NDArray<T> *column, NDArray<T>* target) const;

		// This method adds given column to all columns in this NDArray, that is this array becomes affected
		void addiColumnVector(const NDArray<T> *column);

		// This method multiplies given column by all columns in this NDArray, that is this array becomes affected
		void muliColumnVector(const NDArray<T> *column);

        // this method returns number of bytes used by buffer & shapeInfo
        Nd4jIndex memoryFootprint();

        // this method returns true if this ndarray is 2d
        bool isMatrix() {
            return shape::isMatrix(this->_shapeInfo);
        }

        // this method returns true if this ndarray is vector
        bool isVector() const {
            return !isScalar() && shape::isVector(this->_shapeInfo);
        }

        // this method returns true if this ndarray is column vector
        bool isColumnVector() const {
            return !isScalar() && shape::isColumnVector(this->_shapeInfo);
        }

        // this method returns true if this ndarray is row vector
        bool isRowVector() const {
            return !isScalar() && shape::isRowVector(this->_shapeInfo);
        }

        // this method returns true if this ndarray is scalar
        bool isScalar() const {
            return this->lengthOf() == 1;
        }

        // these methods suited for FlatBuffers use.
        std::vector<T> getBufferAsVector();

        std::vector<int> getShapeAsVector();
        std::vector<int> getShapeInfoAsVector();
		
		// set new order and shape in case of suitable array length 
		bool reshapei(const char order, const std::initializer_list<int>& shape);
	
		// set new order and shape in case of suitable array length 
		bool reshapei(const char order, const std::vector<int>& shape);
	
		// create new array with corresponding order and shape, new array will point to the same _buffer as this array
		NDArray<T>* reshape(const char order, const std::vector<int>& shape);
		
		// calculate strides 
		void updateStrides(const char order);

		// change an array by repeating it the number of times given by reps.
		void tilei(const std::vector<int>& reps);

		// tile an array by repeating it the number of times given by reps.
		NDArray<T> tile(const std::vector<int>& reps) const;

        // tile an array by repeating it the number of times given by reps.
        void tile(const std::vector<int>& reps, NDArray<T>& target) const;
        
		// return array which is broadcasted from this and argument array  
		NDArray<T>*  broadcast(const NDArray<T>& other);
		
		// check whether array's rows (arg=0) or columns create orthogonal basis
		bool hasOrthonormalBasis(const int arg); 
		
		// check whether array is identity matrix
		bool isIdentityMatrix(); 

		// check whether array is unitary matrix
		bool isUnitary(); 
                
        // reduce dimensions in this array relying on index operations
        template<typename OpName>
        NDArray<T>* applyIndexReduce(const std::vector<int>& dimensions, const T *extraParams = nullptr) const;

        template<typename OpName>
        void applyIndexReduce(const NDArray<T>* target, const std::vector<int>& dimensions, const T *extraParams = nullptr) const;

        // apply reduce3 operations to this and other array, return result in new output array
        template<typename OpName>
        NDArray<T>* applyReduce3(const NDArray<T>* other, const T* extraParams = nullptr) const;

        // apply reduce3 (execAll) operations to this and other array, return result in new output array
        template<typename OpName>
        NDArray<T>* applyAllReduce3(const NDArray<T>* other, const std::vector<int>& dimensions, const T* extraParams = nullptr) const;
        
        // apply reduce3 (exec) operations to this and other array, return result in new output array
        template<typename OpName>
        NDArray<T>* applyReduce3(const NDArray<T>* other, const std::vector<int>& dimensions, const T* extraParams = nullptr) const;

        template<typename OpName>
        NDArray<T>* varianceAlongDimension(const bool biasCorrected, const std::vector<int>& dimensions) const;

        template<typename OpName>
        NDArray<T>* varianceAlongDimension(const bool biasCorrected, const std::initializer_list<int>& dimensions) const;

        // operator returns sub-array with buffer pointing at this->_buffer with certain offset
        NDArray<T> operator()(const Intervals& idx)  const;

        // addition operator array + array
        NDArray<T> operator+(const NDArray<T>& other) const;

        // addition operator array + scalar
        NDArray<T> operator+(const T scalar) const;
#ifndef _MSC_VER
        // addition operator scalar + array
        friend NDArray<T> nd4j::operator+<>(const T scalar, const NDArray<T>& arr);
#endif
        // subtraction operator array - array
        NDArray<T> operator-(const NDArray<T>& other) const;

        // subtraction operator array - scalar
        NDArray<T> operator-(const T& scalar) const;

        // subtraction operator scalar - array
        // friend NDArray<T> nd4j::operator-<>(const T scalar, const NDArray<T>& arr);
        friend NDArray<float> nd4j::operator-(const float scalar, const NDArray<float>& arr);
        friend NDArray<float16> nd4j::operator-(const float16 scalar, const NDArray<float16>& arr);
        friend NDArray<double> nd4j::operator-(const double scalar, const NDArray<double>& arr);

        // negative operator, it makes all array elements = -elements
        NDArray<T> operator-() const;

        // multiplication operator array1*array2
        NDArray<T> operator*(const NDArray<T>& other) const;        
    
        // multiplication operator array*scalar
        NDArray<T> operator*(const T scalar) const;

        // multiplication operator array1 *= array2
        void operator*=(const NDArray<T>& other);

        // multiplication operator array*scalar
        void operator*=(const T scalar);

        // division operator array1/array2
        NDArray<T> operator/(const NDArray<T>& other) const;        

        // division operator array1 /= array2
        void operator/=(const NDArray<T>& other);

        // division operator array*scalar
        void operator/=(const T scalar);

        // mathematical multiplication of two arrays
        friend NDArray<T> mmul<>(const NDArray<T>& left, const NDArray<T>& right);

        void assign(const NDArray<T>& other, const Intervals& idx);

        std::vector<int8_t> asByteVector();

        // default destructor
        ~NDArray(); 

    };


//////////////////////////////////////////////////////////////////////////
///// IMLEMENTATION OF INLINE METHODS ///// 
//////////////////////////////////////////////////////////////////////////


    template<typename T>
    FORCEINLINE int NDArray<T>::ews(){
        return shape::elementWiseStride(_shapeInfo);
    }
//////////////////////////////////////////////////////////////////////////
// accessing operator for matrix, i - absolute index
template<typename T>
FORCEINLINE T NDArray<T>::operator()(const Nd4jIndex i) const { 

    if (i >= shape::length(_shapeInfo))
            throw std::invalid_argument("NDArray::operator(i): input index is out of array length !");

    int  ews   = shape::elementWiseStride(_shapeInfo);   
    char order = ordering();   

    if(ews == 1 && order == 'c')
        return _buffer[i];
    else if(ews > 1 && order == 'c')
        return _buffer[i*ews];
    else {
        int idx[MAX_RANK];
        shape::ind2subC(rankOf(), shapeOf(), i, idx);
        Nd4jIndex offset = shape::getOffset(0, shapeOf(), stridesOf(), idx, rankOf());
        return _buffer[offset];        
    }
}

//////////////////////////////////////////////////////////////////////////
// modifying operator for matrix, i - absolute index
template<typename T>
FORCEINLINE T& NDArray<T>::operator()(const Nd4jIndex i) {

    if (i >= shape::length(_shapeInfo))
            throw std::invalid_argument("NDArray::operator(i): input index is out of array length !");

    int  ews   = shape::elementWiseStride(_shapeInfo);   
    char order = ordering();   

    if(ews == 1 && order == 'c')
        return _buffer[i];
    else if(ews > 1 && order == 'c')
        return _buffer[i*ews];
    else {
        int idx[MAX_RANK];
        shape::ind2subC(rankOf(), shapeOf(), i, idx);
        Nd4jIndex offset = shape::getOffset(0, shapeOf(), stridesOf(), idx, rankOf());
        return _buffer[offset];        
    }    
}

//////////////////////////////////////////////////////////////////////////
// accessing operator for 2D matrix, i - row, j - column
template<typename T>
FORCEINLINE T NDArray<T>::operator()(const int i, const int j) const {
    
    if (rankOf() != 2 || i >= shapeOf()[0] || j >= shapeOf()[1])
       throw std::invalid_argument("NDArray::operator(i,j): one of input indexes is out of array length or rank!=2 !");
    
    int coords[2] = {i, j};
    Nd4jIndex xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
    return _buffer[xOffset];
}

//////////////////////////////////////////////////////////////////////////
// modifying operator for 2D matrix, i - row, j - column
template<typename T>
FORCEINLINE T& NDArray<T>::operator()(const int i, const int j) {
    
    if (rankOf() != 2 || i >= shapeOf()[0] || j >= shapeOf()[1])
       throw std::invalid_argument("NDArray::operator(i,j): one of input indexes is out of array length or rank!=2 !");

    int coords[2] = {i, j};
    Nd4jIndex xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
    return _buffer[xOffset];
}

//////////////////////////////////////////////////////////////////////////
// accessing operator for 3D array, i - row, j - column
template<typename T>
FORCEINLINE T NDArray<T>::operator()(const int i, const int j, const int k) const {
    
    if (rankOf() != 3 || i >= shapeOf()[0] || j >= shapeOf()[1] || j >= shapeOf()[2])
       throw std::invalid_argument("NDArray::operator(i,j,k): one of input indexes is out of array length or rank!=3 !");
    
    int coords[3] = {i, j, k};
    Nd4jIndex xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
    return _buffer[xOffset];
}

//////////////////////////////////////////////////////////////////////////
// modifying operator for 3D array
template<typename T>
FORCEINLINE T& NDArray<T>::operator()(const int i, const int j, const int k) {
    
    if (rankOf() != 3 || i >= shapeOf()[0] || j >= shapeOf()[1] || k >= shapeOf()[2])
       throw std::invalid_argument("NDArray::operator(i,j,k): one of input indexes is out of array length or rank!=3 !");

    int coords[3] = {i, j, k};
    Nd4jIndex xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
    return _buffer[xOffset];
}

//////////////////////////////////////////////////////////////////////////
// Return value from linear buffer
template<typename T>
FORCEINLINE T NDArray<T>::getScalar(const Nd4jIndex i) const
{ return (*this)(i); }

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE T NDArray<T>::getIndexedScalar(const Nd4jIndex i) const
{ return (*this)(i); }

//////////////////////////////////////////////////////////////////////////
// Returns value from 2D matrix by coordinates/indexes         
template<typename T>
FORCEINLINE T NDArray<T>::getScalar(const int i, const int j) const
{ return (*this)(i, j); }

//////////////////////////////////////////////////////////////////////////
// returns value from 3D tensor by coordinates        
template<typename T>
FORCEINLINE T NDArray<T>::getScalar(const int i, const int j, const int k) const
{ return (*this)(i, j, k); }

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE void NDArray<T>::putIndexedScalar(const Nd4jIndex i, const T value)
{ (*this)(i) = value; }

//////////////////////////////////////////////////////////////////////////
// This method sets value in linear buffer to position i        
template<typename T>
FORCEINLINE void NDArray<T>::putScalar(const Nd4jIndex i, const T value)
{ (*this)(i) = value; }

//////////////////////////////////////////////////////////////////////////
// This method sets value in 2D matrix to position i, j         
template<typename T>
FORCEINLINE void NDArray<T>::putScalar(const int i, const int j, const T value)
{ (*this)(i,j) = value; }

//////////////////////////////////////////////////////////////////////////
// This method sets value in 3D matrix to position i,j,k        
template<typename T>
FORCEINLINE void NDArray<T>::putScalar(const int i, const int j, const int k, const T value)
{ (*this)(i,j,k) = value; }

//////////////////////////////////////////////////////////////////////////
template<typename T>
Nd4jIndex FORCEINLINE NDArray<T>::memoryFootprint() {    

    Nd4jIndex size = this->lengthOf() * this->sizeOfT();
    size += (this->rankOf() * 2 + 4) * sizeof(int);
    return size;
}

//////////////////////////////////////////////////////////////////////////
// returns true if these two NDArrays have same shape
// still the definition of inline function must be in header file
template<typename T>
FORCEINLINE bool NDArray<T>::isSameShape(const NDArray<T> *other) const {
    if (this->rankOf() != other->rankOf())
        return false;
    for (int e = 0; e < this->rankOf(); e++)
        if (this->shapeOf()[e] != other->shapeOf()[e])
            return false;
    return true;
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE bool NDArray<T>::isSameShape(std::initializer_list<int> other) {
    std::vector<int> shp(other);
    return isSameShape(shp);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE bool NDArray<T>::isSameShape(std::initializer_list<Nd4jIndex> other) {
    std::vector<Nd4jIndex> shp(other);
    return isSameShape(shp);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE bool NDArray<T>::isSameShape(std::vector<Nd4jIndex>& other) {
    if (this->rankOf() != other.size())
        return false;
    for (int e = 0; e < this->rankOf(); e++) {
        if (this->shapeOf()[e] != (int) other.at(e))
            return false;
    }
    return true;
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE bool NDArray<T>::isSameShape(std::vector<int>& other) {
    if (this->rankOf() != (int) other.size())
        return false;
    for (int e = 0; e < this->rankOf(); e++) {
        if (this->shapeOf()[e] != other.at(e))
            return false;
    }
    return true;
}

//////////////////////////////////////////////////////////////////////////
// returns true if these two NDArrays have same _shapeInfo
// still the definition of inline function must be in header file
template<typename T>
FORCEINLINE bool NDArray<T>::isSameShapeStrict(const NDArray<T> *other) const {        
  return shape::equalsStrict(_shapeInfo, other->_shapeInfo);
}



}
#endif
