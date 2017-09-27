#ifndef NDARRAY_H
#define NDARRAY_H


#include <initializer_list>
#include "NativeOps.h"
#include <shape.h>
#include "NativeOpExcutioner.h"
#include <memory/Workspace.h>
#include <indexing/NDIndex.h>
#include <indexing/IndicesList.h>

namespace nd4j {

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


    public:
        void* operator new(size_t i);
        void operator delete(void* p);

		// forbid assignment operator
		NDArray<T>& operator=(const NDArray<T>& other);
        // accessing operator for 2D matrix, i - row, j - column
        // be careful this method doesn't check the rank of array
        T operator()(const int i, const int j) const;

        // modifying operator for 2D matrix, i - row, j - column
        // be careful this method doesn't check the rank of array
        T& operator()(const int i, const int j);

        // default constructor, do not allocate memory, memory for array is passed from outside 
        NDArray(T *buffer = nullptr, int *shapeInfo = nullptr, nd4j::memory::Workspace* workspace = nullptr);

        // this constructor creates 2D NDArray, memory for array is allocated in constructor 
        NDArray(const int rows, const int columns, const char order, nd4j::memory::Workspace* workspace = nullptr);

        // this constructor creates NDArray as single row (dimension is 1xlength), memory for array is allocated in constructor 
        NDArray(const Nd4jIndex length, const char order, nd4j::memory::Workspace* workspace = nullptr);

        // this constructor creates new NDArray with shape matching "other" array, do not copy "other" elements into new array
        NDArray(const NDArray<T> *other, nd4j::memory::Workspace* workspace = nullptr);
		
		// copy constructor
        NDArray(const NDArray<T>& other, nd4j::memory::Workspace* workspace = nullptr);

		// constructor new NDArray using shape information from "shape" array, set all elements in new array to be zeros
		NDArray(const int* shapeInfo, nd4j::memory::Workspace* workspace = nullptr);

        // this constructor creates new array using shape information contained in initializer_list/vector argument
        NDArray(const char order, const std::initializer_list<int> &shape, nd4j::memory::Workspace* workspace = nullptr);
        NDArray(const char order, const std::vector<int> &shape , nd4j::memory::Workspace* workspace = nullptr);

        // This method replaces existing buffer/shapeinfo, AND releases original pointers (if releaseExisting TRUE)
        void replacePointers(T *buffer, int *shapeInfo, const bool releaseExisting = true);
 
        NDArray<T>* repeat(int dimension, const std::vector<int>& reps);

        NDArray<T>* getView();

        NDArray<T> *subarray(IndicesList& indices);

        nd4j::memory::Workspace* getWorkspace() {
            return _workspace;
        }

        T* getBuffer();

        int* getShapeInfo();

        void setShapeInfo(int *shapeInfo) {
            _shapeInfo = shapeInfo;
            _isShapeAlloc = false;
        }

        void setBuffer(T* buffer) {
            _buffer = buffer;
            _isBuffAlloc = false;
        }

        void triggerAllocationFlag(bool bufferAllocated, bool shapeAllocated) {
            _isBuffAlloc = bufferAllocated;
            _isShapeAlloc = shapeAllocated;
        }

        int sizeAt(int dim);

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

		NDArray<T>* permute(const std::initializer_list<int>& dimensions);
        NDArray<T>* permute(const std::vector<int>& dimensions);
        NDArray<T>* permute(const int* dimensions, const int rank);

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
        void printShapeInfo(const char * msg = nullptr) const {
            //shape::printShapeInfo(_shapeInfo);
            if (msg == nullptr)
                shape::printShapeInfoLinear(_shapeInfo);
            else {
                int rank = shape::rank(_shapeInfo);
                printf("%s: [", msg);
                for (int i = 0; i < rank * 2 + 4; i++) {
                    printf("%i, ", _shapeInfo[i]);
                }
                printf("]\n");
            }
        }

        void printBuffer(const char* msg = nullptr);

        // This method assigns values of given NDArray to this one, wrt order
        void assign(NDArray<T> *other);

        // This method assigns given value to all elements in this NDArray
        void assign(const T value);

        // This method returns new copy of this NDArray, optionally in different order
        NDArray<T> *dup(const char newOrder);

        // Returns true if these two NDArrays have same shape
        inline bool isSameShape(const NDArray<T> *other) const;
        inline bool isSameShape(std::initializer_list<int> shape);
        inline bool isSameShape(std::initializer_list<Nd4jIndex> shape);
        inline bool isSameShape(std::vector<int>& shape);
        inline bool isSameShape(std::vector<Nd4jIndex >& shape);

		// Returns true if these two NDArrays have same shape
        inline bool isSameShapeStrict(const NDArray<T> *other) const;

        // This method returns sum of all elements of this NDArray
        T sumNumber() const;

        // This method returns mean number of this NDArray
        T meanNumber() const;

        // method calculates sum along dimension(s) in this array and save it to row: as new NDArray with dimensions 1xN
        NDArray<T> *sum(const std::initializer_list<int> &dimensions) const;

		// eventually this method reduces this array to 1xN row 
        template<typename OpName>
        NDArray<T> *reduceAlongDimension(const std::vector<int> &dimensions) const;
		
        // eventually this method reduces this array to 1xN row 
        template<typename OpName>
        NDArray<T> *reduceAlongDimension(const std::initializer_list<int> &dimensions) const;

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

        // perform pairwise transformation
        template<typename OpName>
        void applyPairwiseTransform(NDArray<T> *other, T *extraParams);

        // perform pairwise transformation
        template<typename OpName>
        void applyPairwiseTransform(NDArray<T> *other, NDArray<T> *target, T *extraParams);

        template<typename OpName>
        void applyBroadcast(std::initializer_list<int> dimensions, NDArray<T>* tad, NDArray<T>* target = nullptr, T* extraArgs = nullptr);

        template<typename OpName>
        void applyBroadcast(std::vector<int>& dimensions, NDArray<T>* tad, NDArray<T>* target = nullptr, T* extraArgs = nullptr);

        template<typename OpName>
        void applyScalar(T scalar, NDArray<T>* target = nullptr, T *extraParams = nullptr);

        template<typename OpName>
        void applyScalar(NDArray<T>& scalar, NDArray<T>* target = nullptr, T *extraParams = nullptr);


        // method makes copy of this array and applies to the copy the transpose operation, that is this array remains unaffected 
        NDArray<T> *transpose() const;

        // This method applies in-place transpose to this array, so this array becomes transposed 
        void transposei();

        NDArray<T>* tensorAlongDimension(int index, std::initializer_list<int> dimensions);

        NDArray<T>* tensorAlongDimension(int index, std::vector<int>& dimensions);

        // this method returns number of tensors along specified dimension(s)
        Nd4jIndex tensorsAlongDimension(std::initializer_list<int> dimensions);
        Nd4jIndex tensorsAlongDimension(std::vector<int>& dimensions);

        // This method returns true if buffer && shapeInfo were defined
        bool nonNull() const {
            return this->_buffer != nullptr && this->_shapeInfo != nullptr;
        }

        // This method returns true if two arrays are equal, with custom or default Eps value of 1e-5, false otherwise
        bool equalsTo(const NDArray<T> *other, T eps = (T) 1e-5f) const;

        // Return value from linear buffer
        T getScalar(const Nd4jIndex i) const;

        T getIndexedScalar(const Nd4jIndex i);

        // Returns value from 2D matrix by coordinates/indexes 
        T getScalar(const int i, const int j) const;

        // returns value from 3D tensor by coordinates
        T getScalar(const int i, const int k, const int j) const;

        void putIndexedScalar(const Nd4jIndex i, const T value);

        // This method sets value in linear buffer to position i
        void putScalar(const Nd4jIndex i, const T value);

        // This method sets value in 2D matrix to position i, j 
        void putScalar(const int i, const int j, const T value);

        // This method sets value in 3D matrix to position i,j,k
        void putScalar(const int i, const int k, const int j, const T value);

        // This method adds given row to all rows in this NDArray, that is this array becomes affected
        void addiRowVector(const NDArray<T> *row);

        void addRowVector(const NDArray<T> *row, NDArray<T>* target);

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
        bool isRowVector() {
            return !isScalar() && shape::isRowVector(this->_shapeInfo);
        }

        // this method returns true if this ndarray is scalar
        bool isScalar() const {
            return this->lengthOf() == 1;
        }

        // these methods suited for FlatBuffers use.
        std::vector<T> getBufferAsVector();

        std::vector<int32_t> getShapeAsVector();
		
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
		NDArray<T>*  tile(const std::vector<int>& reps);
        
		// return array which is broadcasted from this and argument array  
		NDArray<T>*  broadcast(const NDArray<T>& other);
		
		// check whether array's rows (arg=0) or columns create orthogonal basis
		bool hasOrthonormalBasis(const int arg); 
		
		// check whether array is identity matrix
		bool isIdentityMatrix(); 

		// check whether array is unitary matrix
		bool isUnitary(); 
		
		// default destructor
        ~NDArray();

    };


    template<typename T>
    Nd4jIndex inline NDArray<T>::memoryFootprint() {
        Nd4jIndex size = this->lengthOf() * this->sizeOfT();
        size += (this->rankOf() * 2 + 4) * sizeof(int);

        return size;
    }


// returns true if these two NDArrays have same shape
// still the definition of inline function must be in header file
    template<typename T>
    inline bool NDArray<T>::isSameShape(const NDArray<T> *other) const {

        if (this->rankOf() != other->rankOf())
            return false;

        for (int e = 0; e < this->rankOf(); e++)
            if (this->shapeOf()[e] != other->shapeOf()[e])
                return false;

        return true;
    }

    template<typename T>
    inline bool NDArray<T>::isSameShape(std::initializer_list<int> other) {
        std::vector<int> shp(other);
        return isSameShape(shp);
    }

    template<typename T>
    inline bool NDArray<T>::isSameShape(std::initializer_list<Nd4jIndex> other) {
        std::vector<Nd4jIndex> shp(other);
        return isSameShape(shp);
    }

    template<typename T>
    inline bool NDArray<T>::isSameShape(std::vector<Nd4jIndex>& other) {
        if (this->rankOf() != other.size())
            return false;

        for (int e = 0; e < this->rankOf(); e++) {
            if (this->shapeOf()[e] != (int) other.at(e))
                return false;
        }

        return true;
    }


    template<typename T>
    inline bool NDArray<T>::isSameShape(std::vector<int>& other) {
        if (this->rankOf() != other.size())
            return false;

        for (int e = 0; e < this->rankOf(); e++) {
            if (this->shapeOf()[e] != other.at(e))
                return false;
        }

        return true;
    }

    // returns true if these two NDArrays have same _shapeInfo
// still the definition of inline function must be in header file
    template<typename T>
    inline bool NDArray<T>::isSameShapeStrict(const NDArray<T> *other) const {        
    
		return shape::equalsStrict(_shapeInfo, other->_shapeInfo);
    }
}
#endif
