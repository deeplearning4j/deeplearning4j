#ifndef NDARRAY_H
#define NDARRAY_H

#include <initializer_list>
#include <functional>
#include <shape.h>
#include "NativeOpExcutioner.h"
#include <memory/Workspace.h>
#include <indexing/NDIndex.h>
#include <indexing/IndicesList.h>
#include <graph/Intervals.h>
#include <array/DataType.h>

namespace nd4j {

    template<typename T> class ND4J_EXPORT NDArray;
    ND4J_EXPORT NDArray<float> operator-(const float, const NDArray<float>&);
    ND4J_EXPORT NDArray<float16> operator-(const float16, const NDArray<float16>&);
    ND4J_EXPORT NDArray<double> operator-(const double, const NDArray<double>&);
    ND4J_EXPORT NDArray<float> operator+(const float, const NDArray<float>&);
    ND4J_EXPORT NDArray<float16> operator+(const float16, const NDArray<float16>&);
    ND4J_EXPORT NDArray<double> operator+(const double, const NDArray<double>&);
    template<typename T> NDArray<T> mmul(const NDArray<T>&, const NDArray<T>&);


    template<typename T>
    class NDArray {
    
    protected:
       /**
       *  if true then array doesn't own buffer and simply points to another's buffer
       */                  
        bool _isView = false;

        /**
        *  pointer on flattened data array in memory
        */  
        T *_buffer = nullptr;                          

        /**
        *  contains shape info:  matrix rank, numbers of elements per each dimension, dimensions strides, element-wise-stride, c-like or fortan-like order
        */  
        int *_shapeInfo = nullptr;                       

        /**
        *  pointer on externally allocated memory where _buffer and _shapeInfo are stored
        */  
        nd4j::memory::Workspace* _workspace = nullptr;
        
        /**
        *  alternative buffers for special computational devices (like GPUs for CUDA)
        */  
        T* _bufferD = nullptr;
        int* _shapeInfoD = nullptr;

        /**
        *  indicates whether user allocates memory for _buffer/_shapeInfo by himself, in opposite case the memory must be allocated from outside
        */  
        bool _isShapeAlloc = false;                    
        bool _isBuffAlloc = false; 						

        /**
        *  type of array elements
        */  
        DataType _dataType = DataType_FLOAT;

    public:        
        
        /**
        *  default constructor, do not allocate memory, memory for array is passed from outside 
        */
        NDArray(T *buffer = nullptr, int *shapeInfo = nullptr, nd4j::memory::Workspace* workspace = nullptr);
        
        /**
         * Constructor for scalar NDArray
         */
        NDArray(T scalar);

        /**
        *  copy constructor
        */
        NDArray(const NDArray<T>& other);

        /**
        *  constructor, create empty array stored at given workspace
        */
        NDArray(nd4j::memory::Workspace* workspace);

        /**
        *  this constructor creates 2D NDArray with shape [rows x columns], memory for array is allocated in constructor
        */
        NDArray(const int rows, const int columns, const char order, nd4j::memory::Workspace* workspace = nullptr);

        /**
        *  this constructor creates NDArray as single row, dimension is [1 x length], memory for array is allocated in constructor 
        */ 
        NDArray(const Nd4jIndex length, const char order, nd4j::memory::Workspace* workspace = nullptr);

        /**
        *  this constructor creates new NDArray with shape matching "other" array, do not copy "other" elements into new array
        */
        NDArray(const NDArray<T> *other, const bool copyStrides = false, nd4j::memory::Workspace* workspace = nullptr);
				
        /**
		*  constructor creates new NDArray using shape information from "shapeInfo", set all elements in new array to be zeros, if copyStrides is true then use stride values from "shapeInfo", else calculate strides independently 
        */
		NDArray(const int* shapeInfo, const bool copyStrides = false, nd4j::memory::Workspace* workspace = nullptr);

        /**
        *  this constructor creates new array using shape information contained in vector argument    
        */
        NDArray(const char order, const std::vector<int> &shape , nd4j::memory::Workspace* workspace = nullptr);

        /**
        *  this constructor creates new array with elements copied from data and using shape information stored in shape
        */
        NDArray(const char order, const std::vector<int> &shape, const std::vector<T> &data, nd4j::memory::Workspace* workspace = nullptr);

        /**
        *  this constructor creates new array using given buffer (without memory allocating) and shape information stored in shape
        */
        NDArray(T *buffer, const char order, const std::vector<int> &shape , nd4j::memory::Workspace* workspace = nullptr);

        /**
        *  assignment operator
        */
        NDArray<T>& operator=(const NDArray<T>& other);

        /**
        *  assignment operator, assigns the same scalar to all array elements 
        */
        NDArray<T>& operator=(const T scalar);

        /**
        *   operators for memory allocation and deletion
        */ 
        void* operator new(size_t i);
        void operator delete(void* p);

        /**
        *  method replaces existing buffer/shapeinfo, AND releases original pointers (if releaseExisting TRUE)
        */
        void replacePointers(T *buffer, int *shapeInfo, const bool releaseExisting = true);
 
        /**
        *  create a new array by replicating current array by repeats times along given dimension
        *  dimension - dimension along which to repeat elements
        *  repeats - number of repetitions
        */        
        NDArray<T>* repeat(int dimension, const std::vector<int>& repeats) const;

        /**
        *  fill target array by repeating current array 
        *  dimension - dimension along which to repeat elements        
        */
        void repeat(int dimension, NDArray<T>& target) const;

        /**
        *  return _dataType;
        */
        DataType dataType() const;

        /**
        *  creates array which is view of this array
        */
        NDArray<T>* getView();

        /**
        *  creates array which points on certain sub-range of this array, sub-range is defined by given indices
        */
        NDArray<T> *subarray(IndicesList& indices) const;
        NDArray<T> *subarray(IndicesList& indices, std::vector<int>& strides) const;
        NDArray<T>* subarray(const std::initializer_list<NDIndex*>& idx) const;
        NDArray<T>* subarray(const Intervals& idx) const;

        /**
        *  cast array elements to given dtype
        */ 
        NDArray<T>* cast(DataType dtype);
        void cast(NDArray<T>* target, DataType dtype);

        /**
        *   returns _workspace
        */
        nd4j::memory::Workspace* getWorkspace() const {
            return _workspace;
        }

        /**
        *   returns _buffer
        */
        T* getBuffer();        
        T* buffer();

        /**
        *   returns _shapeInfo
        */
        int* shapeInfo();
        int* getShapeInfo() const;

        /**
        *  if _bufferD==nullptr return _buffer, else return _bufferD
        */
        T* specialBuffer();

        /**
        *  if _shapeInfoD==nullptr return _shapeInfo, else return _shapeInfoD
        */
        int* specialShapeInfo();

        /**
        *  set values for _bufferD and _shapeInfoD
        */
        void setSpecialBuffers(T * buffer, int *shape);

        /**
        *  permutes (in-place) the dimensions in array according to "dimensions" array
        */
        bool permutei(const std::initializer_list<int>& dimensions);
        bool permutei(const std::vector<int>& dimensions);
        bool permutei(const int* dimensions, const int rank);

        /**
        *  permutes the dimensions in array according to "dimensions" array
        */
		NDArray<T>* permute(const std::initializer_list<int>& dimensions) const;
        NDArray<T>* permute(const std::vector<int>& dimensions) const;
        NDArray<T>* permute(const int* dimensions, const int rank) const;

        /**
         * This method streamlines given view or permuted array, and reallocates buffer
         */
        void streamline(char order = 'a');

        /**
        *  permutes the dimensions in target according to "dimensions" array
        */
        void permute(const int* dimensions, const int rank, NDArray<T>& target) const;
        void permute(const std::vector<int>& dimensions, NDArray<T>& target) const;

        /**
        *  check whether array is contiguous in memory
        */ 
        bool isContiguous();

        /**
        *  prints information about array shape
        *  msg - message to print out 
        */ 
        void printShapeInfo(const char * msg = nullptr) const;

        /**
        *  prints buffer elements
        *  msg - message to print out 
        *  limit - number of array elements to print out
        */ 
        void printBuffer(const char* msg = nullptr, int limit = -1);

        /**
        *  prints buffer elements, takes into account offset between elements (element-wise-stride)
        *  msg - message to print out 
        *  limit - number of array elements to print out
        */ 
        void printIndexedBuffer(const char* msg = nullptr, int limit = -1);

        /**
        *  this method assigns values of given array to this one
        */ 
        void assign(const NDArray<T>* other);

        /**
        *  this method assigns values of given array to this one
        */ 
        void assign(const NDArray<T>& other);

        /**
        *  this method assigns given value to all elements in array
        */ 
        void assign(const T value);

        /**
        *  returns new copy of this array, optionally in different order
        */
        NDArray<T> *dup(const char newOrder = 'a');

        /** 
        *  returns sum of all elements of array
        */
        T sumNumber() const;

        /**
        *  returns mean number of array
        */ 
        T meanNumber() const;

        /**
        *  calculates sum along dimension(s) in this array and save it to created reduced array
        *  dimensions - array of dimensions to calculate sum over
        *  keepDims - if true then put unities in place of reduced dimensions
        */
        NDArray<T> *sum(const std::initializer_list<int> &dimensions, const bool keepDims = false) const;

		/**
        *  method reduces array by excluding its shapes along dimensions present in given dimensions vector, result is stored in new array to be returned
        *  dimensions - array of dimensions to reduce along
        *  keepDims - if true then put unities in place of reduced dimensions
        */ 
        template<typename OpName>
        NDArray<T>* reduceAlongDimension(const std::vector<int>& dimensions, const bool keepDims = false) const;
		template<typename OpName>
        NDArray<T>* reduceAlongDimension(const std::initializer_list<int>& dimensions, const bool keepDims = false) const;
        template<typename OpName>
        NDArray<T> reduceAlongDims(const std::vector<int>& dimensions, const bool keepDims = false) const;

        /**
        *  method reduces array by excluding its shapes along dimensions present in given dimensions vector
        *  target - where to save result of reducing
        *  dimensions - array of dimensions to reduce along
        *  keepDims - if true then put unities in place of reduced dimensions
        *  extras - extra parameters
        */ 
        template<typename OpName>
        void reduceAlongDimension(NDArray<T>* target, const std::vector<int>& dimensions, const bool keepDims = false, T *extras = nullptr) const;

        /**
        *  return variance of array elements set
        *  biasCorrected -  if true bias correction will be applied
        */ 
        template<typename OpName>
        T varianceNumber(bool biasCorrected = true);

        /**
        *  apply scalar operation to array 
        *  extraParams - extra parameters for operation
        */  
        template<typename OpName>
        T reduceNumber(T *extraParams = nullptr) const;

        /**
        *  returns element index which corresponds to some condition imposed by operation
        *  extraParams - extra parameters for operation
        */ 
        template<typename OpName>
        Nd4jIndex indexReduceNumber(T *extraParams = nullptr);

        /**
        *  returns index of max element in a given array (optionally: along given dimension(s))
        *  dimensions - optional vector with dimensions
        */          
        Nd4jIndex argMax(std::initializer_list<int> dimensions = {});

        /**
        *  apply OpName transformation directly to array
        *  extraParams - extra parameters for operation
        */
        template<typename OpName>
        void applyTransform(T *extraParams = nullptr);

        /**
        *  apply OpName transformation to array and store result in target
        *  target - where to store result
        *  extraParams - extra parameters for operation
        */
        template<typename OpName>
        void applyTransform(NDArray<T> *target, T *extraParams = nullptr);

        /**
        *  apply OpName transformation directly to array
        *  extraParams - extra parameters for operation
        */
        template<typename OpName>
        NDArray<T> transform(T *extraParams = nullptr);

        /**
        *  apply pairwise OpName transformation based on "this" and "other" arras elements, store result in this array
        *  other - second array necessary for pairwise operation
        *  extraParams - extra parameters for operation
        */
        template<typename OpName>
        void applyPairwiseTransform(NDArray<T> *other, T *extraParams);

        /**
        *  apply pairwise OpName transformation based on "this" and "other" arras elements, store result in target array
        *  other - second array necessary for pairwise operation
        *  target - where to store result
        *  extraParams - extra parameters for operation
        */
        template<typename OpName>
        void applyPairwiseTransform(NDArray<T> *other, NDArray<T> *target, T *extraParams);

        /**
        *  apply operation which requires broadcasting, broadcast a smaller array (tad) along  bigger one (this)
        *  tad - array to broadcast
        *  dimensions -  array with dimensions to broadcast along
        *  target - where to store result
        *  extraParams - extra parameters for operation
        */               
        template<typename OpName>
        void applyBroadcast(std::initializer_list<int> dimensions, const NDArray<T>* tad, NDArray<T>* target = nullptr, T* extraArgs = nullptr);
        template <typename OpName>
        void applyBroadcast(std::vector<int> &dimensions, const NDArray<T> *tad, NDArray<T> *target = nullptr, T *extraArgs = nullptr);

        /**
        *  apply operation which requires broadcasting, broadcast one tensor along another, also this method checks the possibility of broadcasting
        *  other - input array 
        *  extraParams - extra parameters for operation
        */                       
        template <typename OpName>
        NDArray<T> applyTrueBroadcast(const NDArray<T>& other, T *extraArgs = nullptr) const;
        template <typename OpName>
        NDArray<T>* applyTrueBroadcast(const NDArray<T>* other, T *extraArgs = nullptr) const;

        /**
        *  apply operation which requires broadcasting, broadcast one tensor along another, also this method checks the possibility of broadcasting
        *  other - input array 
        *  target - where to store result
        *  checkTargetShape - if true check whether target shape is suitable for broadcasting
        *  extraParams - extra parameters for operation
        */                       
        template <typename OpName>
        void applyTrueBroadcast(const NDArray<T>* other, NDArray<T>* target, const bool checkTargetShape = true, T *extraArgs = nullptr) const;

        /** 
        *  apply a scalar operation to an array
        *  scalar - input scalar
        *  target - where to store result
        *  extraParams - extra parameters for operation
        */ 
        template<typename OpName>
        void applyScalar(T scalar, NDArray<T>* target = nullptr, T *extraParams = nullptr);

        /** 
        *  apply a scalar operation to an array
        *  scalar - input array which is simple scalar
        *  target - where to store result
        *  extraParams - extra parameters for operation
        */ 
        template<typename OpName>
        void applyScalar(NDArray<T>& scalar, NDArray<T>* target = nullptr, T *extraParams = nullptr);

        /** 
        *  apply operation "func" to an array
        *  func - what operation to apply
        *  target - where to store result
        */ 
#ifndef __JAVACPP_HACK__
        void applyLambda(const std::function<T(T)>& func, NDArray<T>* target = nullptr);

        /** 
        *  apply pairwise operation "func" to an array
        *  other - input array
        *  func - what pairwise operation to apply
        *  target - where to store result
        */ 
        void applyPairwiseLambda(NDArray<T>* other, const std::function<T(T, T)>& func, NDArray<T>* target = nullptr);
#endif

        /**
        *  apply OpName random operation to array 
        *  buffer - pointer on RandomBuffer
        *  y - optional input array
        *  z - optional input array
        *  extraArgs - extra parameters for operation
        */
        template<typename OpName>
        void applyRandom(nd4j::random::RandomBuffer *buffer, NDArray<T>* y = nullptr, NDArray<T>* z = nullptr, T* extraArgs = nullptr);

        /**
        *   apply transpose operation to the copy of this array, that is this array remains unaffected 
        */
        NDArray<T> *transpose() const;

        /**
        *  perform transpose operation and store result in target, this array remains unaffected 
        *  target - where to store result
        */ 
        void transpose(NDArray<T>& target) const;

        /**
        *  apply in-place transpose operation to this array, so this array becomes transposed 
        */ 
        void transposei();

        /**
        *  return array pointing on certain range of this array
        *  index - the number of array to be returned among set of possible arrays 
        *  dimensions - array of dimensions to point on
        */
        NDArray<T>* tensorAlongDimension(int index, const std::initializer_list<int>& dimensions) const;
        NDArray<T>* tensorAlongDimension(int index, const std::vector<int>& dimensions) const;

        /**
        *  returns the number of arrays pointing on specified dimension(s)
        *  dimensions - array of dimensions to point on
        */
        Nd4jIndex tensorsAlongDimension(const std::initializer_list<int> dimensions) const ;
        Nd4jIndex tensorsAlongDimension(const std::vector<int>& dimensions) const ;

        /**
        *  returns true if elements of two arrays are equal to within given epsilon value
        *  other - input array to compare
        *  eps - epsilon, this value defines the precision of elements comparison
        */
        bool equalsTo(const NDArray<T> *other, T eps = (T) 1e-5f) const;
        
        /**
        *  add given row vector to all rows of this array
        *  row - row vector to add
        */
        void addiRowVector(const NDArray<T> *row);

        /**
        *  add given row vector to all rows of this array, store result in target
        *  row - row vector to add
        *  target - where to store result
        */
        void addRowVector(const NDArray<T> *row, NDArray<T>* target) const;

        /**
        *  subtract given row vector from all rows of this array, store result in target
        *  row - row vector to subtract
        *  target - where to store result
        */
        void subRowVector(const NDArray<T> *row, NDArray<T>* target) const;
        
        /**
        *  multiply all rows of this array on given row vector, store result in target
        *  row - row vector to multiply on
        *  target - where to store result
        */
        void mulRowVector(const NDArray<T> *row, NDArray<T>* target) const;

        /**
        *  divide all rows of this array on given row vector, store result in target
        *  row - row vector to divide on
        *  target - where to store result
        */
        void divRowVector(const NDArray<T> *row, NDArray<T>* target) const;
        
        /**
        *  add given column vector to all columns of this array, store result in target
        *  column - column vector to add
        *  target - where to store result
        */
        void addColumnVector(const NDArray<T> *column, NDArray<T>* target) const;

        /**
        *  add given column vector to all columns of this array, this array becomes affected (in-place operation)
        *  column - column vector to add
        */
		void addiColumnVector(const NDArray<T> *column);

        /**
        *  multiply all columns of this array on given column vector, this array becomes affected (in-place operation)
        *  column - column vector to multiply on
        */
		void muliColumnVector(const NDArray<T> *column);

        /**
        *  returns number of bytes used by _buffer & _shapeInfo
        */
        Nd4jIndex memoryFootprint();        
        
        /**
        *  these methods suited for FlatBuffers use
        */
        std::vector<T> getBufferAsVector();
        std::vector<int> getShapeAsVector();
        std::vector<int> getShapeInfoAsVector();
				
        /**
        *  set new order and shape in case of suitable array length (in-place operation)
        *  order - order to set
        *  shape - shape to set
        */
		bool reshapei(const char order, const std::initializer_list<int>& shape);		
		bool reshapei(const char order, const std::vector<int>& shape);
	
        /**
        *  creates new array with corresponding order and shape, new array will point on _buffer of this array
        *  order - order to set
        *  shape - shape to set
        */
		NDArray<T>* reshape(const char order, const std::vector<int>& shape);
		
        /**
        *  calculate strides and set given order
        *  order - order to set
        */
		void updateStrides(const char order);

        /**
        *  change an array by repeating it the number of times given by reps (in-place operation)
        *  repeats - contains numbers of repetitions
        */
		void tilei(const std::vector<int>& repeats);

        /**
        *  returns new array which is created by by repeating of this array the number of times given by reps 
        *  repeats - contains numbers of repetitions
        */
		NDArray<T> tile(const std::vector<int>& repeats) const;

        /**
        *  change an array by repeating it the number of times given by reps (in-place operation)
        *  repeats - contains numbers of repetitions
        *  target - where to store result
        */
        void tile(const std::vector<int>& repeats, NDArray<T>& target) const;
        
        /**
        *  returns an array which is result of broadcasting of this and other arrays 
        *  other - input array
        */
		NDArray<T>*  broadcast(const NDArray<T>& other);
		
        /**
        *  check whether array's rows (arg=0) or columns (arg=1) create orthogonal basis
        *  arg - 0 -> row, 1 -> column
        */
		bool hasOrthonormalBasis(const int arg); 
				
        /**
        *  check whether array is identity matrix
        */
		bool isIdentityMatrix(); 
		
        /**
        *  check whether array is unitary matrix
        */
		bool isUnitary(); 
                        
        /**
        *  reduces dimensions in this array relying on index operation OpName
        *  dimensions - vector of dimensions to reduce along
        *  extraArgs - extra parameters for operation
        */
        template<typename OpName>
        NDArray<T>* applyIndexReduce(const std::vector<int>& dimensions, const T *extraParams = nullptr) const;

        /**
        *  reduces dimensions in array relying on index operation OpName
        *  target - where to store result
        *  dimensions - vector of dimensions to reduce along
        *  extraArgs - extra parameters for operation
        */
        template<typename OpName>
        void applyIndexReduce(const NDArray<T>* target, const std::vector<int>& dimensions, const T *extraParams = nullptr) const;

        /**
        *  apply reduce3 operation OpName to this and other array, return result in new output array
        *  other - input array
        *  extraArgs - extra parameters for operation
        */
        template<typename OpName>
        NDArray<T>* applyReduce3(const NDArray<T>* other, const T* extraParams = nullptr) const;

        /**
        *  apply reduce3 operation OpName to this and other array, return result in new output array
        *  other - input array
        *  dimensions - vector of dimensions to reduce along
        *  extraArgs - extra parameters for operation
        */
        template<typename OpName>
        NDArray<T>* applyAllReduce3(const NDArray<T>* other, const std::vector<int>& dimensions, const T* extraParams = nullptr) const;
                
        /**
        *  apply reduce3 (exec) operation OpName to this and other array, return result in new output array
        *  other - input array
        *  dimensions - vector of dimensions to reduce along
        *  extraArgs - extra parameters for operation
        */
        template<typename OpName>
        NDArray<T>* applyReduce3(const NDArray<T>* other, const std::vector<int>& dimensions, const T* extraParams = nullptr) const;


        /**
        *  returns variance along given dimensions
        *  biasCorrected -  if true bias correction will be applied
        *  dimensions - vector of dimensions to calculate variance along
        */
        template<typename OpName>
        NDArray<T>* varianceAlongDimension(const bool biasCorrected, const std::vector<int>& dimensions) const;
        template<typename OpName>
        NDArray<T>* varianceAlongDimension(const bool biasCorrected, const std::initializer_list<int>& dimensions) const;

        /**
        *  operator returns sub-array with buffer pointing at this->_buffer with offset defined by given intervals
        *  idx - intervals of indexes which define the sub-arrays  to point on
        */
        NDArray<T> operator()(const Intervals& idx)  const;

        /**
        *  addition operator: array + other
        *  other - input array to add
        */
        NDArray<T> operator+(const NDArray<T>& other) const;

        /**
        *  addition operator: array + scalar
        *  scalar - input scalar to add
        */
        NDArray<T> operator+(const T scalar) const;

        /**
        *  friend functions which implement addition operator: scalar + array
        *  scalar - input scalar to add
        */
        friend NDArray<float> nd4j::operator+(const float scalar, const NDArray<float>& arr);
        friend NDArray<float16> nd4j::operator+(const float16 scalar, const NDArray<float16>& arr);
        friend NDArray<double> nd4j::operator+(const double scalar, const NDArray<double>& arr);
        
        /**
        *  addition unary operator array += other
        *  other - input array to add
        */
        void operator+=(const NDArray<T>& other);

        void operator+=(const T other);
        void operator-=(const T other);
        
        /**
        *  subtraction operator: array - other
        *  other - input array to subtract
        */
        NDArray<T> operator-(const NDArray<T>& other) const;
        
        /**
        *  subtraction operator: array - scalar
        *  scalar - input scalar to subtract
        */
        NDArray<T> operator-(const T& scalar) const;        

        /**
        *  negative operator, it changes sign of all array elements on opposite
        */
        NDArray<T> operator-() const;

        /**
        *  friend functions which implement subtraction operator: scalar - array
        *  scalar - input scalar to subtract
        */
        friend NDArray<float> nd4j::operator-(const float scalar, const NDArray<float>& arr);
        friend NDArray<float16> nd4j::operator-(const float16 scalar, const NDArray<float16>& arr);
        friend NDArray<double> nd4j::operator-(const double scalar, const NDArray<double>& arr);

        /**
        *  pairwise multiplication operator: array * other
        *  other - input array to multiply on
        */
        NDArray<T> operator*(const NDArray<T>& other) const;        
    
        /**
        *  multiplication operator: array * scalar
        *  scalar - input scalar to multiply on
        */
        NDArray<T> operator*(const T scalar) const;
        
        /**
        *  pairwise multiplication unary operator array *= other
        *  other - input array to multiply on
        */
        void operator*=(const NDArray<T>& other);

        /**
        *  multiplication unary operator array *= scalar
        *  scalar - input scalar to multiply on
        */
        void operator*=(const T scalar);

        /**
        *  pairwise division operator: array / other
        *  other - input array to divide on
        */
        NDArray<T> operator/(const NDArray<T>& other) const;        

        /**
        *  division operator: array / scalar
        *  scalar - input scalar to divide each array element on
        */
        NDArray<T> operator/(const T scalar) const;

        /**
        *  pairwise division unary operator: array /= other
        *  other - input array to divide on
        */
        void operator/=(const NDArray<T>& other);

        /**
        *  division unary operator: array /= scalar
        *  scalar - input scalar to divide on
        */
        void operator/=(const T scalar);

        /**
        *  friend function which implements mathematical multiplication of two arrays
        *  left - input array
        *  right - input array
        */
        friend NDArray<T> mmul<>(const NDArray<T>& left, const NDArray<T>& right);

        /**
        *  this method assigns elements of other array to the sub-array of this array defined be given intervals
        *  other - input array to assign elements from
        *  idx - intervals of indexes which define the sub-array
        */ 
        void assign(const NDArray<T>& other, const Intervals& idx);

        /**
        *  return vector containing _buffer as flat binary array
        */
        std::vector<int8_t> asByteVector();

        /**
        *  makes array to be identity matrix (not necessarily square), that is set all diagonal elements = 1, rest = 0
        */
        void setIdentity();

        /**
        *  default destructor
        */        
        ~NDArray(); 

        /**
        *  set _shapeInfo
        */
        FORCEINLINE void setShapeInfo(int *shapeInfo);

        /**
        *  set _buffer
        */
        FORCEINLINE void setBuffer(T* buffer);

        /**
        *  set _isBuffAlloc and _isShapeAlloc
        */
        FORCEINLINE void triggerAllocationFlag(bool bufferAllocated, bool shapeAllocated);
        
        /**
        *  returns the value of "dim" dimension 
        */ 
        int sizeAt(int dim) const;

        /**        
        *  returns order of array
        */
        FORCEINLINE char ordering() const;

        /**
        *  return _isView
        */ 
        FORCEINLINE bool isView();

        /**
        *  returns shape portion of shapeInfo
        */
        FORCEINLINE int *shapeOf() const;
        
        /**
        *  returns strides portion of shapeInfo
        */
        FORCEINLINE int* stridesOf() const;

        /**
        *  returns rank of array
        */
        FORCEINLINE int rankOf() const;        

        /** 
        *  returns length of array
        */
        FORCEINLINE Nd4jIndex lengthOf() const;        

        /**
        *  returns number of rows in array
        */
        FORCEINLINE int rows() const;

        /**
        *  returns number of columns in array
        */ 
        FORCEINLINE int columns() const;

        /**
        *  returns size of array elements type
        */ 
        FORCEINLINE int sizeOfT() const;

        /**
        *  returns element-wise-stride
        */ 
        FORCEINLINE int ews() const;

        // returns true if arrays have same shape
        FORCEINLINE bool isSameShape(const NDArray<T> *other) const;
        FORCEINLINE bool isSameShape(const std::initializer_list<int>& shape) const;
        FORCEINLINE bool isSameShape(const std::initializer_list<Nd4jIndex>& shape) const;
        FORCEINLINE bool isSameShape(const std::vector<int>& shape) const;
        FORCEINLINE bool isSameShape(const std::vector<Nd4jIndex >& shape) const;

        /**
        *  returns true if these two NDArrays have same rank, dimensions, strides, ews and order
        */
        FORCEINLINE bool isSameShapeStrict(const NDArray<T> *other) const;

        /**
        *  returns true if buffer && shapeInfo were defined (non nullptr)
        */
        FORCEINLINE bool nonNull() const;

        /** 
        *  returns array element with given index from linear buffer
        *  i - element index in array
        */
        FORCEINLINE T getScalar(const Nd4jIndex i) const;

        /** 
        *  returns array element with given index, takes into account offset between elements (element-wise-stride)
        *  i - element index in array
        */
        FORCEINLINE T getIndexedScalar(const Nd4jIndex i) const;
        
        /** 
        *  returns element with given indexes from 2D array 
        *  i - number of row 
        *  j - number of column
        */
        FORCEINLINE T getScalar(const int i, const int j) const;

        /** 
        *  returns element with given indexes from 3D array 
        *  i - height
        *  j - width
        *  k - depth
        */
        FORCEINLINE T getScalar(const int i, const int j, const int k) const;        
        
        /** 
        *  assigns given scalar to array element by given index, takes into account offset between elements (element-wise-stride)
        *  i - element index in array
        *  value - scalar value to assign
        */
        FORCEINLINE void putIndexedScalar(const Nd4jIndex i, const T value);        

        /** 
        *  assigns given scalar to array element by given index, regards array buffer as linear
        *  i - element index in array
        *  value - scalar value to assign
        */
        FORCEINLINE void putScalar(const Nd4jIndex i, const T value);        

        /** 
        *  assigns given scalar to 2D array element by given indexes
        *  i - number of row
        *  j - number of row
        *  value - scalar value to assign
        */
        FORCEINLINE void putScalar(const int i, const int j, const T value);        

        /** 
        *  assigns given scalar to 3D array element by given indexes
        *  i - height
        *  j - width
        *  k - depth
        *  value - scalar value to assign
        */
        FORCEINLINE void putScalar(const int i, const int j, const int k, const T value);

        /**
        *  returns true if array is 2D
        */
        FORCEINLINE bool isMatrix() const;

        /**
        *  returns true if array is vector
        */
        FORCEINLINE bool isVector() const;

        /**
        *  returns true if array is column vector
        */
        FORCEINLINE bool isColumnVector() const;

        /**
        *  returns true if array is row vector
        */
        FORCEINLINE bool isRowVector() const;

        /**
        *  returns true if array is scalar
        */
        FORCEINLINE bool isScalar() const;

        /**
        *  inline accessing operator for matrix, i - absolute index        
        */
        FORCEINLINE T operator()(const Nd4jIndex i) const;

        /**
        *  inline modifying operator for matrix, i - absolute index        
        */
        FORCEINLINE T& operator()(const Nd4jIndex i);

        /**
        *  inline accessing operator for 2D array, i - row, j - column
        */
        FORCEINLINE T operator()(const int i, const int j) const;        

        /**
        *  inline modifying operator for 2D array, i - row, j - column
        */
        FORCEINLINE T& operator()(const int i, const int j);

        /**
        *  inline accessing operator for 3D array, i - height, j - width, k - depth
        */
        FORCEINLINE T operator()(const int i, const int j, const int k) const;        

        /**
        *  inline modifying operator for 3D array, i - height, j - width, k - depth
        */ 
        FORCEINLINE T& operator()(const int i, const int j, const int k);


    };




//////////////////////////////////////////////////////////////////////////
///// IMLEMENTATION OF INLINE METHODS ///// 
//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE void NDArray<T>::setShapeInfo(int *shapeInfo) {
    if(_isShapeAlloc && _workspace == nullptr)
        delete []_shapeInfo;

    _shapeInfo = shapeInfo;
    _isShapeAlloc = false;
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE void NDArray<T>::setBuffer(T* buffer) {
    if(_isBuffAlloc && _workspace == nullptr)
        delete []_buffer;
 
    _buffer = buffer;
    _isBuffAlloc = false;
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE void NDArray<T>::triggerAllocationFlag(bool bufferAllocated, bool shapeAllocated) {
  
    _isBuffAlloc = bufferAllocated;
    _isShapeAlloc = shapeAllocated;
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE char NDArray<T>::ordering() const {

    return shape::order(_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE bool NDArray<T>::isView() {

    return _isView;
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE int* NDArray<T>::shapeOf() const {
    
    return shape::shapeOf(_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE int* NDArray<T>::stridesOf() const {
    
    return shape::stride(_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE int NDArray<T>::rankOf() const {

    return shape::rank(_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE Nd4jIndex NDArray<T>::lengthOf() const {
    
    return shape::length(_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE int NDArray<T>::rows() const {
    
    return shapeOf()[0];
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE int NDArray<T>::columns() const {

    return shapeOf()[1];
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE int NDArray<T>::sizeOfT() const {
    
    return sizeof(T);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE int NDArray<T>::ews() const {

    return shape::elementWiseStride(_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE bool NDArray<T>::nonNull() const {
    
    return this->_buffer != nullptr && this->_shapeInfo != nullptr;
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE bool NDArray<T>::isMatrix() const {

    return shape::isMatrix(this->_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE bool NDArray<T>::isVector() const {
            
    return !isScalar() && shape::isVector(this->_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE bool NDArray<T>::isColumnVector() const {
   
    return !isScalar() && shape::isColumnVector(this->_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE bool NDArray<T>::isRowVector() const {
    // 1D edge case
    if (shape::rank(this->_shapeInfo) == 1)
        return true;

    return !isScalar() && shape::isRowVector(this->_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE bool NDArray<T>::isScalar() const {
    
    return shape::isScalar(this->_shapeInfo);
}

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
FORCEINLINE T NDArray<T>::getIndexedScalar(const Nd4jIndex i) const { 
    return (*this)(i); 
}

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
FORCEINLINE bool NDArray<T>::isSameShape(const std::initializer_list<int>& other) const {
    std::vector<int> shp(other);
    return isSameShape(shp);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE bool NDArray<T>::isSameShape(const std::initializer_list<Nd4jIndex>& other) const {
    std::vector<Nd4jIndex> shp(other);
    return isSameShape(shp);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE bool NDArray<T>::isSameShape(const std::vector<Nd4jIndex>& other) const {
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
FORCEINLINE bool NDArray<T>::isSameShape(const std::vector<int>& other) const{
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
