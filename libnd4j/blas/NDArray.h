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
#include <stdint.h>


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
        Nd4jLong *_shapeInfo = nullptr;

        /**
        *  pointer on externally allocated memory where _buffer and _shapeInfo are stored
        */  
        nd4j::memory::Workspace* _workspace = nullptr;
        
        /**
        *  alternative buffers for special computational devices (like GPUs for CUDA)
        */  
        T* _bufferD = nullptr;
        Nd4jLong *_shapeInfoD = nullptr;

        /**
        *  indicates whether user allocates memory for _buffer/_shapeInfo by himself, in opposite case the memory must be allocated from outside
        */  
        bool _isShapeAlloc = false;                    
        bool _isBuffAlloc = false; 						

        /**
        *  type of array elements
        */  
        DataType _dataType = DataType_FLOAT;

        std::string toStringValue(T value);
    public:        
        
        /**
        *  default constructor, do not allocate memory, memory for array is passed from outside 
        */
        NDArray(T *buffer = nullptr, Nd4jLong* shapeInfo = nullptr, nd4j::memory::Workspace* workspace = nullptr);

        NDArray(std::initializer_list<Nd4jLong> shape, nd4j::memory::Workspace* workspace = nullptr);

        
        /**
         * Constructor for scalar NDArray
         */
        NDArray(T scalar);

        /**
        *  copy constructor
        */
        NDArray(const NDArray<T>& other);

        /**
        *  move constructor
        */
        NDArray(NDArray<T>&& other) noexcept;

#ifndef __JAVACPP_HACK__
        // this method only available out of javacpp
        /**
         * This constructor creates vector of T
         *
         * @param values
         */
        NDArray(std::initializer_list<T> values, nd4j::memory::Workspace* workspace = nullptr);
        NDArray(std::vector<T> &values, nd4j::memory::Workspace* workspace = nullptr);
#endif

        /**
        *  constructor, create empty array stored at given workspace
        */
        NDArray(nd4j::memory::Workspace* workspace);

        /**
        *  this constructor creates new NDArray with shape matching "other" array, do not copy "other" elements into new array
        */
        NDArray(const NDArray<T> *other, const bool copyStrides = false, nd4j::memory::Workspace* workspace = nullptr);
				
        /**
		*  constructor creates new NDArray using shape information from "shapeInfo", set all elements in new array to be zeros, if copyStrides is true then use stride values from "shapeInfo", else calculate strides independently 
        */
		NDArray(const Nd4jLong* shapeInfo, const bool copyStrides = false, nd4j::memory::Workspace* workspace = nullptr);

        /**
        *  this constructor creates new array using shape information contained in vector argument    
        */
        NDArray(const char order, const std::vector<Nd4jLong> &shape, nd4j::memory::Workspace* workspace = nullptr);

        /**
        * This constructor creates new array with elements copied from data and using shape information stored in shape
        *
        * PLEASE NOTE: data will be copied AS IS, without respect to specified order. You must ensure order match here.
        */
        NDArray(const char order, const std::vector<Nd4jLong> &shape, const std::vector<T> &data, nd4j::memory::Workspace* workspace = nullptr);

        /**
        *  this constructor creates new array using given buffer (without memory allocating) and shape information stored in shape
        */
        NDArray(T *buffer, const char order, const std::vector<Nd4jLong> &shape , nd4j::memory::Workspace* workspace = nullptr);

        /**
        *  copy assignment operator
        */
        NDArray<T>& operator=(const NDArray<T>& other);

        /**
        *  move assignment operator
        */
        NDArray<T>& operator=(NDArray<T>&& other) noexcept;

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
        void replacePointers(T *buffer, Nd4jLong *shapeInfo, const bool releaseExisting = true);
 
        /**
        *  create a new array by replicating current array by repeats times along given dimension
        *  dimension - dimension along which to repeat elements
        *  repeats - number of repetitions
        */        
        NDArray<T>* repeat(int dimension, const std::vector<Nd4jLong>& repeats) const;

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
        NDArray<T> *subarray(IndicesList& indices, std::vector<Nd4jLong>& strides) const;
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
        Nd4jLong* shapeInfo();
        Nd4jLong* getShapeInfo() const;

        /**
        *  if _bufferD==nullptr return _buffer, else return _bufferD
        */
        T* specialBuffer();

        /**
        *  if _shapeInfoD==nullptr return _shapeInfo, else return _shapeInfoD
        */
        Nd4jLong* specialShapeInfo();

        /**
        *  set values for _bufferD and _shapeInfoD
        */
        void setSpecialBuffers(T * buffer, Nd4jLong *shape);

        /**
        *  permutes (in-place) the dimensions in array according to "dimensions" array
        */
        bool permutei(const std::initializer_list<int>& dimensions);
        bool permutei(const std::vector<int>& dimensions);
        bool permutei(const int* dimensions, const int rank);

        bool permutei(const std::initializer_list<Nd4jLong>& dimensions);
        bool permutei(const std::vector<Nd4jLong>& dimensions);
        bool permutei(const Nd4jLong* dimensions, const int rank);

        /**
        *  permutes the dimensions in array according to "dimensions" array, new array points on _buffer of this array
        */
		NDArray<T>* permute(const std::initializer_list<int>& dimensions) const;
        NDArray<T>* permute(const std::vector<int>& dimensions) const;
        NDArray<T>* permute(const int* dimensions, const int rank) const;

        void permute(const int* dimensions, const int rank, NDArray<T>& target) const;
        void permute(const std::vector<int>& dimensions, NDArray<T>& target) const;

        NDArray<T>* permute(const std::initializer_list<Nd4jLong>& dimensions) const;
        NDArray<T>* permute(const std::vector<Nd4jLong>& dimensions) const;
        NDArray<T>* permute(const Nd4jLong* dimensions, const int rank) const;

        void permute(const Nd4jLong* dimensions, const int rank, NDArray<T>& target) const;
        void permute(const std::vector<Nd4jLong>& dimensions, NDArray<T>& target) const;

        /**
         * This method streamlines given view or permuted array, and reallocates buffer
         */
        void streamline(char order = 'a');



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
        void printBuffer(const char* msg = nullptr, Nd4jLong limit = -1);

        /**
        *  prints buffer elements, takes into account offset between elements (element-wise-stride)
        *  msg - message to print out 
        *  limit - number of array elements to print out
        */ 
        void printIndexedBuffer(const char* msg = nullptr, Nd4jLong limit = -1) const;

        std::string asIndexedString(Nd4jLong limit = -1);
        std::string asString(Nd4jLong limit = -1);

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
         * This method explicitly enforces new shape for this NDArray, old shape/stride information is lost
         */
        void enforce(const std::initializer_list<Nd4jLong> &dimensions, char order = 'a');
        void enforce(std::vector<Nd4jLong> &dimensions, char order = 'a');

        /**
        *  calculates sum along dimension(s) in this array and save it to created reduced array
        *  dimensions - array of dimensions to calculate sum over
        *  keepDims - if true then put unities in place of reduced dimensions
        */
        NDArray<T> *sum(const std::vector<int> &dimensions) const;

		/**
        *  method reduces array by excluding its shape along dimensions present in given dimensions vector, result is stored in new array to be returned
        *  dimensions - array of dimensions to reduce along
        *  keepDims - if true then put unities in place of reduced dimensions
        */ 
        template<typename OpName>
        NDArray<T>* reduceAlongDimension(const std::vector<int>& dimensions, const bool keepDims = false, const bool supportOldShapes = false) const;
		template<typename OpName>
        NDArray<T>* reduceAlongDimension(const std::initializer_list<int>& dimensions, const bool keepDims = false, const bool supportOldShapes = false) const;
        template<typename OpName>
        NDArray<T> reduceAlongDims(const std::vector<int>& dimensions, const bool keepDims = false, const bool supportOldShapes = false) const;

        /**
        *  method reduces array by excluding its shapes along dimensions present in given dimensions vector
        *  target - where to save result of reducing
        *  dimensions - array of dimensions to reduce along
        *  keepDims - if true then put unities in place of reduced dimensions
        *  extras - extra parameters
        */ 
        template<typename OpName>
        void reduceAlongDimension(NDArray<T>* target, const std::vector<int>& dimensions, const bool keepDims = false, const bool supportOldShapes = false, T *extras = nullptr) const;

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
        Nd4jLong indexReduceNumber(T *extraParams = nullptr);

        /**
        *  returns index of max element in a given array (optionally: along given dimension(s))
        *  dimensions - optional vector with dimensions
        */          
        Nd4jLong argMax(std::initializer_list<int> dimensions = {});

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
        *  apply OpName transformation to this array and store result in new array being returned
        *  extraParams - extra parameters for operation
        */
        template<typename OpName>
        NDArray<T> transform(T *extraParams = nullptr) const;

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
        *  dimensions -  dimensions array to broadcast along
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
        void applyScalar(T scalar, NDArray<T>* target = nullptr, T *extraParams = nullptr) const;

        /** 
        *  apply a scalar operation to an array
        *  scalar - input array which is simple scalar
        *  target - where to store result
        *  extraParams - extra parameters for operation
        */ 
        template<typename OpName>
        void applyScalar(NDArray<T>& scalar, NDArray<T>* target = nullptr, T *extraParams = nullptr) const;


#ifndef __JAVACPP_HACK__
        /**
        *  apply operation "func" to an array
        *  func - what operation to apply
        *  target - where to store result
        */
        void applyLambda(const std::function<T(T)>& func, NDArray<T>* target = nullptr);

        void applyIndexedLambda(const std::function<T(Nd4jLong, T)>& func, NDArray<T>* target = nullptr);

        /** 
        *  apply pairwise operation "func" to an array
        *  other - input array
        *  func - what pairwise operation to apply
        *  target - where to store result
        */ 
        void applyPairwiseLambda(NDArray<T>* other, const std::function<T(T, T)>& func, NDArray<T>* target = nullptr);

        void applyIndexedPairwiseLambda(NDArray<T>* other, const std::function<T(Nd4jLong, T, T)>& func, NDArray<T>* target = nullptr);

        void applyTriplewiseLambda(NDArray<T>* second, NDArray<T> *third, const std::function<T(T, T, T)>& func, NDArray<T>* target = nullptr);
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
        NDArray<T>* tensorAlongDimension(Nd4jLong index, const std::initializer_list<int>& dimensions) const;
        NDArray<T>* tensorAlongDimension(Nd4jLong index, const std::vector<int>& dimensions) const;

        /**
        *  returns the number of arrays pointing on specified dimension(s)
        *  dimensions - array of dimensions to point on
        */
        Nd4jLong tensorsAlongDimension(const std::initializer_list<int> dimensions) const ;
        Nd4jLong tensorsAlongDimension(const std::vector<int>& dimensions) const ;

        /**
        *  returns true if elements of two arrays are equal to within given epsilon value
        *  other - input array to compare
        *  eps - epsilon, this value defines the precision of elements comparison
        */
        bool equalsTo(const NDArray<T> *other, T eps = (T) 1e-5f) const;
        bool equalsTo(NDArray<T> &other, T eps = (T) 1e-5f) const;
        
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
        Nd4jLong memoryFootprint();
        
        /**
        *  these methods suited for FlatBuffers use
        */
        std::vector<T> getBufferAsVector();
        std::vector<Nd4jLong> getShapeAsVector();
        std::vector<Nd4jLong> getShapeInfoAsVector();
        std::vector<int64_t> getShapeInfoAsFlatVector();
				
        /**
        *  set new order and shape in case of suitable array length (in-place operation)
        *  order - order to set
        *  shape - shape to set
        *
        *  if there was permute applied before or there are weird strides, then new buffer is allocated for array
        */
		bool reshapei(const char order, const std::initializer_list<Nd4jLong>& shape);
		bool reshapei(const char order, const std::vector<Nd4jLong>& shape);

        bool reshapei(const std::initializer_list<Nd4jLong>& shape);
		bool reshapei(const std::vector<Nd4jLong>& shape);
	
        /**
        *  creates new array with corresponding order and shape, new array will point on _buffer of this array
        *  order - order to set
        *  shape - shape to set
        *
        * if permute have been applied before or there are weird strides, then new buffer is allocated for new array
        */
		NDArray<T>* reshape(const char order, const std::vector<Nd4jLong>& shape) const;
		
        /**
        *  calculate strides and set given order
        *  order - order to set
        */
		void updateStrides(const char order);

        /**
        *  change an array by repeating it the number of times given by reps (in-place operation)
        *  repeats - contains numbers of repetitions
        */
		void tilei(const std::vector<Nd4jLong>& repeats);

        /**
        *  returns new array which is created by by repeating of this array the number of times given by reps 
        *  repeats - contains numbers of repetitions
        */
		NDArray<T> tile(const std::vector<Nd4jLong>& repeats) const;

        /**
        *  change an array by repeating it the number of times given by reps (in-place operation)
        *  repeats - contains numbers of repetitions
        *  target - where to store result
        */
        void tile(const std::vector<Nd4jLong>& repeats, NDArray<T>& target) const;

        /**
        *  change an array by repeating it the number of times to acquire the new shape which is the same as target shape        
        *  target - where to store result
        */
        void tile(NDArray<T>& target) const;
        
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


        template<typename OpName>
        void varianceAlongDimension(const NDArray<T>* target, const bool biasCorrected, const std::vector<int>& dimensions);
        template<typename OpName>
        void varianceAlongDimension(const NDArray<T>* target, const bool biasCorrected, const std::initializer_list<int>& dimensions);

        /**
        *  operator returns sub-array with buffer pointing at this->_buffer with offset defined by given intervals
        *  idx - intervals of indexes which define the sub-arrays to point on
        *  keepUnitiesInShape - if false then eliminate unities from resulting array shape, for example {1,a,1,b} -> {a,b}
        */
        NDArray<T> operator()(const Intervals& idx, bool keepUnitiesInShape = false)  const;

        /**
        *  operator returns sub-array with buffer pointing at this->_buffer with offset defined by given intervals
        *  idx - intervals of indexes which define the sub-arrays to point on, idx has form {dim0Start,dim0End,  dim1Start,dim1End, ....} and length (2 * this->rankOf())
        *        when (dimStart == dimEnd) then whole range will be used for current dimension
        *  keepUnitiesInShape - if false then eliminate unities from resulting array shape, for example {1,a,1,b} -> {a,b}
        */
        NDArray<T> operator()(const int* idx, bool keepUnitiesInShape = false)  const;

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

        /**
        *  subtraction unary operator array -= other
        *  other - input array to add
        */
        void operator-=(const NDArray<T>& other);

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
        *  this method assigns elements of other array to the sub-array of this array defined by given intervals
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
        *  swaps the contents of tow arrays, 
        *  PLEASE NOTE: method doesn't take into account the shapes of arrays, shapes may be different except one condition: arrays lengths must be the same 
        */
        void swapUnsafe(NDArray<T>& other);

        /**
        *  return vector with buffer which points on corresponding diagonal elements of array
        *  type - means of vector to be returned: column ('c') or row ('r')
        */
        NDArray<T>* diagonal(const char type ) const;

        /**
        *  fill matrix with given value starting from specified diagonal in given direction, works only with 2D matrix
        *
        *  diag - diagonal starting from matrix is filled. 
        *      diag = 0 corresponds to main diagonal, 
        *      diag < 0 below main diagonal
        *      diag > 0 above main diagonal
        *  direction - in what direction to fill matrix. There are 2 possible directions:
        *      'u' - fill up, mathematically this corresponds to lower triangular matrix 
        *      'l' - fill down, mathematically this corresponds to upper triangular matrix
        */
        void setValueInDiagMatrix(const T& value, const int diag, const char direction);

		/**
        *  change an array by repeating it the number of times in order to acquire new shape equal to the input shape
        *
        *  shape  - contains new shape to broadcast array to 
        *  target - optional argument, if target != nullptr the resulting array will be placed it target, in opposite case tile operation is done in place
        */
        void tileToShape(const std::vector<Nd4jLong>& shape, NDArray<T>* target = nullptr);
        void tileToShape(const std::initializer_list<Nd4jLong>& shape, NDArray<T>* target = nullptr);

        template <typename N>
        NDArray<N>* asT();

        /**
        *  calculates the trace of an array, that is sum of elements on main diagonal = sum array[i, i, i, ...]
        */
        T getTrace() const;
        
        /**
        *  default destructor
        */        
        ~NDArray() noexcept; 

        /**
        *  set _shapeInfo
        */
        FORCEINLINE void setShapeInfo(Nd4jLong *shapeInfo);

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
        Nd4jLong sizeAt(int dim) const;

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
        FORCEINLINE Nd4jLong* shapeOf() const;
        
        /**
        *  returns strides portion of shapeInfo
        */
        FORCEINLINE Nd4jLong* stridesOf() const;

        /**
        *  returns rank of array
        */
        FORCEINLINE int rankOf() const;        

        /** 
        *  returns length of array
        */
        FORCEINLINE Nd4jLong lengthOf() const;

        /**
        *  returns number of rows in array
        */
        FORCEINLINE Nd4jLong rows() const;

        /**
        *  returns number of columns in array
        */ 
        FORCEINLINE Nd4jLong columns() const;

        /**
        *  returns size of array elements type
        */ 
        FORCEINLINE int sizeOfT() const;

        /**
        *  returns element-wise-stride
        */ 
        FORCEINLINE Nd4jLong ews() const;

        // returns true if arrays have same shape
        FORCEINLINE bool isSameShape(const NDArray<T> *other) const;
        FORCEINLINE bool isSameShape(NDArray<T> &other) const;
        FORCEINLINE bool isSameShape(const std::initializer_list<Nd4jLong>& shape) const;
        FORCEINLINE bool isSameShape(const std::vector<Nd4jLong>& shape) const;

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
        FORCEINLINE T getScalar(const Nd4jLong i) const;

        /** 
        *  returns array element with given index, takes into account offset between elements (element-wise-stride)
        *  i - element index in array
        */
        FORCEINLINE T getIndexedScalar(const Nd4jLong i) const;
        
        /** 
        *  returns element with given indexes from 2D array 
        *  i - number of row 
        *  j - number of column
        */
        FORCEINLINE T getScalar(const Nd4jLong i, const Nd4jLong j) const;

        /** 
        *  returns element with given indexes from 3D array 
        *  i - height
        *  j - width
        *  k - depth
        */
        FORCEINLINE T getScalar(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k) const;
        
        /** 
        *  assigns given scalar to array element by given index, takes into account offset between elements (element-wise-stride)
        *  i - element index in array
        *  value - scalar value to assign
        */
        FORCEINLINE void putIndexedScalar(const Nd4jLong i, const T value);

        /** 
        *  assigns given scalar to array element by given index, regards array buffer as linear
        *  i - element index in array
        *  value - scalar value to assign
        */
        FORCEINLINE void putScalar(const Nd4jLong i, const T value);

        /** 
        *  assigns given scalar to 2D array element by given indexes
        *  i - number of row
        *  j - number of row
        *  value - scalar value to assign
        */
        FORCEINLINE void putScalar(const Nd4jLong i, const Nd4jLong j, const T value);

        /** 
        *  assigns given scalar to 3D array element by given indexes
        *  i - height
        *  j - width
        *  k - depth
        *  value - scalar value to assign
        */
        FORCEINLINE void putScalar(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const T value);

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
        FORCEINLINE T operator()(const Nd4jLong i) const;

        /**
        *  inline modifying operator for matrix, i - absolute index        
        */
        FORCEINLINE T& operator()(const Nd4jLong i);

        /**
        *  inline accessing operator for 2D array, i - row, j - column
        */
        FORCEINLINE T operator()(const Nd4jLong i, const Nd4jLong j) const;

        /**
        *  inline modifying operator for 2D array, i - row, j - column
        */
        FORCEINLINE T& operator()(const Nd4jLong i, const Nd4jLong j);

        /**
        *  inline accessing operator for 3D array, i - height, j - width, k - depth
        */
        FORCEINLINE T operator()(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k) const;

        /**
        *  inline modifying operator for 3D array, i - height, j - width, k - depth
        */ 
        FORCEINLINE T& operator()(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k);

        /**
        *  inline modifying operator for 4D array, i - height, j - width, k - depth
        */ 
        FORCEINLINE T& operator()(const Nd4jLong t, const Nd4jLong u, const Nd4jLong v, const Nd4jLong w);

        /**
        *  inline accessing operator for 4D array, i - height, j - width, k - depth
        */
        FORCEINLINE T operator()(const Nd4jLong t, const Nd4jLong u, const Nd4jLong v, const Nd4jLong w) const;

        template <typename T2>
        FORCEINLINE std::vector<T2> asVectorT();


        FORCEINLINE bool isAttached();

        NDArray<T>* detach();
    };




//////////////////////////////////////////////////////////////////////////
///// IMLEMENTATION OF INLINE METHODS ///// 
//////////////////////////////////////////////////////////////////////////

template <typename T>
template <typename T2>
 std::vector<T2> NDArray<T>::asVectorT() {
    std::vector<T2> result(this->lengthOf());

#pragma omp parallel for simd
    for (int e = 0; e < this->lengthOf(); e++)
        result[e] = (T2) this->getIndexedScalar(e);

    return result;
}

template<typename T>
 bool NDArray<T>::isAttached() {
    return this->_workspace != nullptr;
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
 void NDArray<T>::setShapeInfo(Nd4jLong *shapeInfo) {
    if(_isShapeAlloc && _workspace == nullptr)
        delete []_shapeInfo;

    _shapeInfo = shapeInfo;
    _isShapeAlloc = false;
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
 void NDArray<T>::setBuffer(T* buffer) {
    if(_isBuffAlloc && _workspace == nullptr)
        delete []_buffer;
 
    _buffer = buffer;
    _isBuffAlloc = false;
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
 void NDArray<T>::triggerAllocationFlag(bool bufferAllocated, bool shapeAllocated) {
  
    _isBuffAlloc = bufferAllocated;
    _isShapeAlloc = shapeAllocated;
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
 char NDArray<T>::ordering() const {

    return shape::order(_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
 bool NDArray<T>::isView() {

    return _isView;
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
 Nd4jLong* NDArray<T>::shapeOf() const {
    
    return shape::shapeOf(_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
Nd4jLong* NDArray<T>::stridesOf() const {
    
    return shape::stride(_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
int NDArray<T>::rankOf() const {

    return shape::rank(_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
Nd4jLong NDArray<T>::lengthOf() const {
    
    return shape::length(_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
Nd4jLong NDArray<T>::rows() const {
    
    return shapeOf()[0];
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
Nd4jLong NDArray<T>::columns() const {

    return shapeOf()[1];
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
 int NDArray<T>::sizeOfT() const {
    
    return sizeof(T);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
Nd4jLong NDArray<T>::ews() const {

    return shape::elementWiseStride(_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
 bool NDArray<T>::nonNull() const {
    
    return this->_buffer != nullptr && this->_shapeInfo != nullptr;
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
 bool NDArray<T>::isMatrix() const {
    return shape::isMatrix(this->_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
 bool NDArray<T>::isVector() const {
    return !isScalar() && shape::isVector(this->_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
 bool NDArray<T>::isColumnVector() const {
   
    return !isScalar() && shape::isColumnVector(this->_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
 bool NDArray<T>::isRowVector() const {
    // 1D edge case
    if (shape::rank(this->_shapeInfo) == 1)
        return true;

    return !isScalar() && shape::isRowVector(this->_shapeInfo);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
 bool NDArray<T>::isScalar() const {
    
    return shape::isScalar(this->_shapeInfo);
}

// accessing operator for matrix, i - absolute index
template<typename T>
 T NDArray<T>::operator()(const Nd4jLong i) const {

    if (i >= shape::length(_shapeInfo))
            throw std::invalid_argument("NDArray::operator(i): dinput index is out of array length !");

    auto ews   = shape::elementWiseStride(_shapeInfo);
    char order = ordering();   

    if(ews == 1 && order == 'c')
        return _buffer[i];
    else if(ews > 1 && order == 'c')
        return _buffer[i*ews];
    else {
        Nd4jLong idx[MAX_RANK];
        shape::ind2subC(rankOf(), shapeOf(), i, idx);
        Nd4jLong offset = shape::getOffset(0, shapeOf(), stridesOf(), idx, rankOf());
        return _buffer[offset];        
    }
}

//////////////////////////////////////////////////////////////////////////
// modifying operator for matrix, i - absolute index
template<typename T>
 T& NDArray<T>::operator()(const Nd4jLong i) {

    if (i >= shape::length(_shapeInfo))
            throw std::invalid_argument("NDArray::operator(i): input index is out of array length !");

    auto  ews   = shape::elementWiseStride(_shapeInfo);
    auto order = ordering();

    if(ews == 1 && order == 'c')
        return _buffer[i];
    else if(ews > 1 && order == 'c')
        return _buffer[i*ews];
    else {
        Nd4jLong idx[MAX_RANK];
        shape::ind2subC(rankOf(), shapeOf(), i, idx);
        auto offset = shape::getOffset(0, shapeOf(), stridesOf(), idx, rankOf());
        return _buffer[offset];
    }    
}

//////////////////////////////////////////////////////////////////////////
// accessing operator for 2D matrix, i - row, j - column
template<typename T>
 T NDArray<T>::operator()(const Nd4jLong i, const Nd4jLong j) const {
    
    if (rankOf() != 2 || i >= shapeOf()[0] || j >= shapeOf()[1])
       throw std::invalid_argument("NDArray::operator(i,j): one of input indexes is out of array length or rank!=2 !");
    
    Nd4jLong coords[2] = {i, j};
    auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
    return _buffer[xOffset];
}

//////////////////////////////////////////////////////////////////////////
// modifying operator for 2D matrix, i - row, j - column
template<typename T>
 T& NDArray<T>::operator()(const Nd4jLong  i, const Nd4jLong j) {
    
    if (rankOf() != 2 || i >= shapeOf()[0] || j >= shapeOf()[1])
       throw std::invalid_argument("NDArray::operator(i,j): one of input indexes is out of array length or rank!=2 !");

    Nd4jLong coords[2] = {i, j};
    auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
    return _buffer[xOffset];
}

//////////////////////////////////////////////////////////////////////////
// accessing operator for 3D array, i - row, j - column
template<typename T>
 T NDArray<T>::operator()(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k) const {
    
    if (rankOf() != 3 || i >= shapeOf()[0] || j >= shapeOf()[1] || j >= shapeOf()[2])
       throw std::invalid_argument("NDArray::operator(i,j,k): one of input indexes is out of array length or rank!=3 !");
    
    Nd4jLong coords[3] = {i, j, k};
    auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
    return _buffer[xOffset];
}

//////////////////////////////////////////////////////////////////////////
// modifying operator for 3D array
template<typename T>
 T& NDArray<T>::operator()(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k) {
    
    if (rankOf() != 3 || i >= shapeOf()[0] || j >= shapeOf()[1] || k >= shapeOf()[2])
       throw std::invalid_argument("NDArray::operator(i,j,k): one of input indexes is out of array length or rank!=3 !");

    Nd4jLong coords[3] = {i, j, k};
    auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
    return _buffer[xOffset];
}

template<typename T>
 T NDArray<T>::operator()(const Nd4jLong t, const Nd4jLong u, const Nd4jLong v, const Nd4jLong w) const {
    
    if (rankOf() != 4 || t >= shapeOf()[0] || u >= shapeOf()[1] || v >= shapeOf()[2] || w >= shapeOf()[3])
       throw std::invalid_argument("NDArray::operator(t,u,v,w): one of input indexes is out of array length or rank!=4 !");

    Nd4jLong coords[4] = {t, u, v, w};
    auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
    return _buffer[xOffset];
}

template<typename T>
 T& NDArray<T>::operator()(const Nd4jLong t, const Nd4jLong u, const Nd4jLong v, const Nd4jLong w) {
    
    if (rankOf() != 4 || t >= shapeOf()[0] || u >= shapeOf()[1] || v >= shapeOf()[2] || w >= shapeOf()[3])
       throw std::invalid_argument("NDArray::operator(t,u,v,w): one of input indexes is out of array length or rank!=4 !");

    Nd4jLong coords[4] = {t, u, v, w};
    auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
    return _buffer[xOffset];
}

//////////////////////////////////////////////////////////////////////////
// Return value from linear buffer
template<typename T>
 T NDArray<T>::getScalar(const Nd4jLong i) const
{ return (*this)(i); }

//////////////////////////////////////////////////////////////////////////
template<typename T>
 T NDArray<T>::getIndexedScalar(const Nd4jLong i) const {
    return (*this)(i); 
}

//////////////////////////////////////////////////////////////////////////
// Returns value from 2D matrix by coordinates/indexes         
template<typename T>
 T NDArray<T>::getScalar(const Nd4jLong i, const Nd4jLong j) const
{ return (*this)(i, j); }

//////////////////////////////////////////////////////////////////////////
// returns value from 3D tensor by coordinates        
template<typename T>
 T NDArray<T>::getScalar(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k) const
{ return (*this)(i, j, k); }

//////////////////////////////////////////////////////////////////////////
template<typename T>
 void NDArray<T>::putIndexedScalar(const Nd4jLong i, const T value)
{ (*this)(i) = value; }

//////////////////////////////////////////////////////////////////////////
// This method sets value in linear buffer to position i        
template<typename T>
    void NDArray<T>::putScalar(const Nd4jLong i, const T value)
{ (*this)(i) = value; }

//////////////////////////////////////////////////////////////////////////
// This method sets value in 2D matrix to position i, j         
template<typename T>
 void NDArray<T>::putScalar(const Nd4jLong i, const Nd4jLong j, const T value)
{ (*this)(i,j) = value; }

//////////////////////////////////////////////////////////////////////////
// This method sets value in 3D matrix to position i,j,k        
template<typename T>
 void NDArray<T>::putScalar(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const T value)
{ (*this)(i,j,k) = value; }

//////////////////////////////////////////////////////////////////////////
template<typename T>
Nd4jLong  NDArray<T>::memoryFootprint() {

    Nd4jLong size = this->lengthOf() * this->sizeOfT();
    size += shape::shapeInfoByteLength(this->rankOf());
    return size;
}

//////////////////////////////////////////////////////////////////////////
// returns true if these two NDArrays have same shape
// still the definition of inline function must be in header file
template<typename T>
 bool NDArray<T>::isSameShape(const std::vector<Nd4jLong>& other) const{
    if (this->rankOf() != (int) other.size())
        return false;
    for (int e = 0; e < this->rankOf(); e++) {
        if (this->shapeOf()[e] != other.at(e) && other.at(e) != -1)
            return false;
    }
    return true;
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
 bool NDArray<T>::isSameShape(const NDArray<T> *other) const {
    return isSameShape(std::vector<Nd4jLong>(other->_shapeInfo+1, other->_shapeInfo+1+other->_shapeInfo[0]));
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
 bool NDArray<T>::isSameShape(NDArray<T> &other) const {
    return isSameShape(&other);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
 bool NDArray<T>::isSameShape(const std::initializer_list<Nd4jLong>& other) const {
    
    return isSameShape(std::vector<Nd4jLong>(other));
}

//////////////////////////////////////////////////////////////////////////
// returns true if these two NDArrays have same _shapeInfo
// still the definition of inline function must be in header file
template<typename T>
bool NDArray<T>::isSameShapeStrict(const NDArray<T> *other) const {
  return shape::equalsStrict(_shapeInfo, other->_shapeInfo);
}



}
#endif
