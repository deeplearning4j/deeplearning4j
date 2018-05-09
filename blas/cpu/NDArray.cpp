#ifndef NDARRAY_CPP
#define NDARRAY_CPP

#include "../NDArray.h"
#include "../NativeOpExcutioner.h"
#include "../NDArrayFactory.h"
#include <memory/Workspace.h>
#include <memory/MemoryRegistrator.h>
#include <ops.h>
#include <ops/gemm.h>
#include <pointercast.h>
#include <stdexcept>
#include <memory>
#include <helpers/logger.h>
#include <loops/pairwise_transform.h>
#include <loops/transform.h>
#include <loops/random.h>
#include <loops/broadcasting.h>
#include <indexing/NDIndex.h>
#include <indexing/IndicesList.h>
#include <helpers/ShapeUtils.h>
#include <sstream>

namespace nd4j {

    template<typename T>
    void* NDArray<T>::operator new(size_t i) {
        if (nd4j::memory::MemoryRegistrator::getInstance()->hasWorkspaceAttached()) {
            nd4j::memory::Workspace* ws = nd4j::memory::MemoryRegistrator::getInstance()->getWorkspace();

            return ws->allocateBytes((Nd4jIndex) i);
        } else {
            auto p = malloc(i);
            
            CHECK_ALLOC(p, "Failed to allocate new NDArray");

            return p;
        }
    }

    template<typename T>
    void NDArray<T>::operator delete(void* p) {
        if (!nd4j::memory::MemoryRegistrator::getInstance()->hasWorkspaceAttached()) {
            free(p);
        }
    }

    template <typename T>
    NDArray<T>* NDArray<T>::getView() {
        auto view = new NDArray<T>();
        view->_isView = true;
        view->_shapeInfo = _shapeInfo;
        view->_buffer = _buffer;
        view->_workspace = _workspace;
        view->_isShapeAlloc = false;
        view->_isBuffAlloc = false;

        return view;
    }

    template <typename T>
    template <typename N>
    NDArray<N>* NDArray<T>::asT() {
        auto result = new NDArray<N>(this->ordering(), this->getShapeAsVector());
        auto l = this->lengthOf();

        // FIXME: we want to avoid put/get indexed scalars here really
#pragma omp parallel for
        for (int e = 0; e < l; e++) {
            result->putIndexedScalar(e, (N) this->getIndexedScalar(e));
        }

        return result;
    }

////////////////////////////////////////////////////////////////////////
// default constructor, do not allocate memory, memory for array is passed from outside 
    template <typename T>
    NDArray<T>::NDArray(T *buffer, int *shapeInfo, nd4j::memory::Workspace* workspace) {

        _buffer    = buffer;
        _shapeInfo = shapeInfo;
        _isBuffAlloc = false;                                  // indicate that memory for array is passed from outside
        _isShapeAlloc = false;
        _workspace = workspace;
    }

////////////////////////////////////////////////////////////////////////
//constructor, create empty array at given workspace
    template <typename T>
    NDArray<T>::NDArray(nd4j::memory::Workspace* workspace) {

        _buffer    = nullptr;
        _shapeInfo = nullptr;
        _isBuffAlloc = false;                                  // indicate that memory for array is passed from outside
        _isShapeAlloc = false;
        _workspace = workspace;

    }

////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray<T>::NDArray(std::initializer_list<int> s, nd4j::memory::Workspace* workspace) {
        std::vector<int> shape(s);
        int rank = (int) shape.size();


        ALLOCATE(_shapeInfo, workspace, shape::shapeInfoLength(rank), int);

        shape::shapeBuffer(rank, shape.data(), _shapeInfo);

        ALLOCATE(_buffer, workspace, shape::length(_shapeInfo), T);

        _isShapeAlloc = true;
        _isBuffAlloc = true;
        _workspace = workspace;
    }

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T>::NDArray(T scalar) {
    nd4j::memory::Workspace* workspace = nullptr;

    ALLOCATE(_buffer, workspace, 1, T);
    ALLOCATE(_shapeInfo, workspace, shape::shapeInfoLength(0), int);
    _shapeInfo[0] = 0;
    _shapeInfo[1] = 0;
    _shapeInfo[2] = 1;
    _shapeInfo[3] = 99;

    _buffer[0] = scalar;

    _isBuffAlloc = true; 
    _isShapeAlloc = true;
}

#ifndef __JAVACPP_HACK__
    template <typename T>
    NDArray<T>::NDArray(std::initializer_list<T> v, nd4j::memory::Workspace* workspace) {
        std::vector<T> values(v);
        ALLOCATE(_buffer, workspace, values.size(), T);
        ALLOCATE(_shapeInfo, workspace, shape::shapeInfoLength(1), int);
        shape::shapeVector(values.size(), _shapeInfo);
        memcpy(_buffer, values.data(), values.size() * sizeOfT());

        _isBuffAlloc = true;
        _isShapeAlloc = true;
        _workspace = workspace;
    }

    template <typename T>
    NDArray<T>::NDArray(std::vector<T> &values, nd4j::memory::Workspace* workspace) {
        ALLOCATE(_buffer, workspace, values.size(), T);
        ALLOCATE(_shapeInfo, workspace, shape::shapeInfoLength(1), int);
        shape::shapeVector(values.size(), _shapeInfo);
        memcpy(_buffer, values.data(), values.size() * sizeOfT());

        _isBuffAlloc = true;
        _isShapeAlloc = true;
        _workspace = workspace;
    }
#endif

////////////////////////////////////////////////////////////////////////
// creates new NDArray using shape information from "shapeInfo" array, set all elements in new array to be zeros
template <typename T>
    NDArray<T>::NDArray(const int* shapeInfo, const bool copyStrides, nd4j::memory::Workspace* workspace) {
   
    int arrLength = shape::length(const_cast<int*>(shapeInfo));
    int shapeLength = shape::shapeInfoLength(const_cast<int*>(shapeInfo));

    _workspace = workspace;
    if (workspace == nullptr) {
        _buffer =  new T[arrLength];
        _shapeInfo = new int[shapeLength];
    } else {
        _buffer = (T*) _workspace->allocateBytes(arrLength * sizeOfT());
        _shapeInfo = (int*) _workspace->allocateBytes(shape::shapeInfoByteLength(const_cast<int*>(shapeInfo)));
    }

    memset(_buffer, 0, arrLength*sizeOfT());          // set all elements in new array to be zeros

    memcpy(_shapeInfo, shapeInfo, shape::shapeInfoByteLength(const_cast<int*>(shapeInfo)));     // copy shape information into new array

    if(!copyStrides)
        shape::updateStrides(_shapeInfo, ordering());

    _isBuffAlloc = true;
    _isShapeAlloc = true;
}

    template<typename T>
    std::string NDArray<T>::toStringValue(T value) {
        std::ostringstream os ;

        //throw the value into the string stream
        os << value ;

        //convert the string stream into a string and return
        return os.str() ;
    }

    template<>
    std::string NDArray<float16>::toStringValue(float16 value) {
        std::ostringstream os ;

        //throw the value into the string stream
        os << (float) value ;

        //convert the string stream into a string and return
        return os.str() ;
    }

    template<typename T>
    std::string NDArray<T>::asIndexedString(int limit) {
        std::ostringstream os;
        os << "[";

        if (limit < 1 || limit > this->lengthOf())
            limit = this->lengthOf();

        for (int e = 0; e < limit; e++) {
            os << toStringValue(this->getIndexedScalar(e));

            if (e < limit - 1)
                os << ", ";
        }

        os << "]";

        return os.str();
    }

    template<typename T>
    std::string NDArray<T>::asString(int limit) {
        std::ostringstream os;
        os << "[";

        if (limit < 1 || limit > this->lengthOf())
            limit = this->lengthOf();

        for (int e = 0; e < limit; e++) {
            os << toStringValue(_buffer[e]);

            if (e < limit - 1)
                os << ", ";
        }

        os << "]";

        return os.str();
    }

////////////////////////////////////////////////////////////////////////
    template<typename T>
    std::vector<T> NDArray<T>::getBufferAsVector() {
        std::vector<T> vector(this->lengthOf());

        for (int e = 0; e < this->lengthOf(); e++) {
            vector[e] = this->getScalar(e);
        }

        return vector;
    }

////////////////////////////////////////////////////////////////////////
    template<typename T>
    std::vector<int> NDArray<T>::getShapeAsVector() {
        std::vector<int> vector(this->rankOf());

        for (int e = 0; e < this->rankOf(); e++)
            vector[e] = this->sizeAt(e);

        return vector;
    }

////////////////////////////////////////////////////////////////////////
    template<typename T>
    std::vector<int> NDArray<T>::getShapeInfoAsVector() {
        std::vector<int> vector;

        int magicNumber = shape::shapeInfoLength(this->rankOf());
        for (int e = 0; e < magicNumber; e++)
            vector.push_back(this->_shapeInfo[e]);

        return vector;
    }

////////////////////////////////////////////////////////////////////////
#ifndef __JAVACPP_HACK__
    template<typename T>
    void NDArray<T>::applyTriplewiseLambda(NDArray<T>* second, NDArray<T> *third, const std::function<T(T, T, T)>& func, NDArray<T>* target) {
        if (target == nullptr)
            target = this;

        if (second == nullptr) {
            nd4j_printf("applyTriplewiseLambda requires three operands to be valid NDArrays, but Second is NULL\n","");
            throw "second is null";
        }

        if (third == nullptr) {
            nd4j_printf("applyTriplewiseLambda requires three operands to be valid NDArrays, but Third is NULL\n","");
            throw "third is null";
        }

        if (this->lengthOf() != second->lengthOf() || this->lengthOf() != third->lengthOf() || !this->isSameShape(second) || !this->isSameShape(third)) {
            nd4j_printf("applyPairwiseLambda requires both operands to have the same shape\n","");
            throw "Shapes mismach";
        }        

        if (this->ordering() == second->ordering() && this->ordering() == third->ordering()  && this->ordering() == target->ordering() && (this->ews() == 1 && target->ews() == 1) && this->ews() == second->ews() && this->ews() == third->ews()) {
#pragma omp parallel for simd schedule(guided)
            for (int e = 0; e < this->lengthOf(); e++)
                target->_buffer[e] = func(this->_buffer[e], second->_buffer[e], third->_buffer[e]);
        } else {
            int tCoord[MAX_RANK];
            int uCoord[MAX_RANK];
            int vCoord[MAX_RANK];
            int zCoord[MAX_RANK]; 

            #pragma omp parallel for schedule(guided) private(tCoord, uCoord, vCoord, zCoord)
            for (int e = 0; e < this->lengthOf(); e++) {
                shape::ind2subC(this->rankOf(), this->shapeOf(), e, tCoord);
                shape::ind2subC(second->rankOf(), second->shapeOf(), e, uCoord);
                shape::ind2subC(third->rankOf(), third->shapeOf(), e, vCoord);
                shape::ind2subC(target->rankOf(), target->shapeOf(), e, zCoord);

                Nd4jIndex tOffset = shape::getOffset(0, this->shapeOf(), this->stridesOf(), tCoord, this->rankOf());
                Nd4jIndex uOffset = shape::getOffset(0, second->shapeOf(), second->stridesOf(), uCoord, second->rankOf());
                Nd4jIndex vOffset = shape::getOffset(0, third->shapeOf(), third->stridesOf(), vCoord, third->rankOf());
                Nd4jIndex zOffset = shape::getOffset(0, target->shapeOf(), target->stridesOf(), zCoord, target->rankOf());

                target->_buffer[zOffset] = func(this->_buffer[tOffset], second->_buffer[uOffset], third->_buffer[vOffset]);
            }
        }
    }

    template<typename T>
    void NDArray<T>::applyPairwiseLambda(NDArray<T>* other, const std::function<T(T, T)>& func, NDArray<T>* target) {
        if (target == nullptr)
            target = this;

        if (other == nullptr) {
            nd4j_printf("applyPairwiseLambda requires both operands to be valid NDArrays, but Y is NULL\n","");
            throw "Other is null";
        }

        if (this->lengthOf() != other->lengthOf()) {
            nd4j_printf("applyPairwiseLambda requires both operands to have the same shape\n","");
            throw "Shapes mismach";
        }

        if (this->ordering() == other->ordering() && this->ordering() == target->ordering() && (this->ews() == 1 && target->ews() == 1) && this->ews() == other->ews()) {
#pragma omp parallel for simd schedule(guided)
            for (int e = 0; e < this->lengthOf(); e++)
                target->_buffer[e] = func(this->_buffer[e], other->_buffer[e]);
        } else {
            int xCoord[MAX_RANK];
            int yCoord[MAX_RANK];
            int zCoord[MAX_RANK];
#pragma omp parallel for schedule(guided) private(xCoord, yCoord, zCoord)
            for (int e = 0; e < this->lengthOf(); e++) {
                shape::ind2subC(this->rankOf(), this->shapeOf(), e, xCoord);
                shape::ind2subC(other->rankOf(), other->shapeOf(), e, yCoord);
                shape::ind2subC(target->rankOf(), target->shapeOf(), e, zCoord);

                Nd4jIndex xOffset = shape::getOffset(0, this->shapeOf(), this->stridesOf(), xCoord, this->rankOf());
                Nd4jIndex yOffset = shape::getOffset(0, other->shapeOf(), other->stridesOf(), yCoord, other->rankOf());
                Nd4jIndex zOffset = shape::getOffset(0, target->shapeOf(), target->stridesOf(), zCoord, target->rankOf());

                target->_buffer[zOffset] = func(this->_buffer[xOffset], other->_buffer[yOffset]);
            }
        }
    }

////////////////////////////////////////////////////////////////////////
    template<typename T>
    void NDArray<T>::applyLambda(const std::function<T(T)>& func, NDArray<T>* target) {
        if (target == nullptr)
            target = this;

        if (this->ordering() == target->ordering() && (this->ews() == 1 && target->ews() == 1)) {
#pragma omp parallel for simd schedule(guided)
            for (int e = 0; e < this->lengthOf(); e++)
                target->_buffer[e] = func(this->_buffer[e]);
        } else {
            int xCoord[MAX_RANK];
            int zCoord[MAX_RANK];
#pragma omp parallel for schedule(guided) private(xCoord, zCoord)
            for (int e = 0; e < this->lengthOf(); e++) {
                shape::ind2subC(this->rankOf(), this->shapeOf(), e, xCoord);
                shape::ind2subC(target->rankOf(), target->shapeOf(), e, zCoord);

                Nd4jIndex xOffset = shape::getOffset(0, this->shapeOf(), this->stridesOf(), xCoord, this->rankOf());
                Nd4jIndex zOffset = shape::getOffset(0, target->shapeOf(), target->stridesOf(), zCoord, target->rankOf());

                target->_buffer[zOffset] = func(this->_buffer[xOffset]);
            }
        }
    }

    template<typename T>
    void NDArray<T>::applyIndexedLambda(const std::function<T(Nd4jIndex, T)>& func, NDArray<T>* target) {
        if (target == nullptr)
            target = this;

        if (this->ordering() == target->ordering() && (this->ews() == 1 && target->ews() == 1)) {
#pragma omp parallel for simd schedule(guided)
            for (int e = 0; e < this->lengthOf(); e++)
                target->_buffer[e] = func((Nd4jIndex) e, this->_buffer[e]);
        } else {
            int xCoord[MAX_RANK];
            int zCoord[MAX_RANK];

#pragma omp parallel for schedule(guided) private(xCoord, zCoord)
            for (int e = 0; e < this->lengthOf(); e++) {
                shape::ind2subC(this->rankOf(), this->shapeOf(), e, xCoord);
                shape::ind2subC(target->rankOf(), target->shapeOf(), e, zCoord);

                Nd4jIndex xOffset = shape::getOffset(0, this->shapeOf(), this->stridesOf(), xCoord, this->rankOf());
                Nd4jIndex zOffset = shape::getOffset(0, target->shapeOf(), target->stridesOf(), zCoord, target->rankOf());

                target->_buffer[zOffset] = func((Nd4jIndex) e, this->_buffer[xOffset]);
            }
        }
    }

    template<typename T>
    void NDArray<T>::applyIndexedPairwiseLambda(NDArray<T>* other, const std::function<T(Nd4jIndex, T, T)>& func, NDArray<T>* target) {
        if (target == nullptr)
            target = this;

        if (other == nullptr) {
            nd4j_printf("applyIndexedPairwiseLambda requires both operands to be valid NDArrays, but Y is NULL\n","");
            throw "Other is null";
        }

        if (this->lengthOf() != other->lengthOf()) {
            nd4j_printf("applyIndexedPairwiseLambda requires both operands to have the same shape\n","");
            throw "Shapes mismach";
        }

        if (this->ordering() == other->ordering() && this->ordering() == target->ordering() && (this->ews() == 1 && target->ews() == 1) && this->ews() == other->ews()) {
#pragma omp parallel for simd schedule(guided)
            for (int e = 0; e < this->lengthOf(); e++)
                target->_buffer[e] = func((Nd4jIndex) e, this->_buffer[e], other->_buffer[e]);
        } else {
            int xCoord[MAX_RANK];
            int yCoord[MAX_RANK];
            int zCoord[MAX_RANK];

#pragma omp parallel for schedule(guided) private(xCoord, yCoord, zCoord)
            for (int e = 0; e < this->lengthOf(); e++) {
                shape::ind2subC(this->rankOf(), this->shapeOf(), e, xCoord);
                shape::ind2subC(other->rankOf(), other->shapeOf(), e, yCoord);
                shape::ind2subC(target->rankOf(), target->shapeOf(), e, zCoord);

                Nd4jIndex xOffset = shape::getOffset(0, this->shapeOf(), this->stridesOf(), xCoord, this->rankOf());
                Nd4jIndex yOffset = shape::getOffset(0, other->shapeOf(), other->stridesOf(), yCoord, other->rankOf());
                Nd4jIndex zOffset = shape::getOffset(0, target->shapeOf(), target->stridesOf(), zCoord, target->rankOf());

                target->_buffer[zOffset] = func((Nd4jIndex) e, this->_buffer[xOffset], other->_buffer[yOffset]);
            }
        }
    }
#endif

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T>::NDArray(const NDArray<T> *other, const bool copyStrides, nd4j::memory::Workspace* workspace) {
    int arrLength = shape::length(other->_shapeInfo);
    int shapeLength = shape::shapeInfoByteLength(other->_shapeInfo);

    _workspace = workspace;
    if (workspace == nullptr) {
        _buffer =  new T[arrLength];
        _shapeInfo = new int[shapeLength];
    } else {
        _buffer = (T*) _workspace->allocateBytes(arrLength * sizeOfT());
        _shapeInfo = (int*) _workspace->allocateBytes(shapeLength);
    }

    // FIXME: memcpy should be removed
    // memcpy(_buffer, other->_buffer, arrLength*sizeOfT());      // copy other._buffer information into new array

    memcpy(_shapeInfo, other->_shapeInfo, shapeLength);     // copy shape information into new array

    if(!copyStrides) 
        shape::updateStrides(_shapeInfo, ordering());

    _isBuffAlloc = true;
    _isShapeAlloc = true;
}

////////////////////////////////////////////////////////////////////////
    template <typename T>
    std::vector<int8_t> NDArray<T>::asByteVector() {
        std::vector<int8_t> result((unsigned long) this->lengthOf() * sizeOfT());

        if (this->isView()) {
            auto tmp = this->dup();

            memcpy(result.data(), tmp->_buffer, (unsigned long) tmp->lengthOf() * sizeOfT());

            delete tmp;
        } else {
            memcpy(result.data(), _buffer, (unsigned long) this->lengthOf() * sizeOfT());
        }

        return result;
    }

////////////////////////////////////////////////////////////////////////
// copy constructor
template <typename T>
NDArray<T>::NDArray(const NDArray<T>& other) {

    int arrLength = shape::length(other._shapeInfo);
    int shapeLength = shape::shapeInfoByteLength(other._shapeInfo);

    _workspace = other._workspace;
    if (_workspace == nullptr) {
        _buffer =  new T[arrLength];
        _shapeInfo = new int[shapeLength];
    } else {
        _buffer = (T*) _workspace->allocateBytes(arrLength * sizeOfT());
        _shapeInfo = (int*) _workspace->allocateBytes(shapeLength);
    }

    // memcpy(_buffer, other._buffer, arrLength*sizeOfT());      // copy other._buffer information into new array
    memcpy(_shapeInfo, other._shapeInfo, shapeLength);     // copy shape information into new array
    shape::updateStrides(_shapeInfo, other.ordering());

    _isBuffAlloc = true; 
    _isShapeAlloc = true;
    this->assign(&other);
}

////////////////////////////////////////////////////////////////////////
// move constructor
template <typename T>
NDArray<T>::NDArray(NDArray<T>&& other) noexcept {

    _isView       = other._isView;
    _buffer       = other._buffer; 
    _shapeInfo    = other._shapeInfo;
    _workspace    = other._workspace;
    _bufferD      = other._bufferD;
    _shapeInfoD   = other._shapeInfoD;
    _isShapeAlloc = other._isShapeAlloc;
    _isBuffAlloc  = other._isBuffAlloc;
    _dataType     = other._dataType;

    other._buffer = other._bufferD = nullptr;
    other._shapeInfo = other._shapeInfoD = nullptr;
}

////////////////////////////////////////////////////////////////////////
    template<typename T>
    T* NDArray<T>::getBuffer() {
        return _buffer;
    }

    template<typename T>
    T* NDArray<T>::buffer() {
        return _buffer;
    }

////////////////////////////////////////////////////////////////////////
    template<typename T>
    int* NDArray<T>::getShapeInfo() const{
        return _shapeInfo;
    }

    template<typename T>
    int* NDArray<T>::shapeInfo() {
        return _shapeInfo;
    }

////////////////////////////////////////////////////////////////////////
    template<typename T>
    T* NDArray<T>::specialBuffer() {
        if (_bufferD == nullptr)
            return _buffer;

        // FIXME: this should be fixed once CUDA backend added
        return _bufferD;
    }

////////////////////////////////////////////////////////////////////////
    template<typename T>
    int* NDArray<T>::specialShapeInfo() {
        if (_shapeInfoD == nullptr)
            return _shapeInfo;

        // FIXME: this should be fixed once CUDA backend added

        return _shapeInfoD;
    }

////////////////////////////////////////////////////////////////////////
    template<typename T>
    void NDArray<T>::setSpecialBuffers(T * buffer, int *shape) {
        _bufferD = buffer;
        _shapeInfoD = shape;
    }

////////////////////////////////////////////////////////////////////////
// assignment operator
template<typename T>
    NDArray<T>& NDArray<T>::operator=(const NDArray<T>& other) {
	if (this == &other) return *this;

    if (_shapeInfo != nullptr && _buffer != nullptr && shape::equalsSoft(_shapeInfo, other._shapeInfo))
        this->assign(&other);        
        // memcpy(_buffer, other._buffer, arrLength*sizeOfT());               // copy elements of other current array
    else {
        if(_isBuffAlloc && _workspace == nullptr)
            delete []_buffer;
        if(_isShapeAlloc && _workspace == nullptr)
            delete []_shapeInfo;

        int arrLength = other.lengthOf();
		int shapeLength = shape::shapeInfoLength(other.rankOf());

        ALLOCATE(_buffer, _workspace, arrLength, T);
        // memcpy(_buffer, other._buffer, arrLength*sizeOfT());               // copy elements of other current array
        ALLOCATE(_shapeInfo, _workspace, shapeLength, int);
        memcpy(_shapeInfo, other._shapeInfo, shapeLength*sizeof(int));     // copy shape information into new array

        shape::updateStrides(_shapeInfo, other.ordering());

        _isBuffAlloc = true;
        _isShapeAlloc = true;
        this->assign(&other);
    }

    return *this;
}

////////////////////////////////////////////////////////////////////////
// move assignment operator
template <typename T>
NDArray<T>& NDArray<T>::operator=(NDArray<T>&& other) noexcept {

    if (this == &other) 
        return *this;

    if(_isBuffAlloc && _workspace == nullptr)
        delete []_buffer;
    if(_isShapeAlloc && _workspace == nullptr)
        delete []_shapeInfo;

    _isView       = other._isView;
    _buffer       = other._buffer; 
    _shapeInfo    = other._shapeInfo;
    _workspace    = other._workspace;
    _bufferD      = other._bufferD;
    _shapeInfoD   = other._shapeInfoD;
    _isShapeAlloc = other._isShapeAlloc;
    _isBuffAlloc  = other._isBuffAlloc;
    _dataType     = other._dataType;

    other._buffer = other._bufferD = nullptr;
    other._shapeInfo = other._shapeInfoD = nullptr;

    return *this;
}

////////////////////////////////////////////////////////////////////////
template<typename T>
NDArray<T>& NDArray<T>::operator=(const T scalar) {

    this->assign(scalar);
    return *this;
}


template <typename T>
void NDArray<T>::replacePointers(T *buffer, int *shapeInfo, const bool releaseExisting ) {
    this->_buffer = buffer;
    this->_shapeInfo = shapeInfo;

    if (releaseExisting) {
        if (_isShapeAlloc && _workspace == nullptr)
            delete[] _shapeInfo;

        if (_isBuffAlloc && _workspace == nullptr)
            delete[] _buffer;
    }
}

    template<typename T>
    NDArray<T>::NDArray(const char order, const std::vector<int> &shape, const std::vector<T> &data, nd4j::memory::Workspace* workspace) {
        int rank = (int) shape.size();

        if (rank > MAX_RANK)
            throw std::invalid_argument("Rank of NDArray can't exceed 32");

        int shapeOf[MAX_RANK];
        int cnt = 0;

        for (auto &item: shape)
            shapeOf[cnt++] = item;

        _workspace = workspace;
        if (workspace == nullptr) {
            if (order == 'f')
                _shapeInfo = shape::shapeBufferFortran(rank, shapeOf);
            else
                _shapeInfo = shape::shapeBuffer(rank, shapeOf);

            _buffer =  new T[shape::length(_shapeInfo)];
        } else {
            _buffer = (T*) _workspace->allocateBytes(data.size() * sizeOfT());
            _shapeInfo = (int*) _workspace->allocateBytes(shape::shapeInfoByteLength(rank));
            if (order == 'f')
                shape::shapeBufferFortran(rank, shapeOf, _shapeInfo);
            else
                shape::shapeBuffer(rank, shapeOf, _shapeInfo);

            //_buffer = (T*) _workspace->allocateBytes(shape::length(_shapeInfo) * sizeOfT());

        }

        if (shape::length(_shapeInfo) != data.size()) {
            nd4j_printf("Data size [%i] doesn't match shape length [%i]\n", data.size(), shape::length(_shapeInfo));
            throw "Data size doesn't match shape";
        }

        //memset(_buffer, 0, sizeOfT() * shape::length(_shapeInfo));
        memcpy(_buffer, data.data(), sizeOfT() * shape::length(_shapeInfo));

		_isBuffAlloc = true;
		_isShapeAlloc = true;

        shape::updateStrides(_shapeInfo, order);
    }

    template<typename T>
    NDArray<T>::NDArray(const char order, const std::vector<int> &shape, nd4j::memory::Workspace* workspace) {

        int rank = (int) shape.size();

        if (rank > MAX_RANK)
            throw std::invalid_argument("Rank of NDArray can't exceed 32");

        int *shapeOf = new int[rank];
        int cnt = 0;

        for (auto &item: shape)
            shapeOf[cnt++] = item;

        _workspace = workspace;
        if (workspace == nullptr) {
            if (order == 'f')
                _shapeInfo = shape::shapeBufferFortran(rank, shapeOf);
            else
                _shapeInfo = shape::shapeBuffer(rank, shapeOf);

            _buffer =  new T[shape::length(_shapeInfo)];
        } else {
            _shapeInfo = (int*) _workspace->allocateBytes(shape::shapeInfoByteLength(rank));

            if (order == 'f')
                shape::shapeBufferFortran(rank, shapeOf, _shapeInfo);
            else
                shape::shapeBuffer(rank, shapeOf, _shapeInfo);

            _buffer = (T*) _workspace->allocateBytes(shape::length(_shapeInfo) * sizeOfT());
        }

        memset(_buffer, 0, sizeOfT() * shape::length(_shapeInfo));
        
		_isBuffAlloc = true; 
		_isShapeAlloc = true;

        shape::updateStrides(_shapeInfo, order);
        
        delete[] shapeOf;
    }

    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    NDArray<T>::NDArray(T* buffer, const char order, const std::vector<int> &shape, nd4j::memory::Workspace* workspace) {

        int rank = (int) shape.size();

        if (rank > MAX_RANK)
            throw std::invalid_argument("Rank of NDArray can't exceed 32");

        int *shapeOf = new int[rank];
        int cnt = 0;

        for (const auto& item: shape)
            shapeOf[cnt++] = item;

        _workspace = workspace;
        _buffer = buffer;

        if (workspace == nullptr) {
            if (order == 'f')
                _shapeInfo = shape::shapeBufferFortran(rank, shapeOf);
            else
                _shapeInfo = shape::shapeBuffer(rank, shapeOf);
        } else {
            _shapeInfo = (int*) _workspace->allocateBytes(shape::shapeInfoByteLength(rank));

            if (order == 'f')
                shape::shapeBufferFortran(rank, shapeOf, _shapeInfo);
            else
                shape::shapeBuffer(rank, shapeOf, _shapeInfo);
        }

        _isBuffAlloc = false;
        _isShapeAlloc = true;

        delete[] shapeOf;
    }

// This method assigns values of given NDArray to this one, wrt order
    template<typename T>
    void NDArray<T>::assign(const NDArray<T> *other) {
        if (this->isScalar() && other->isScalar()) {
            this ->_buffer[0] = other->_buffer[0];
            return;
        } else if (other->isScalar()) {
            this->assign(other->_buffer[0]);
            return;;
        }
        else if(other->isScalar()) {
            this->assign(other->_buffer[0]);
            return;
        }

        if (other->lengthOf() != lengthOf()) {
            auto shapeThis = ShapeUtils<T>::shapeAsString(this);
            auto shapeThat = ShapeUtils<T>::shapeAsString(other);
            nd4j_printf("Can't assign new value to the array: this shape %s; other shape: %s\n", shapeThis.c_str(), shapeThat.c_str());
            throw "Lengths of arrays are mismatched";
        }

        // memcpy is allowed only for same order && same ews (being equal to 1)
        if (ordering() == other->ordering() && shape::elementWiseStride(this->_shapeInfo) == 1 && shape::elementWiseStride(other->_shapeInfo) == 1) {
            memcpy(_buffer, other->_buffer, lengthOf() * sizeOfT());
        } else {
            // now we invoke dup pwt against target buffer
            NativeOpExcutioner<T>::execPairwiseTransform(1, _buffer, _shapeInfo, other->_buffer, other->_shapeInfo, _buffer, _shapeInfo, nullptr);
        }
    }

// This method assigns values of given NDArray to this one
    template<typename T>
    void NDArray<T>::assign(const NDArray<T>& other) {
        if (this->isScalar() && other.isScalar()) {
            this->_buffer[0] = other._buffer[0];
            return;
        } else if (other.isScalar()) {
            this->assign(other._buffer[0]);
            return;;
        }

        if (this == &other) 
            return;
        if (other.lengthOf() != lengthOf()) {
            auto shapeThis = ShapeUtils<T>::shapeAsString(this);
            auto shapeThat = ShapeUtils<T>::shapeAsString(&other);
            nd4j_printf("Can't assign new value to the array: this shape %s; other shape: %s\n", shapeThis.c_str(), shapeThat.c_str());
            throw "Lengths of arrays are mismatched";
        }

        // memcpy is allowed only for same order && same ews (being equal to 1)
        if (ordering() == other.ordering() && shape::elementWiseStride(_shapeInfo) == 1 && shape::elementWiseStride(other._shapeInfo) == 1) {
            
            memcpy(_buffer, other._buffer, lengthOf() * sizeOfT());
        } else {
            // now we invoke dup pwt against target buffer
            NativeOpExcutioner<T>::execPairwiseTransform(1, _buffer, _shapeInfo, other._buffer, other._shapeInfo, _buffer, _shapeInfo, nullptr);
        }
    }

// This method assigns given value to all elements in this NDArray
    template<typename T>
    void NDArray<T>::assign(const T value) {

        // just fire scalar
        NativeOpExcutioner<T>::execScalar(13, _buffer, _shapeInfo, _buffer, _shapeInfo, value, nullptr);
    }


    template<typename T>
    NDArray<T>* NDArray<T>::detach() {
        if (!isAttached())
            return this;

        Nd4jIndex newLength = shape::length(_shapeInfo);
        T* newBuffer;
        int* newShapeInfo;

        newBuffer = new T[newLength];

        if (this->ordering() == 'f')
            newShapeInfo = shape::shapeBufferFortran(rankOf(), shapeOf());
        else
            newShapeInfo = shape::shapeBuffer(rankOf(), shapeOf());


        auto result = new NDArray<T>(newBuffer, newShapeInfo, nullptr);
        result->_isBuffAlloc = true;
        result->_isShapeAlloc = true;

        result->assign(this);

        return result;
    }

////////////////////////////////////////////////////////////////////////
// This method returns new copy of this NDArray, optionally in different order
template <typename T>
    NDArray<T>* NDArray<T>::dup(const char newOrder) {
    // op
    Nd4jIndex newLength = shape::length(_shapeInfo);
    T* newBuffer;
    int* newShapeInfo;

    char order = newOrder;

    if (order == 'a')
        order = this->ordering();

    if (_workspace == nullptr) {
        newBuffer = new T[newLength];

        if (order == 'f')
            newShapeInfo = shape::shapeBufferFortran(rankOf(), shapeOf());
        else
            newShapeInfo = shape::shapeBuffer(rankOf(), shapeOf());

    } else {
        newBuffer = (T*) _workspace->allocateBytes(newLength * sizeOfT());
        newShapeInfo = (int*) _workspace->allocateBytes(shape::shapeInfoByteLength(this->rankOf()));

        if (order == 'f')
            shape::shapeBufferFortran(rankOf(), shapeOf(), newShapeInfo);
        else
            shape::shapeBuffer(rankOf(), shapeOf(), newShapeInfo);
    }
    // FIXME: we know that EWS is always 1 after dup() result
    newShapeInfo[rankOf() * 2 + 2] = 1;

    NDArray<T> *result = new NDArray<T>(newBuffer, newShapeInfo, _workspace);
    // this values should be set, to avoid memleak
    result->_isBuffAlloc = true;
    result->_isShapeAlloc = true;

    result->assign(this);

    return result;
}

    template<typename T>
    template<typename OpName>
    T NDArray<T>::varianceNumber(bool biasCorrected) {
        return functions::summarystats::SummaryStatsReduce<T>::template execScalar<OpName>(biasCorrected, this->getBuffer(), this->getShapeInfo(), nullptr);
    }

//////////////////////////////////////////////////////////////////////////
// This method returns sum of all elements of this NDArray
    template<typename T>
    T NDArray<T>::sumNumber() const {
        return NativeOpExcutioner<T>::execReduceScalar(1, _buffer, _shapeInfo, nullptr);
    }

//////////////////////////////////////////////////////////////////////////
// This method returns mean number of this NDArray
    template<typename T>
    T NDArray<T>::meanNumber() const {
        return NativeOpExcutioner<T>::execReduceScalar(0, _buffer, _shapeInfo, nullptr);
    }

//////////////////////////////////////////////////////////////////////////
// method calculates sum along dimension(s) in this array and save it to row: as new NDArray with dimensions 1xN
    template<typename T>
    NDArray<T> *NDArray<T>::sum(const std::vector<int> &dimensions) const {

        return reduceAlongDimension<simdOps::Sum<T>>(dimensions);
//    NativeOpExcutioner<T>::execReduce(1, _buffer, _shapeInfo, nullptr, result->_buffer, result->_shapeInfo, dims, dimensions.size(), tad->tadOnlyShapeInfo, tad->tadOffsets);
    }

//////////////////////////////////////////////////////////////////////////
    template<typename T>
    bool NDArray<T>::isContiguous() {
        Nd4jIndex z = 1;
        int d;
        for(d = this->rankOf() - 1; d >= 0; d--)  {
            if(this->sizeAt(d) != 1) {
                if(this->stridesOf()[d] == z)
                    z *= this->sizeAt(d);
                else
                    return false;
            }
        }
        return true;
    }

//////////////////////////////////////////////////////////////////////////
// eventually method reduces array by excluding its shapes along axes present in dimensions vector
    template<typename T>
    template<typename OpName>
    NDArray<T> *NDArray<T>::reduceAlongDimension(const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {
        
        std::vector<int> copy(dimensions);
        
        int* newShape = ShapeUtils<T>::evalReduceShapeInfo('c', copy, *this, keepDims, supportOldShapes, _workspace);
        NDArray<T>* result = new NDArray<T>(newShape, _workspace);
        RELEASE(newShape, _workspace);        
        
        if(rankOf() == copy.size())
            result->_buffer[0] = functions::reduce::ReduceFunction<T>::template execScalar<OpName>(_buffer, _shapeInfo, nullptr);        
        else {
            shape::TAD tad(_shapeInfo, copy.data(), copy.size());
            tad.createTadOnlyShapeInfo();
            tad.createOffsets();        
            
            functions::reduce::ReduceFunction<T>::template exec<OpName>(_buffer, _shapeInfo, nullptr, result->_buffer,
                                                                        result->_shapeInfo, copy.data(), copy.size(),
                                                                        tad.tadOnlyShapeInfo, tad.tadOffsets);       
        }
        
        return result;
    }

//////////////////////////////////////////////////////////////////////////
// eventually method reduces array by excluding its shapes along axes present in dimensions vector
    template<typename T>
    template<typename OpName>
    NDArray<T> NDArray<T>::reduceAlongDims(const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {
        
        std::vector<int> copy(dimensions);
        
        int* newShape = ShapeUtils<T>::evalReduceShapeInfo('c', copy, *this, keepDims, supportOldShapes);
        NDArray<T> result(newShape, _workspace);
        RELEASE(newShape, _workspace);        
        
        if(rankOf() == copy.size())
            result._buffer[0] = functions::reduce::ReduceFunction<T>::template execScalar<OpName>(_buffer, _shapeInfo, nullptr);        
        else {
            shape::TAD tad(_shapeInfo, copy.data(), copy.size());
            tad.createTadOnlyShapeInfo();
            tad.createOffsets();        
            
            functions::reduce::ReduceFunction<T>::template exec<OpName>(_buffer, _shapeInfo, nullptr, result._buffer,
                                                                        result._shapeInfo, copy.data(), copy.size(),
                                                                        tad.tadOnlyShapeInfo, tad.tadOffsets);       
        }
        
        return result;
    }

//////////////////////////////////////////////////////////////////////////
// method reduces array by excluding its shapes along axes present in dimensions vector
    template<typename T>
    template<typename OpName>
    void NDArray<T>::reduceAlongDimension(NDArray<T>* target, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes, T *extras) const {

        std::vector<int> copy(dimensions);

        int* newShape = ShapeUtils<T>::evalReduceShapeInfo('c', copy, *this, keepDims, supportOldShapes, _workspace);
        if(!shape::shapeEquals(newShape, target->getShapeInfo())) {
            nd4j_printf("NDArray::reduceAlongDimension method: wrong target shape!\n", "");
            throw "NDArray::reduceAlongDimension method: wrong target shape!";
        }
        RELEASE(newShape, _workspace);

        if(rankOf() == copy.size())
            target->_buffer[0] = functions::reduce::ReduceFunction<T>::template execScalar<OpName>(_buffer, _shapeInfo, extras);
        else {
            shape::TAD tad(_shapeInfo, copy.data(), copy.size());
            tad.createTadOnlyShapeInfo();
            tad.createOffsets();

            functions::reduce::ReduceFunction<T>::template exec<OpName>(_buffer, _shapeInfo, extras, target->_buffer,
                                                                        target->_shapeInfo, copy.data(), copy.size(),
                                                                        tad.tadOnlyShapeInfo, tad.tadOffsets);
        }
    }

// method reduces array by excluding its shapes along axes present in dimensions vector
    template<typename T>
    template<typename OpName>
    NDArray<T> *NDArray<T>::reduceAlongDimension(const std::initializer_list<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {
		        
        return reduceAlongDimension<OpName>(std::vector<int>(dimensions), keepDims, supportOldShapes);
	}


//
    template<typename T>
    template<typename OpName>
    T NDArray<T>::reduceNumber(T *extraParams) const {
        return functions::reduce::ReduceFunction<T>::template execScalar<OpName>(_buffer, _shapeInfo, extraParams);
    }

    template<typename T>
    template<typename OpName>
    Nd4jIndex NDArray<T>::indexReduceNumber(T *extraParams) {
        return (Nd4jIndex) functions::indexreduce::IndexReduce<T>::template execScalar<OpName>(_buffer, _shapeInfo, extraParams);
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

// perform array transformation
    template<typename T>
    template<typename OpName>
    NDArray<T> NDArray<T>::transform(T *extraParams) {
    
        NDArray<T> result(this->_shapeInfo, true, this->_workspace);
        functions::transform::Transform<T>::template exec<OpName>(this->_buffer, this->_shapeInfo, result._buffer,
                                                                  result._shapeInfo, extraParams, nullptr, nullptr);
        return result;
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
        if (other->lengthOf() != target->lengthOf())
            throw std::invalid_argument("NDArray::applyPairwiseTransform method - lengths of arrays are mismatched");

        functions::pairwise_transforms::PairWiseTransform<T>::template exec<OpName>(this->_buffer, this->_shapeInfo,
                                                                                    other->_buffer, other->_shapeInfo,
                                                                                    target->_buffer, target->_shapeInfo,
                                                                                    extraParams);
    }


    template<typename T>
    template<typename OpName>
    void NDArray<T>::applyRandom(nd4j::random::RandomBuffer *buffer, NDArray<T>* y, NDArray<T>* z, T* extraArgs) {
        Nd4jPointer state = (Nd4jPointer) buffer;
        if (y == nullptr && z == nullptr) {
            // we're executing indexed z here
            functions::random::RandomFunction<T>::template execTransform<OpName>(state, this->buffer(), this->shapeInfo(), extraArgs);
        } else if (y == nullptr && z != nullptr) {
            // XZ case
            functions::random::RandomFunction<T>::template execTransform<OpName>(state, this->buffer(), this->shapeInfo(), z->buffer(), z->shapeInfo(), extraArgs);
        } else if (y != nullptr && z != nullptr) {
            // XYZ case
            functions::random::RandomFunction<T>::template execTransform<OpName>(state, this->buffer(), this->shapeInfo(), y->buffer(), y->shapeInfo(), z->buffer(), z->shapeInfo(), extraArgs);
        }
    }

    template <typename T>
    Nd4jIndex NDArray<T>::tensorsAlongDimension(std::initializer_list<int> dimensions) const {

        return tensorsAlongDimension(std::vector<int>(dimensions));
    }

    template <typename T>
    Nd4jIndex NDArray<T>::tensorsAlongDimension(const std::vector<int>& dimensions) const {
        
        std::vector<int> copy(dimensions);
        shape::checkDimensions(rankOf(), copy);

        Nd4jIndex tadLength = shape::tadLength(this->_shapeInfo, copy.data(), copy.size());
        Nd4jIndex numTads = this->lengthOf() / tadLength;

        return numTads;
    }

    template <typename T>
    NDArray<T>* NDArray<T>::tensorAlongDimension(int index, const std::initializer_list<int>& dimensions) const {

        return tensorAlongDimension(index, std::vector<int>(dimensions));
    }

    template <typename T>
    void NDArray<T>::printShapeInfo(const char * msg) const {
        //shape::printShapeInfo(_shapeInfo);
        if (msg == nullptr)
            shape::printShapeInfoLinear(_shapeInfo);
        else {
            int rank = shape::rank(_shapeInfo);
            int lim = shape::shapeInfoLength(rank);
            printf("%s: [", msg);
            for (int i = 0; i < shape::shapeInfoLength(rank); i++) {
                printf("%i", _shapeInfo[i]);

                if (i < lim - 1)
                    printf(", ");
            }
            printf("]\n");
        }
        fflush(stdout);
    }

    template <typename T>
    void NDArray<T>::printBuffer(const char* msg, int limit) {
        if (limit == -1)
            limit = (int) this->lengthOf();

        if (msg != nullptr)
            printf("%s: [", msg);
        else
            printf("[");

        for (Nd4jIndex e = 0; e < limit; e++) {
            printf("%f", (float) this->_buffer[e]);
            if (e < limit - 1)
                printf(", ");
        }
        printf("]\n");
        fflush(stdout);
    }

    template <typename T>
    void NDArray<T>::printIndexedBuffer(const char* msg, int limit) const {
        if (limit == -1)
            limit = (int) this->lengthOf();

        if (msg != nullptr)
            printf("%s [", msg);
        else
            printf("[");
        for (Nd4jIndex e = 0; e < limit; e++) {
            printf("%f", (float) this->getIndexedScalar(e));
            if (e < limit - 1)
                printf(", ");
        }
        printf("]\n");
        fflush(stdout);
    }

    template <typename T>
    NDArray<T>* NDArray<T>::tensorAlongDimension(int index, const std::vector<int>& dimensions) const {
        
        std::vector<int> copy(dimensions);
        shape::checkDimensions(rankOf(), copy);

        Nd4jIndex tadLength = shape::tadLength(this->_shapeInfo, copy.data(), copy.size());
        Nd4jIndex numTads = this->lengthOf() / tadLength;

        if (index >= numTads)
            throw "Can't get index higher than total number of TADs";

        shape::TAD tad(this->_shapeInfo, copy.data(), copy.size());
        tad.createTadOnlyShapeInfo();
        tad.createOffsets();

        // shape::printShapeInfoLinear(tad.tadOnlyShapeInfo);

        T* buffer = this->_buffer + tad.tadOffsets[index];

        int* shapeInfo;
        if (_workspace == nullptr) {
            shapeInfo = new int[shape::shapeInfoLength(tad.tadOnlyShapeInfo[0])];
        } else {
            shapeInfo = (int *) _workspace->allocateBytes(shape::shapeInfoByteLength(tad.tadOnlyShapeInfo[0]));
        }
        std::memcpy(shapeInfo, tad.tadOnlyShapeInfo, shape::shapeInfoByteLength(tad.tadOnlyShapeInfo));

        auto array = new NDArray<T>(buffer, shapeInfo, _workspace);
        array->_isBuffAlloc = false;
        array->_isShapeAlloc = true;
        array->_isView = true;

        return array;
    }

// method makes copy of this array and applies to the copy transpose operation, this array remains unaffected 
template <typename T>
    NDArray<T>* NDArray<T>::transpose() const {
        int shapeInfoLength = shape::shapeInfoLength(rankOf());
        int* newShapeInfo = nullptr;

        ALLOCATE(newShapeInfo , _workspace, shapeInfoLength, int);
        memcpy(newShapeInfo, _shapeInfo, shapeInfoLength*sizeof(int));

        NDArray<T>* newArr = new NDArray<T>(_buffer, newShapeInfo, _workspace);
        newArr->_isShapeAlloc = true;
        newArr->_isBuffAlloc  = false;

        newArr->transposei();

        return newArr;
        /*
    int *rearrange = new int[rankOf()];
    int cnt = 0;
    for (int d = rankOf() - 1; d >= 0; d--) {
        rearrange[cnt++] = d;
    }

    int sLen = shape::shapeInfoLength(rankOf());
    int *newShapeBuffer;
    T *newBuffer;
    if (_workspace == nullptr) {
        newShapeBuffer = new int[sLen];
        newBuffer = new T[lengthOf()];
    } else {
        newShapeBuffer = (int*) _workspace->allocateBytes(shape::shapeInfoByteLength(rankOf()));
        newBuffer = (T*) _workspace->allocateBytes(lengthOf() * sizeOfT());
    }
    memcpy(newShapeBuffer, _shapeInfo, shape::shapeInfoByteLength(rankOf()));

    shape::doPermuteShapeInfo(newShapeBuffer, rearrange);

    // fixme: this is bad
    newShapeBuffer[sLen - 2] = 1;

    memcpy(newBuffer, _buffer, sizeOfT() * lengthOf());

    NDArray<T> *result = new NDArray(newBuffer, newShapeBuffer, _workspace);
    result->_isBuffAlloc = true;
    result->_isShapeAlloc = true;

    delete[] rearrange;

    return result;
        */
}

////////////////////////////////////////////////////////////////////////
// method performs transpose operation based on this array and store result in target, this array remains unaffected 
    template <typename T>
    void NDArray<T>::transpose(NDArray<T>& target) const {
        
        int* correctShape = ShapeUtils<T>::evalTranspShapeInfo(*this, _workspace);
        if(!shape::equalsStrict(correctShape, target.getShapeInfo()))
            throw "NDArray::transpose method: the shapeInfo of target array is wrong !";

    // check whether target has allocated (its own) buffer
    if (target._isBuffAlloc) 
        RELEASE(target._buffer, target._workspace);

    target._buffer = _buffer;
    // don't forget to indicate that memory for new array was allocated
    target._isBuffAlloc = false;
    target._isView = true;

    RELEASE(correctShape, _workspace);
}


////////////////////////////////////////////////////////////////////////
// This method applies in-place transpose to this array, so this array becomes transposed 
template <typename T>
    void NDArray<T>::transposei() {
        std::vector<int> perm;
        for (int e = this->rankOf() - 1; e >= 0; e--)
            perm.emplace_back(e);

        this->permutei(perm);


        /*
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

    shape::doPermuteShapeInfo(newShapeBuffer, rearrange);

    // fixme: this is bad
    newShapeBuffer[sLen - 2] = 1;
    _shapeInfo = newShapeBuffer;
    delete []rearrange;
        */
}

    template<typename T>
    bool NDArray<T>::equalsTo(NDArray<T> &other, T eps) const {
        return equalsTo(&other, eps);
    }

// This method returns true if two arrays are equal, with custom or default Eps value of 1e-5, false otherwise
    template<typename T>
    bool NDArray<T>::equalsTo(const NDArray<T> *other, T eps) const {

        if (lengthOf() != other->lengthOf())
            return false;

        // we need to be able to compare [1, len] to [len]
        if ((rankOf() == 1 && other->rankOf() == 2) || (rankOf() == 2 && other->rankOf() == 1)) {
            // FIXME: do something here?
        } else if (!shape::equalsSoft(_shapeInfo, other->_shapeInfo))
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


//////////////////////////////////////////////////////////////////////////
template<typename T>
    void NDArray<T>::addRowVector(const NDArray<T> *row, NDArray<T>* target) const {
        if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() || !row->isRowVector() || columns() != row->lengthOf())
            throw std::invalid_argument("NDArray::addRowVector: wrong arguments !");

        int dimension[1] = {1};

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, dimension, 1));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExcutioner<T>::execBroadcast(0, _buffer, _shapeInfo, row->_buffer, row->_shapeInfo, target->getBuffer(), target->getShapeInfo(),
                                             dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets,
                                             tad->tadOnlyShapeInfo, tad->tadOffsets);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
    void NDArray<T>::subRowVector(const NDArray<T> *row, NDArray<T>* target) const {
        if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() || !row->isRowVector() || columns() != row->columns())
            throw std::invalid_argument("NDArray::subRowVector: wrong arguments !");

        int dimension[1] = {1};

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, dimension, 1));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExcutioner<T>::execBroadcast(1, _buffer, _shapeInfo, row->_buffer, row->_shapeInfo, target->getBuffer(), target->getShapeInfo(),
                                             dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets,
                                             tad->tadOnlyShapeInfo, tad->tadOffsets);
}

//////////////////////////////////////////////////////////////////////////
    template<typename T>
    void NDArray<T>::mulRowVector(const NDArray<T> *row, NDArray<T>* target) const {
        if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() || !row->isRowVector() || columns() != row->columns())
            throw std::invalid_argument("NDArray::divRowVector: wrong arguments !");

        int dimension[1] = {1};

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, dimension, 1));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExcutioner<T>::execBroadcast(2, _buffer, _shapeInfo, row->_buffer, row->_shapeInfo, target->getBuffer(), target->getShapeInfo(),
                                             dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets,
                                             tad->tadOnlyShapeInfo, tad->tadOffsets);

    }

//////////////////////////////////////////////////////////////////////////
    template<typename T>
    void NDArray<T>::divRowVector(const NDArray<T> *row, NDArray<T>* target) const {
        if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() || !row->isRowVector() || columns() != row->columns())
            throw std::invalid_argument("NDArray::divRowVector: wrong arguments !");

        int dimension[1] = {1};

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, dimension, 1));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExcutioner<T>::execBroadcast(3, _buffer, _shapeInfo, row->_buffer, row->_shapeInfo, target->getBuffer(), target->getShapeInfo(),
                                             dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets,
                                             tad->tadOnlyShapeInfo, tad->tadOffsets);

    }

//////////////////////////////////////////////////////////////////////////
// This method adds given row to all rows in this NDArray, this array becomes affected
    template<typename T>
    void NDArray<T>::addiRowVector(const NDArray<T> *row) {
    if (rankOf() != 2 || !row->isRowVector() || columns() != row->lengthOf())
        throw std::invalid_argument("NDArray::addiRowVector: wrong arguments !");

        int dimension[1] = {1};

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, dimension, 1));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExcutioner<T>::execBroadcast(0, _buffer, _shapeInfo, row->_buffer, row->_shapeInfo, _buffer, _shapeInfo,
                                             dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets,
                                             tad->tadOnlyShapeInfo, tad->tadOffsets);
    }


//////////////////////////////////////////////////////////////////////////
    template<typename T>
    void NDArray<T>::addColumnVector(const NDArray<T> *column, NDArray<T>* target) const {
        if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() || !column->isColumnVector() || rows() != column->lengthOf())
            throw std::invalid_argument("NDArray::addColumnVector: wrong arguments !");

        int dimension[1] = {0};

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, dimension, 1));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExcutioner<T>::execBroadcast(0, _buffer, _shapeInfo, column->_buffer, column->_shapeInfo, target->getBuffer(), target->getShapeInfo(),
                                             dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets,
                                             tad->tadOnlyShapeInfo, tad->tadOffsets);
}

//////////////////////////////////////////////////////////////////////////
// This method adds given column to all columns in this NDArray, this array becomes affected
    template<typename T>
    void NDArray<T>::addiColumnVector(const NDArray<T> *column) {
        if (rankOf() != 2 || !column->isColumnVector() || rows() != column->lengthOf())
            throw std::invalid_argument("NDArray::addiColumnVector: wrong arguments !");

        int dimension[1] = {0};

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, dimension, 1));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExcutioner<T>::execBroadcast(0, _buffer, _shapeInfo, column->_buffer, column->_shapeInfo, _buffer, _shapeInfo,
                                             dimension, 1, tad->tadOnlyShapeInfo, tad->tadOffsets,
                                             tad->tadOnlyShapeInfo, tad->tadOffsets);
    }

//////////////////////////////////////////////////////////////////////////
// This method multiplies each column of this array by given argument-column, this array becomes affected
    template<typename T>
    void NDArray<T>::muliColumnVector(const NDArray<T> *column) {
        if (rankOf() != 2 || !column->isColumnVector() || rows() != column->lengthOf())
            throw std::invalid_argument("NDArray::muliColumnVector: wrong arguments !");

        int dimension[1] = {0};

        std::unique_ptr<shape::TAD> tad(new shape::TAD(_shapeInfo, dimension, 1));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        NativeOpExcutioner<T>::execBroadcast(2, _buffer, _shapeInfo, column->_buffer, column->_shapeInfo, _buffer, _shapeInfo,
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
template <typename T>
    void NDArray<T>::updateStrides(const char order) {
	
	shape::updateStrides(_shapeInfo, order);
}

//////////////////////////////////////////////////////////////////////////
// set new order and shape in case of suitable array length 
template <typename T>
    bool NDArray<T>::reshapei(const char order, const std::initializer_list<int>& shape) {
        std::vector<int> vShape(shape);
        return reshapei(order, vShape);
}

template <typename T>
bool NDArray<T>::reshapei(const std::initializer_list<int>& shape) {
    return reshapei('c', shape);
}	

template <typename T>
bool NDArray<T>::reshapei(const std::vector<int>& shape) {
    return reshapei('c', shape);
}

//////////////////////////////////////////////////////////////////////////
    template <typename T>
    void NDArray<T>::enforce(const std::initializer_list<int> &dimensions, char order) {
        std::vector<int> dims(dimensions);
        enforce(dims, order);
    }

    template <typename T>
    void NDArray<T>::enforce(std::vector<int> &dimensions, char o) {

        Nd4jIndex prod = 1;
        for (int e = 0; e < dimensions.size(); e++)
            prod *= dimensions[e];

        if (prod != this->lengthOf()) {
            std::string current = ShapeUtils<T>::shapeAsString(this);
            std::string enforced = ShapeUtils<T>::shapeAsString(dimensions);
            nd4j_printf("Can't enforce new shape, lengths mismatch. Original shape: %s; Requested shape: %s\n", current.c_str(), enforced.c_str());
            throw "Incompatible shape";
        }

        int *newShape;
        ALLOCATE(newShape, _workspace, shape::shapeInfoLength(dimensions.size()), int);

        char order = o == 'a' ? this->ordering() : o;

        if (order == 'c')
            shape::shapeBuffer(dimensions.size(), dimensions.data(), newShape);
        else
            shape::shapeBufferFortran(dimensions.size(), dimensions.data(), newShape);

        if (_isShapeAlloc)
            RELEASE(_shapeInfo, _workspace);

        _shapeInfo = newShape;
        _isShapeAlloc = true;
    }

//////////////////////////////////////////////////////////////////////////
// set new order and shape in case of suitable array length 
template <typename T>
    bool NDArray<T>::reshapei(const char order, const std::vector<int>& cshape) {

    // check firstly whether cshape is identical to shape of array, if yes then reshape is unnecessary 
    if(order == ordering() && rankOf() == cshape.size()) {
        bool areShapesSame = true;
        for(int i = 0; i < cshape.size(); ++i)
            if(cshape[i] != sizeAt(i)) {
                areShapesSame = false;
                break;
            }
        if(areShapesSame)
            return areShapesSame;        
    }

    std::vector<int> shape(cshape);
    int rank = shape.size();

    // looking for negative in shape

    int numberNegativesOnes = 0;

    int* shape_ = shape.data();
    for (int i = 0; i < (int) shape.size(); i++) {
        if (shape[i] < 0) {
            if (numberNegativesOnes >= 1)
                throw "Only one dimension can be negative at once";

            numberNegativesOnes++;

            int shapeLength = 1;
            for (int j = 0; j < (int) shape.size(); j++)
                if (i != j)
                    shapeLength *= shape_[j];

            int realShape = nd4j::math::nd4j_abs<int>(lengthOf() / shapeLength);
            int* thisNewShape = new int[shape.size()];

            for (int j = 0; j < (int) shape.size(); j++) 
                if (i != j) 
                    thisNewShape[j] = shape_[j];
                else
                    thisNewShape[j] = realShape;
            
            shape_ = thisNewShape;
        }
    }

    for (int e = 0; e < (int) shape.size(); e++) 
        shape[e] = shape_[e];

    if (numberNegativesOnes > 0)
        delete[] shape_;

    int arrLength = 1;
    for(const auto& item : shape)
        arrLength *= item;

    if(_buffer==nullptr || arrLength != this->lengthOf()) {
        this->printShapeInfo("Mismatched shape");
        nd4j::Logger::printv("Shape requested: ", shape);
        nd4j_debug("Requested length in reshape: %i; Existing length: %i;\n", arrLength, this->lengthOf());
        throw "Bad shape!";
    }

    int shapeLength = shape::shapeInfoLength(rank);
    // remember old values

    // we can do this only if there was no permute applied, or there are no weird strides
    if (shape::canReshape(this->rankOf(), this->_shapeInfo, shape.size(), shape.data(), order == 'f')) {
        int *shapeInfoNew;
        ALLOCATE(shapeInfoNew, _workspace, shape::shapeInfoLength(rank), int);

        shape::reshapeCF(this->rankOf(), this->_shapeInfo, shape.size(), shape.data(), order == 'f', shapeInfoNew);

        if (_isShapeAlloc)
            RELEASE(_shapeInfo, _workspace);

        _shapeInfo = shapeInfoNew;
        _isShapeAlloc = true;
    } else {
        int *shapeInfoNew;
        ALLOCATE(shapeInfoNew, _workspace, shape::shapeInfoLength(rank), int);

        if (order == 'c')
            shape::shapeBuffer(shape.size(), shape.data(), shapeInfoNew);
        else
            shape::shapeBufferFortran(shape.size(), shape.data(), shapeInfoNew);

        T *newBuffer;
        ALLOCATE(newBuffer, _workspace, this->lengthOf(), T);

        functions::pairwise_transforms::PairWiseTransform<T>::template exec<simdOps::Copy<T>>(newBuffer, shapeInfoNew, this->_buffer, this->_shapeInfo, newBuffer, shapeInfoNew, nullptr);

        if (_isBuffAlloc)
            RELEASE(_buffer, _workspace);


        if (_isShapeAlloc)
            RELEASE(_shapeInfo, _workspace);


        _buffer = newBuffer;
        _shapeInfo = shapeInfoNew;
        _isShapeAlloc = true;
        _isBuffAlloc = true;
    }

    return true;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
    Nd4jIndex NDArray<T>::argMax(std::initializer_list<int> dimensions) {
        if (dimensions.size() == 0) {
            Nd4jIndex max = 0;
            T mv = -MAX_FLOAT;
            for (Nd4jIndex e = 0; e < this->lengthOf(); e++) {
                T val = this->getScalar(e);
                if (mv < val) {
                    mv = val;
                    max = e;
                }
            }

            return max;
        } else
            throw "Not implemented yet";
}

//////////////////////////////////////////////////////////////////////////
// create new array with corresponding order and shape, new array will point to the same _buffer as this array
template <typename T>
NDArray<T>* NDArray<T>::reshape(const char order, const std::vector<int>& shape) const {
	int shapeInfoLength = shape::shapeInfoLength(rankOf());
	int* newShapeInfo = nullptr;

	ALLOCATE(newShapeInfo , _workspace, shapeInfoLength, int);
	memcpy(newShapeInfo, _shapeInfo, shapeInfoLength*sizeof(int));

	NDArray<T>* newArr = new NDArray<T>(_buffer, newShapeInfo, _workspace);
	newArr->_isShapeAlloc = true;
	newArr->_isBuffAlloc  = false;
	newArr->reshapei(order, shape);

	return newArr;
}

//////////////////////////////////////////////////////////////////////////
// change an array by repeating it the number of times given by reps.
template <typename T>
void NDArray<T>::tilei(const std::vector<int>& reps) {

    *this = this->tile(reps);
	
}


//////////////////////////////////////////////////////////////////////////
// change an array by repeating it the number of times given by reps.
template <typename T>
NDArray<T> NDArray<T>::tile(const std::vector<int>& reps) const {
    
    int dim = reps.size();  
    int product = 1;
    for(const auto& item : reps)
        product *= item;
    if(product == 0)
        throw "NDArray::tile method: one of the elements in reps array is zero !";

    int rankOld = rankOf();
    int diff = rankOld - dim;
    if(product==1) {        // in this case 2 possibilities are present: just reshape or nothing to do
        NDArray<T> result(*this);
        if(diff < 0) {      // reshape to higher dimension          
            std::vector<int> shapeNew = reps;               // need to have unities at first "diff" positions of new shape
            memcpy(&shapeNew[-diff], result._shapeInfo+1, rankOld*sizeof(int));   // put old shape numbers at rest of positions
            result.reshapei(ordering(), shapeNew);
        }       
        return result;             // nothing to do, if diff >= 0 -> identity tile 
    }   
    
    // evaluate shapeInfo for resulting array
    int* newShapeInfo = ShapeUtils<T>::evalTileShapeInfo(*this, reps, _workspace);
    // create new buffer, in any case the memory amount new buffer points to is bigger then those for old _buffer   
    T* newBuff = nullptr;
    ALLOCATE(newBuff, _workspace, shape::length(newShapeInfo), T);
    // assign new shape and new buffer to resulting array
    NDArray<T> result(newBuff, newShapeInfo, _workspace);
    result._isShapeAlloc = true;
    result._isBuffAlloc = true;

    // fill newBuff, loop through all elements of newBuff 
    // looping through _buffer goes automatically by means of getSubArrayIndex applying
    const auto resultLen = result.lengthOf();
    if(result.ordering() == 'c') {           //  ews == 1 always here 
#pragma omp parallel for simd if(resultLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for(int i=0;  i<resultLen; ++i)
            newBuff[i] = (*this)(shape::subArrayIndex(newShapeInfo, _shapeInfo, i));
    }
    else {
        int idx[MAX_RANK];
        auto resultShape   = result.shapeOf();
        auto resultStrides = result.stridesOf();
        const auto resultRank = result.rankOf();
#pragma omp parallel for simd if(resultLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided) private(idx)
        for(int i=0;  i<resultLen; ++i) {
            shape::ind2subC(resultRank, resultShape, i, idx);
            newBuff[ shape::getOffset(0, resultShape, resultStrides, idx, resultRank) ] = (*this)(shape::subArrayIndex(newShapeInfo, _shapeInfo, i));
        }
    }
              
    return result;
}

//////////////////////////////////////////////////////////////////////////
// change an array by repeating it the number of times given by reps.
template <typename T>
void NDArray<T>::tile(const std::vector<int>& reps, NDArray<T>& target) const {
        
    // evaluate true tile shapeInfo for comparison with target shapeInfo
    int* newShapeInfo = ShapeUtils<T>::evalTileShapeInfo(*this, reps, _workspace);
    if(!shape::equalsSoft(newShapeInfo, target.getShapeInfo()))  {
        delete []newShapeInfo;    
        throw "NDArray::tile method - shapeInfo of target array is not suitable for tile operation !";
    }
    delete[] newShapeInfo;

    // fill newBuff, loop through all elements of newBuff 
    // looping through _buffer goes automatically by means of getSubArrayIndex applying
    const int ews = target.ews();
    const int targetLen = target.lengthOf();
    T* targetBuff = target.getBuffer();
    if(target.ordering() == 'c' && ews == 1) {           //  ews == 1 always here 
#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided) 
        for(int i=0;  i<targetLen; ++i)
            targetBuff[i] = (*this)(shape::subArrayIndex(target._shapeInfo, _shapeInfo, i));
    }
    else if(target.ordering() == 'c' && ews > 1) {
#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for(int i=0;  i<targetLen; ++i)
            targetBuff[i*ews] = (*this)(shape::subArrayIndex(target._shapeInfo, _shapeInfo, i));   
    }
    else {
        int idx[MAX_RANK];
        auto targetShape     = target.shapeOf();
        auto targetStrides   = target.stridesOf();
        const auto targetRank = target.rankOf();
#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided) private(idx)
        for(int i=0;  i<targetLen; ++i) {
            shape::ind2subC(targetRank, targetShape, i, idx);
            targetBuff[ shape::getOffset(0, targetShape, targetStrides, idx, targetRank) ] = (*this)(shape::subArrayIndex(target._shapeInfo, _shapeInfo, i));
        }
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray<T>::tile(NDArray<T>& target) const {
        
    if(rankOf() > target.rankOf())
        throw "NDArray::tile method - rank of target array must be bigger or equal to the rank of this array !";
    
    if(!ShapeUtils<T>::areShapesBroadcastable(*this, target))         
        throw "NDArray::tile method - shapeInfo of target array is not suitable for tile operation !";

    // fill newBuff, loop through all elements of newBuff 
    // looping through _buffer goes automatically by means of getSubArrayIndex applying
    const int ews = target.ews();
    const int targetLen = target.lengthOf();
    T* targetBuff = target.getBuffer();
    if(target.ordering() == 'c' && ews == 1) {           //  ews == 1 always here 
#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for(int i=0;  i<targetLen; ++i)
            targetBuff[i] = (*this)(shape::subArrayIndex(target._shapeInfo, _shapeInfo, i));
    }
    else if(target.ordering() == 'c' && ews > 1) {
#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for(int i=0;  i<targetLen; ++i)
            targetBuff[i*ews] = (*this)(shape::subArrayIndex(target._shapeInfo, _shapeInfo, i));   
    }
    else {
        int idx[MAX_RANK];
        auto targetShape     = target.shapeOf();
        auto targetStrides   = target.stridesOf();
        const auto targetRank = target.rankOf();
#pragma omp parallel for simd if(targetLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided) private(idx)
        for(int i=0;  i<targetLen; ++i) {
            shape::ind2subC(targetRank, targetShape, i, idx);
            targetBuff[ shape::getOffset(0, targetShape, targetStrides, idx, targetRank) ] = (*this)(shape::subArrayIndex(target._shapeInfo, _shapeInfo, i));
        }
    }
}

//////////////////////////////////////////////////////////////////////////
    template<typename T>
    int NDArray<T>::sizeAt(int dim) const {
        if (dim >= this->rankOf() || dim < -this->rankOf())
            throw "Bad size index requested";

        if (dim >= 0)
            return this->_shapeInfo[1+dim];
        else
            return this->_shapeInfo[1+(this->rankOf() + dim)];
    }


//////////////////////////////////////////////////////////////////////////
// create new  array by repeating it the number of times given by reps
template<typename T>
    NDArray<T>* NDArray<T>::repeat(int dimension, const std::vector<int>& repeats) const {

    std::vector<int> outShape = ShapeUtils<T>::evalRepeatShape(dimension, repeats, *this);
    
    // the size of outShape == rank
    int rank = rankOf();            // = outShape.size()

    int *newShape = new int[rank];
    for (int i = 0; i < rank; i++)
        newShape[i] = outShape[i];

    NDArray<T>* ret = new NDArray<T>('c', outShape, _workspace);

    auto repeatDelta = shape::prodLong(newShape, rank) / this->lengthOf();
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

    delete[] newShape;

    return ret;
}

//////////////////////////////////////////////////////////////////////////
// fill array by repeating it the number of times given by reps
    template<typename T>
    void NDArray<T>::repeat(int dimension, NDArray<T>& target) const {
    
    Nd4jIndex repeatDelta = shape::prodLong(target.shapeOf(), rankOf()) / this->lengthOf();
    Nd4jIndex numTads = this->tensorsAlongDimension({dimension});
    for (int i = 0; i < numTads; i++) {
        NDArray<T>* thisTensor = this->tensorAlongDimension(i, {dimension});
        NDArray<T>* retTensor = target.tensorAlongDimension(i, {dimension});
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
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
bool NDArray<T>::permutei(const int* dimensions, const int rank) {

    // check if current object is _shapeInfo owner
    if (!_isShapeAlloc) {             // if _shapeInfo is not its own
        _shapeInfo = ShapeUtils<T>::evalPermShapeInfo(dimensions, rank, *this, _workspace);
        _isShapeAlloc = true;
    } else {
        if (!nonNull() || rank != rankOf())
            throw "NDArray::permutei method: wrong arguments in permutei method: either array is nullptr or rank is not suitable!";
        shape::doPermuteShapeInfo(_shapeInfo, dimensions);
    }

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
NDArray<T>* NDArray<T>::permute(const int* dimensions, const int rank) const {

    // evaluate shapeInfo for output (permuted) array ret
    int* shapeInfoNew = ShapeUtils<T>::evalPermShapeInfo(dimensions, rank, *this, _workspace);
    // create array to be returned
    NDArray<T>* ret = new NDArray<T>(_buffer, shapeInfoNew, _workspace);
    // don't forget to indicate that memory for new array was allocated
    ret->_isBuffAlloc = false;
    ret->_isShapeAlloc = true;
	ret->_isView = true;

    return ret;
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T>* NDArray<T>::permute(const std::vector<int>& dimensions) const {
    return permute(dimensions.data(), dimensions.size());
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T>* NDArray<T>::permute(const std::initializer_list<int>& dimensions) const {
    std::vector<int> vec(dimensions);
    return permute(vec);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray<T>::permute(const int* dimensions, const int rank, NDArray<T>& target) const {

    if (!nonNull() || !target.nonNull() || rank != rankOf() || rank != target.rankOf() )
        throw "NDArray<T>::permute method: either arrays are nullptr or ranks are not suitable!";

    // check whether target has allocated (its own) buffer
    if (target._isBuffAlloc) 
        RELEASE(target._buffer, target._workspace);

    target._buffer = _buffer;
    // don't forget to indicate that memory for new array was allocated
    target._isBuffAlloc = false;
    target._isView = true;

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray<T>::permute(const std::vector<int>& dimensions, NDArray<T>& target) const {
    permute(dimensions.data(), dimensions.size(), target);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
template<typename OpName>
void NDArray<T>::applyBroadcast(std::initializer_list<int> dimensions, const NDArray<T>* tadArray, NDArray<T>* target, T* extraArgs) {
    std::vector<int> vec(dimensions);
    applyBroadcast<OpName>(vec, tadArray, target, extraArgs);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
template<typename OpName>
void NDArray<T>::applyBroadcast(std::vector<int>& dimensions, const NDArray<T>* tadArray, NDArray<T>* target, T* extraArgs) {
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
    functions::broadcast::Broadcast<T>::template exec<OpName>(this->_buffer, this->_shapeInfo, tadArray->_buffer, tadArray->_shapeInfo, result->_buffer, result->_shapeInfo, copy.data(), (int)copy.size(), tad.tadOnlyShapeInfo, tad.tadOffsets, tad.tadOnlyShapeInfo, tad.tadOffsets);
}


//////////////////////////////////////////////////////////////////////////
template<typename T>
template <typename OpName>
void NDArray<T>::applyTrueBroadcast(const NDArray<T>* other, NDArray<T>* target, const bool checkTargetShape, T *extraArgs) const {

    if(target == nullptr || other == nullptr)
        throw "NDArray::applyTrueBroadcast method: target or other = nullptr !";

    const NDArray<T>* min(nullptr), *max(nullptr);
    if(this->rankOf() >= other->rankOf()) {
        max = this;
        min = other;
    }
    else {
        max = other;
        min = this;
    }

    if(checkTargetShape) {
        int* newShapeInfo = nullptr;
        if(!ShapeUtils<T>::evalBroadcastShapeInfo(*max, *min, false, newShapeInfo, _workspace))          // the rank of target array must be equal to max->rankOf)()
            throw "NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for broadcast operation !" ;
        if(!shape::equalsSoft(target->getShapeInfo(), newShapeInfo))
            throw "NDArray::applyTrueBroadcast method: the shape of target array is wrong !";

        // if workspace is not null - do not call delete.
        if (_workspace == nullptr)
            delete[] newShapeInfo;
    }

    // check whether min array have to be tiled
    if(!max->isSameShape(target)) {
        // evaluate repeating dimensions for tile operation
        std::vector<int> repeatMax(max->rankOf());
        for(int i = 1; i <= max->rankOf(); ++i)
            repeatMax[i-1] = (target->_shapeInfo[i] / max->_shapeInfo[i]);
        max->tile(repeatMax, *target);
    }
    else
        target->assign(max);

    // check whether min array have to be tiled
    std::vector<int> repeatMin(min->rankOf());
    int product = 1;
    for(int i = min->rankOf(); i >=1 ; --i) {
        repeatMin[i-1] = (target->_shapeInfo[target->rankOf() - min->rankOf() + i] / min->_shapeInfo[i]);
        product *= repeatMin[i-1];
    }

    if(product != 1 ) {
        NDArray<T> tiledMin = min->tile(repeatMin);
        std::vector<int> sameDims = ShapeUtils<T>::getDimsWithSameShape(target, &tiledMin);
        target->template applyBroadcast<OpName>(sameDims, &tiledMin, nullptr, extraArgs);
    }
    else {
        std::vector<int> sameDims = ShapeUtils<T>::getDimsWithSameShape(target, min);
        target->template applyBroadcast<OpName>(sameDims, min, nullptr, extraArgs);
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
template <typename OpName>
NDArray<T>* NDArray<T>::applyTrueBroadcast(const NDArray<T>* other, T *extraArgs) const {

    int* newShapeInfo = nullptr;
    if(!ShapeUtils<T>::evalBroadcastShapeInfo(*this, *other, true, newShapeInfo, _workspace))          // the rank of new array = max->rankOf)()
        throw "NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for broadcast operation !" ;
    NDArray<T>* result = new NDArray<T>(newShapeInfo, false, this->_workspace);

    // if workspace is not null - do not call delete.
    if (_workspace == nullptr)
        delete[] newShapeInfo;

    this->template applyTrueBroadcast<OpName>(other, result, false, extraArgs);
  
    return result;
}


//////////////////////////////////////////////////////////////////////////
template<typename T>
template <typename OpName>
NDArray<T> NDArray<T>::applyTrueBroadcast(const NDArray<T>& other, T *extraArgs) const {

    NDArray<T>* pResult = this->template applyTrueBroadcast<OpName>(&other, extraArgs);
    pResult->_isShapeAlloc = false;
    pResult->_isBuffAlloc  = false;

    NDArray<T> result(pResult->_buffer, pResult->_shapeInfo, _workspace);
    result._isShapeAlloc = true;
    result._isBuffAlloc  = true;
    
    delete pResult;

    return result;
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
	int* shapeInfoNew = new int[shape::shapeInfoLength(biggerRank)];
	memcpy(shapeInfoNew, biggerShapeInfo, shape::shapeInfoByteLength(biggerRank));
	for (int i = smallerRank; i>=1; --i) 
		if(shapeInfoNew[diff+i] == 1 || smallerShapeInfo[i] == 1) 
			shapeInfoNew[diff+i] *= smallerShapeInfo[i];

	NDArray<T>* ret = new NDArray<T>(shapeInfoNew, _workspace);
	ret->updateStrides(order);
	delete []shapeInfoNew;

	return ret;
}


//////////////////////////////////////////////////////////////////////////
// check whether array's rows (arg=0) or columns (arg=1) create orthogonal basis
template<typename T>
bool NDArray<T>::hasOrthonormalBasis(const int arg) {

	if(rankOf() !=2 )
		throw "NDArray::hasOrthBasis method: rank of ndarray is not equal 2 !";

	if(arg!=0  && arg!=1)
		throw "NDArray::hasOrthBasis method: input argument is not equal to 0 or 1 !";

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
			if(dot != (T) 0.f && nd4j::math::nd4j_abs(nd4j::math::nd4j_sqrt<T>(dot) - (T) 1.f) > eps)
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
				dot = (T) 0.f;
			}
		for(int i=0; i<rows(); ++i) {		// check whether norm of row vector = 1
			for(int j=0; j<columns(); ++j)
					dot += getScalar(i,j)*getScalar(i,j);
			if(dot!= (T) 0.f && nd4j::math::nd4j_abs(nd4j::math::nd4j_sqrt<T>(dot) - (T) 1.f) > eps)
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
		for(int j=0; j<columns(); ++j) {
			if (i == j) continue;
			if(nd4j::math::nd4j_abs(getScalar(i,j)) > eps)
				return false;
		}
	return true;
}

//////////////////////////////////////////////////////////////////////////
// check whether array is unitary matrix
template<typename T>
bool NDArray<T>::isUnitary() {

    if(rankOf() !=2 || rows() != columns())
        throw "isUnitary method: matrix must be square and have rank = 2 !";

    NDArray<T>* tr = this->transpose();
    NDArray<T>* trMul = nd4j::NDArrayFactory<T>::mmulHelper(this, tr, nullptr, 1.f, 0.f);

    bool result = trMul->isIdentityMatrix();
    delete tr;
    delete trMul;
    
    return result;
}

////////////////////////////////////////////////////////////////////////
    template<typename T>
    NDArray<T>* NDArray<T>::subarray(IndicesList& idx, std::vector<int>& strides) const {
        auto raw = subarray(idx);

        for (int e = 0; e < strides.size(); e++)
            raw->stridesOf()[e] *= strides[e];

        return raw;
    }

    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    NDArray<T>* NDArray<T>::subarray(IndicesList& idx) const {
        // scalar subarray edge case
        if (idx.isScalar()) {
            auto pnt = idx.at(0)->getIndices().at(0);
            return new NDArray<T>('c', {1, 1}, {this->getScalar(pnt)});   
        }

        if (idx.size() != this->rankOf())
            throw "Number of indices should match";

        int *newShape;
        ALLOCATE(newShape, _workspace, shape::shapeInfoLength(this->rankOf()), int);
        memcpy(newShape, this->_shapeInfo, shape::shapeInfoByteLength(this->rankOf()));
        newShape[shape::shapeInfoLength(this->rankOf()) - 2] = -1;

        int *shapeOf = shape::shapeOf(newShape);
        int *stridesOf = shape::stride(newShape);

        Nd4jIndex offset = 0;

        //shape::printShapeInfoLinear(newShape);

        for (int d = 0; d < idx.size(); d++) {
            // building new shape first
            auto index = idx.at(d);
            if (index->isAll()) {
                // shape is unchanged  for this dimension
            } else {
                // size at this dimension equals to either 1 or interval
                shapeOf[d] = index->getIndices().size();

                // for offset we're taking only the first index
                int first = index->getIndices().at(0);
                offset += first * stridesOf[d];

                shape::stride(newShape)[d] *= index->stride();
                nd4j_debug("dimension_ [%i] stride [%i]\n", d, index->stride());
            }
        }

        //shape::printShapeInfoLinear(newShape);

        auto result = new NDArray<T>(this->_buffer + offset, newShape, this->_workspace);
        result->_isShapeAlloc = true;

        return result;
    }
    
    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    NDArray<T>* NDArray<T>::subarray(const std::initializer_list<NDIndex*>& idx) const {
        if (idx.size() != this->rankOf())
            throw "NDArray::subarray: number of indices should match the array rank";

        int *newShape;
        ALLOCATE(newShape, _workspace, shape::shapeInfoLength(this->rankOf()), int);
        memcpy(newShape, this->_shapeInfo, shape::shapeInfoByteLength(this->rankOf()));
        newShape[shape::shapeInfoLength(this->rankOf()) - 2] = -1;

        int *shapeOf = shape::shapeOf(newShape);
        int *stridesOf = shape::stride(newShape);

        Nd4jIndex offset = 0;
        int d = 0;
        for (const auto& index : idx) {
            // building new shape first
            if (index->isAll()) {
                // shape is unchanged  for this dimension
            } else {
                // size at this dimension equals to either 1 or interval
                shapeOf[d] = index->getIndices().size();
                // for offset we're taking only the first index
                int first = index->getIndices().at(0);
                offset += first * stridesOf[d];
            }
            ++d;
        }

        auto result = new NDArray<T>(this->_buffer + offset, newShape, this->_workspace);
        result->_isShapeAlloc = true;

        for (auto v: idx) {
            delete v;
        }

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    NDArray<T>* NDArray<T>::subarray(const Intervals& idx) const {

        if (idx.size() != this->rankOf())
            throw "NDArray::subarray: number of indices should match the rank of array!";

        int *newShape;
        ALLOCATE(newShape, _workspace, shape::shapeInfoLength(this->rankOf()), int);
        memcpy(newShape, this->_shapeInfo, shape::shapeInfoByteLength(this->rankOf()));
        newShape[shape::shapeInfoLength(this->rankOf()) - 2] = -1;

        int *shapeOf = shape::shapeOf(newShape);
        int *stridesOf = shape::stride(newShape);

        Nd4jIndex offset = 0;
        for (int d = 0; d < idx.size(); ++d) {
            // building new shape first
            if (!idx[d].empty()) {
                if (idx[d].size() != 2)
                    throw "NDArray::subarray: the interval must contain only two numbers {first, last} !";
                shapeOf[d] = idx[d][1] - idx[d][0];
                // for offset we're taking only the first index
                offset += idx[d][0] * stridesOf[d];
            }
        }

        auto result = new NDArray<T>(this->_buffer + offset, newShape, this->_workspace);
        result->_isShapeAlloc = true;

        return result;
    }

    template <typename T>
    NDArray<T>* NDArray<T>::cast(DataType dtype) {
        // TODO: to be implemented
        return nullptr;
    }

    template <typename T>
    void NDArray<T>::cast(NDArray<T>* target, DataType dtype) {
        // TODO: to be implemented properly
        target->assign(this);
    }

    template<typename T>
    template<typename OpName>
    void NDArray<T>::applyIndexReduce(const NDArray<T>* target, const std::vector<int>& dimensions, const T *extraParams) const {
        if (target->isScalar()) {
            target->_buffer[0] = functions::indexreduce::IndexReduce<T>::template execScalar<OpName>(_buffer, _shapeInfo, const_cast<T*>(extraParams));
        } else {
            std::vector<int> copy(dimensions);
            if (dimensions.size() > 1)
                std::sort(copy.begin(), copy.end());

            shape::TAD tad(_shapeInfo, copy.data(), copy.size());
            tad.createTadOnlyShapeInfo();
            tad.createOffsets();

            functions::indexreduce::IndexReduce<T>::template exec<OpName>(_buffer, _shapeInfo, const_cast<T*>(extraParams), target->_buffer,
                                                                          target->_shapeInfo, copy.data(), copy.size(),
                                                                          tad.tadOnlyShapeInfo, tad.tadOffsets);
        }
    }
    ////////////////////////////////////////////////////////////////////////
    // reduce dimensions in this array relying on index operations
    template<typename T>
    template<typename OpName>
    NDArray<T>* NDArray<T>::applyIndexReduce(const std::vector<int>& dimensions, const T* extraParams ) const {
        
        std::vector<int> copy(dimensions);
        if (dimensions.size() > 1)
            std::sort(copy.begin(), copy.end());

        int* newShape = ShapeUtils<T>::evalReduceShapeInfo('c', copy, *this);
        NDArray<T>* result = new NDArray<T>(newShape, _workspace);
        RELEASE(newShape, _workspace);

        if(rankOf() == copy.size())
            result->_buffer[0] = functions::indexreduce::IndexReduce<T>::template execScalar<OpName>(_buffer, _shapeInfo, const_cast<T*>(extraParams));
        else {
            shape::TAD tad(_shapeInfo, copy.data(), copy.size());
            tad.createTadOnlyShapeInfo();
            tad.createOffsets();
        
            functions::indexreduce::IndexReduce<T>::template exec<OpName>(_buffer, _shapeInfo, const_cast<T*>(extraParams), result->_buffer,
                                                                    result->_shapeInfo, copy.data(), copy.size(),
                                                                    tad.tadOnlyShapeInfo, tad.tadOffsets);
        }
        
        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    // apply reduce3 operations to this and other array, return result in new output array
    template<typename T>
    template<typename OpName>
    NDArray<T>* NDArray<T>::applyReduce3(const NDArray<T>* other, const T* extraParams) const {
        // check shapes consistency
        if(!isSameShape(other))
            throw "NDArray::applyReduce3 method: the shapes of array must be the same !";
        // create shapeInfo for scalar
        int* newShape = nullptr;
        ALLOCATE(newShape, _workspace, 8, int);
        newShape[0] = 2;    // set rank    
        newShape[1] = 1;    // set first dimension (scalar)
        newShape[2] = 1;    // set second dimension (scalar)
        shape::updateStrides(newShape, 'c');
        // create output array (scalar)
        NDArray<T>* result = new NDArray<T>(newShape, _workspace);
        RELEASE(newShape, _workspace);
        // create temporary array of extra parameters if array extraParams is empty (==nullptr)
        T* extraParamsVals = nullptr;
        if(extraParams == nullptr) {
            extraParamsVals = new T[3] {(T) 0.0, (T) 0.0, (T) 0.0};
            extraParams = extraParamsVals;  
        }
        
        result->_buffer[0] = functions::reduce3::Reduce3<T>::template execScalar<OpName>(_buffer, _shapeInfo, const_cast<T*>(extraParams), other->_buffer, other->_shapeInfo);        
        
        delete []extraParamsVals;

        return result;
    }
    
    ////////////////////////////////////////////////////////////////////////
    // apply reduce3 (execAll) operations to this and other array, return result in new output array
    template<typename T>
    template<typename OpName>
    NDArray<T>*  NDArray<T>::applyAllReduce3(const NDArray<T>* other, const std::vector<int>& dimensions, const T* extraParams) const {
        // be careful, copy array may undergo changes (sort, transformation of negative dimensions to positive, duplicates removing )
        std::vector<int> copy(dimensions);
        shape::checkDimensions(rankOf(), copy);
        shape::checkDimensions(other->rankOf(), copy);               
        // create tads
        shape::TAD tadX(_shapeInfo, copy.data(), copy.size());
        tadX.createTadOnlyShapeInfo();
        tadX.createOffsets();

        shape::TAD tadY(other->_shapeInfo, copy.data(), copy.size());
        tadY.createTadOnlyShapeInfo();
        tadY.createOffsets();        
        // check tads shapes
        if(!shape::equalsSoft(tadX.tadOnlyShapeInfo, tadY.tadOnlyShapeInfo)) 
            throw "NDArray::applyAllReduce3 method: the shapes of array tads are different !";
        // evaluate numbers of tads
        Nd4jIndex tadLengthX = shape::tadLength(_shapeInfo, copy.data(), copy.size());
        Nd4jIndex numTadsX = lengthOf() / tadLengthX;
        
        Nd4jIndex tadLengthY = shape::tadLength(other->_shapeInfo, copy.data(), copy.size());
        Nd4jIndex numTadsY = other->lengthOf() / tadLengthY;
        // set newShape for output array        
        int* newShape = nullptr;
        ALLOCATE(newShape, _workspace, 8, int);
        newShape[0] = 2;        // output rank is always equal to 2 for execAll case
        newShape[1] = numTadsX;
        newShape[2] = numTadsY;
        shape::updateStrides(newShape, 'c');
        // create output array
        NDArray<T>* result = new NDArray<T>(newShape, _workspace);
        RELEASE(newShape, _workspace);
        // create temporary array of extra parameters if array extraParams is empty (==nullptr)
        T* extraParamsVals = nullptr;
        if(extraParams == nullptr) {
            extraParamsVals = new T[3] {(T) 0.0, (T) 0.0, (T) 0.0};
            extraParams = extraParamsVals;  
        }
        // perform calculations
        functions::reduce3::Reduce3<T>::template execAll<OpName>(_buffer, _shapeInfo, const_cast<T*>(extraParams),
                                                                 other->_buffer, other->_shapeInfo, result->_buffer,result->_shapeInfo,
                                                                 copy.data(), copy.size(), tadX.tadOnlyShapeInfo, tadX.tadOffsets, tadY.tadOnlyShapeInfo, tadY.tadOffsets);
        delete []extraParamsVals;
        return result;
    }
 
    ////////////////////////////////////////////////////////////////////////
    // apply reduce3 (exec) operations to this and other array, return result in new output array
    template<typename T>
    template<typename OpName>
    NDArray<T>* NDArray<T>::applyReduce3(const NDArray<T>* other, const std::vector<int>& dimensions, const T* extraParams) const {
        
        std::vector<int> copy(dimensions);
        shape::checkDimensions(rankOf(), copy);
        shape::checkDimensions(other->rankOf(), copy);               

        int* newShape = ShapeUtils<T>::evalReduceShapeInfo('c', copy, *this);
        NDArray<T>* result = new NDArray<T>(newShape, _workspace);
        RELEASE(newShape, _workspace);
        // create temporary array of extra parameters if array extraParams is empty (==nullptr)
        T* extraParamsVals = nullptr;
        if(extraParams == nullptr) {
            extraParamsVals = new T[3] {(T) 0.0, (T) 0.0, (T) 0.0};
            extraParams = extraParamsVals;  
        }
        // perform calculations
        if(rankOf() == copy.size() && other->rankOf() == copy.size())
            result->_buffer[0] = functions::reduce3::Reduce3<T>::template execScalar<OpName>(_buffer, _shapeInfo, const_cast<T*>(extraParams), other->_buffer, other->_shapeInfo);
        else {
            shape::TAD tadX(_shapeInfo, copy.data(), copy.size());
            tadX.createTadOnlyShapeInfo();
            tadX.createOffsets();

            shape::TAD tadY(other->_shapeInfo, copy.data(), copy.size());
            tadY.createTadOnlyShapeInfo();
            tadY.createOffsets();        
        
            functions::reduce3::Reduce3<T>::template exec<OpName>(_buffer, _shapeInfo, const_cast<T*>(extraParams),
                                                                 other->_buffer, other->_shapeInfo, result->_buffer,result->_shapeInfo,
                                                                 copy.data(), copy.size(), tadX.tadOnlyShapeInfo, tadX.tadOffsets);
        }
        
        delete []extraParamsVals;
        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    template<typename OpName>
    NDArray<T>* NDArray<T>::varianceAlongDimension(const bool biasCorrected, const std::vector<int>& dimensions) const {
    
        std::vector<int> copy(dimensions);
        if (copy.size() > 1)
            std::sort(copy.begin(), copy.end());
            
        int* newShape = ShapeUtils<T>::evalReduceShapeInfo('c', copy, *this);
        NDArray<T>* result = new NDArray<T>(newShape, _workspace);
        RELEASE(newShape, _workspace);        
        
        if(rankOf() == copy.size())
            result->_buffer[0] = functions::summarystats::SummaryStatsReduce<T>::template execScalar<OpName>(biasCorrected, _buffer, _shapeInfo, nullptr);
        else
            functions::summarystats::SummaryStatsReduce<T>::template exec<OpName>(biasCorrected, _buffer, _shapeInfo, nullptr,
                                                                              result->_buffer, result->_shapeInfo, copy.data(), copy.size());
        return result;    
    
    }
    
    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    template<typename OpName>
    NDArray<T>* NDArray<T>::varianceAlongDimension(const bool biasCorrected, const std::initializer_list<int>& dimensions) const {
    
        return varianceAlongDimension<OpName>(biasCorrected, std::vector<int>(dimensions));
    }

    template<typename T>
    template<typename OpName>
    void NDArray<T>::varianceAlongDimension(const NDArray<T> *target, const bool biasCorrected, const std::vector<int>& dimensions) {
        std::vector<int> copy(dimensions);
        if (copy.size() > 1)
            std::sort(copy.begin(), copy.end());

        if(rankOf() == copy.size())
            target->_buffer[0] = functions::summarystats::SummaryStatsReduce<T>::template execScalar<OpName>(biasCorrected, _buffer, _shapeInfo, nullptr);
        else
            functions::summarystats::SummaryStatsReduce<T>::template exec<OpName>(biasCorrected, _buffer, _shapeInfo, nullptr,
                                                                                  target->_buffer, target->_shapeInfo, copy.data(), copy.size());

    }

    template<typename T>
    template<typename OpName>
    void NDArray<T>::varianceAlongDimension(const NDArray<T> *target, const bool biasCorrected, const std::initializer_list<int>& dimensions) {
         varianceAlongDimension<OpName>(target, biasCorrected, std::vector<int>(dimensions));
    }

    ////////////////////////////////////////////////////////////////////////
    // operator returns sub-array with buffer pointing at this->_buffer + certain offset
    template<typename T>
    NDArray<T> NDArray<T>::operator()(const int* idx, bool keepUnitiesInShape)  const {
        
        const int rank = rankOf();
        int *newShape;
        ALLOCATE(newShape, _workspace, shape::shapeInfoLength(rank), int);
        memcpy(newShape, _shapeInfo, shape::shapeInfoByteLength(rank));
        newShape[shape::shapeInfoLength(rank) - 2] = -1;

        int *shapeOf = shape::shapeOf(newShape);
        int *stridesOf = shape::stride(newShape);

        Nd4jIndex offset = 0;
        int first, last;
        for (int d = 0; d < rank; ++d) {
            // building new shape first
            if (idx[2*d] != idx[2*d+1]) {

                first = idx[2*d]   >= 0 ? idx[2*d]   : idx[2*d]   + sizeAt(d) + 1;
                last  = idx[2*d+1] >= 0 ? idx[2*d+1] : idx[2*d+1] + sizeAt(d) + 1;

                shapeOf[d] = last - first;
                // for offset we're taking only the first index
                offset += first * stridesOf[d];
            }
        }

        NDArray<T> result(_buffer + offset, newShape, _workspace);
        result._isShapeAlloc = true;

        if(!keepUnitiesInShape) {
            // check whether units are present in newShape, if yes then remove them by applying corresponding reshape
            // for example if result has shape {1,a,1,b} then after reshaping it acquire new shape {a,b}
            std::vector<int> nonUnitDims;
            for(int i = 0; i < result.rankOf(); ++i)
                if(newShape[i+1] != 1)
                    nonUnitDims.push_back(newShape[i+1]);

            if(nonUnitDims.size() != result.rankOf())
                result.reshapei(nonUnitDims);
        }

        return result;
    }
    

    ////////////////////////////////////////////////////////////////////////
    // operator returns sub-array with buffer pointing at this->_buffer + certain offset
    template<typename T>
    NDArray<T> NDArray<T>::operator()(const Intervals& idx, bool keepUnitiesInShape)  const {

        const int rank = rankOf();
        if (idx.size() != rank)
            throw "NDArray::operator(Intervals): number of indices should match the rank of array!";

        int *newShape;
        ALLOCATE(newShape, _workspace, shape::shapeInfoLength(rank), int);
        memcpy(newShape, _shapeInfo, shape::shapeInfoByteLength(rank));
        newShape[shape::shapeInfoLength(rank) - 2] = -1;

        int *shapeOf = shape::shapeOf(newShape);
        int *stridesOf = shape::stride(newShape);

        Nd4jIndex offset = 0;
        int first, last;
        for (int d = 0; d < idx.size(); ++d) {
            // building new shape first
            if (!idx[d].empty()) {
                if (idx[d].size() != 2)
                    throw "NDArray::operator(Intervals): the interval must contain only two numbers {first, last} !";
                first = idx[d][0] >= 0 ? idx[d][0] : idx[d][0] + sizeAt(d) + 1;
                last  = idx[d][1] >= 0 ? idx[d][1] : idx[d][1] + sizeAt(d) + 1;

                shapeOf[d] = last - first;
                // for offset we're taking only the first index
                offset += first * stridesOf[d];
            }
        }

        NDArray<T> result(_buffer + offset, newShape, _workspace);
        result._isShapeAlloc = true;

        if(!keepUnitiesInShape) {
            // check whether units are present in newShape, if yes then remove them by applying corresponding reshape
            // for example if result has shape {1,a,1,b} then after reshaping it acquire new shape {a,b}
            std::vector<int> nonUnitDims;
            for(int i = 0; i < result.rankOf(); ++i)
                if(newShape[i+1] != 1)
                    nonUnitDims.push_back(newShape[i+1]);

            if(nonUnitDims.size() != result.rankOf())
                result.reshapei(nonUnitDims);
        }

        return result;
    }
        
////////////////////////////////////////////////////////////////////////
// addition operator array + array
template<typename T>
NDArray<T> NDArray<T>::operator+(const NDArray<T>& other) const {

    if (other.lengthOf() == lengthOf()) {
        NDArray<T> result(this->_shapeInfo, this->_workspace);        
        functions::pairwise_transforms::PairWiseTransform<T>::template exec<simdOps::Add<T>>(this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, result._buffer, result._shapeInfo, nullptr);
        return result;
    }

    return this->template applyTrueBroadcast<simdOps::Add<T>>(other);

}

    ////////////////////////////////////////////////////////////////////////
    // addition operator array + scalar
    template<typename T>
    NDArray<T> NDArray<T>::operator+(const T scalar) const {

        NDArray<T> result(this->_shapeInfo, this->_workspace);
        functions::scalar::ScalarTransform<T>::template transform<simdOps::Add<T>>(this->_buffer, this->_shapeInfo, result._buffer, result._shapeInfo, scalar, nullptr);

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    // addition operator scalar + array
    // template<typename T>
    // NDArray<T> operator+(const T scalar, const NDArray<T>& arr) {
    //     return arr + scalar;
    // }
    ND4J_EXPORT NDArray<float16> operator+(const float16 scalar, const NDArray<float16>& arr) {
        return arr + scalar;        
    }
    ND4J_EXPORT NDArray<float> operator+(const float scalar, const NDArray<float>& arr) {
        return arr + scalar;        
    }
    ND4J_EXPORT NDArray<double> operator+(const double scalar, const NDArray<double>& arr) {
        return arr + scalar;        
    }

    ////////////////////////////////////////////////////////////////////////
    // subtraction operator scalar - array
    // template<typename T>
    // NDArray<T> operator-(const T scalar, const NDArray<T>& arr) {

    //     NDArray<T> result(arr._shapeInfo, false, arr._workspace);
    //     functions::scalar::ScalarTransform<T>::template transform<simdOps::ReverseSubtract<T>>(arr._buffer, arr._shapeInfo, result._buffer, result._shapeInfo, scalar, nullptr);

    //     return result;
    // }    
    ND4J_EXPORT NDArray<float16> operator-(const float16 scalar, const NDArray<float16>& arr) {
        
        NDArray<float16> result(arr._shapeInfo, arr._workspace);
        functions::scalar::ScalarTransform<float16>::template transform<simdOps::ReverseSubtract<float16>>(arr._buffer, arr._shapeInfo, result._buffer, result._shapeInfo, scalar, nullptr);

        return result;
    }        
    ND4J_EXPORT NDArray<float> operator-(const float scalar, const NDArray<float>& arr) {
        
        NDArray<float> result(arr._shapeInfo, arr._workspace);
        functions::scalar::ScalarTransform<float>::template transform<simdOps::ReverseSubtract<float>>(arr._buffer, arr._shapeInfo, result._buffer, result._shapeInfo, scalar, nullptr);

        return result;
    }        
    ND4J_EXPORT NDArray<double> operator-(const double scalar, const NDArray<double>& arr) {
        
        NDArray<double> result(arr._shapeInfo, arr._workspace);
        functions::scalar::ScalarTransform<double>::template transform<simdOps::ReverseSubtract<double>>(arr._buffer, arr._shapeInfo, result._buffer, result._shapeInfo, scalar, nullptr);

        return result;
    }    
    
    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    void NDArray<T>::operator+=(const NDArray<T>& other) {    
        if (!this->isScalar() && other.isScalar()) {
            functions::scalar::ScalarTransform<T>::template transform<simdOps::Add<T>>(this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, other._buffer[0], nullptr);
        } else if (other.lengthOf() == lengthOf())
            functions::pairwise_transforms::PairWiseTransform<T>::template exec<simdOps::Add<T>>(this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, this->_buffer, this->_shapeInfo, nullptr);
        else
            *this = this->template applyTrueBroadcast<simdOps::Add<T>>(other);
    }

    template<typename T>
    void NDArray<T>::operator-=(const NDArray<T>& other) {    

        if (other.lengthOf() == lengthOf())
            functions::pairwise_transforms::PairWiseTransform<T>::template exec<simdOps::Subtract<T>>(this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, this->_buffer, this->_shapeInfo, nullptr);
        else
            *this = this->template applyTrueBroadcast<simdOps::Subtract<T>>(other);
    }

    template<typename T>
    void NDArray<T>::operator+=(const T other) {
        if (this->isScalar())
            this->_buffer[0] += other;
        else
            functions::scalar::ScalarTransform<T>::template transform<simdOps::Add<T>>(this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, other, nullptr);
    }
    
    template<typename T>
    void NDArray<T>::operator-=(const T other) {  
        if (this->isScalar())
            this->_buffer[0] -= other;
        else  
            functions::scalar::ScalarTransform<T>::template transform<simdOps::Subtract<T>>(this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, other, nullptr);
    }

    ////////////////////////////////////////////////////////////////////////
    // subtraction operator array - array
    template<typename T>
    NDArray<T> NDArray<T>::operator-(const NDArray<T>& other) const {
        
        if (other.lengthOf() == lengthOf()) {
            NDArray<T> result(this->_shapeInfo, this->_workspace);
            functions::pairwise_transforms::PairWiseTransform<T>::template exec<simdOps::Subtract<T>>(this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, result._buffer, result._shapeInfo, nullptr);
            return result;
        }

        return this->template applyTrueBroadcast<simdOps::Subtract<T>>(other);
    }

    ////////////////////////////////////////////////////////////////////////
    // subtraction operator array - scalar
    template<typename T>
    NDArray<T> NDArray<T>::operator-(const T& scalar) const {

        NDArray<T> result(this->_shapeInfo, this->_workspace);
        functions::scalar::ScalarTransform<T>::template transform<simdOps::Subtract<T>>(this->_buffer, this->_shapeInfo, result._buffer, result._shapeInfo, scalar, nullptr);

        return result;
    }

    // negative operator, it makes all array elements = -elements
    template<typename T>
    NDArray<T> NDArray<T>::operator-() const {

        NDArray<T> result(this->_shapeInfo, this->_workspace);
        functions::transform::Transform<T>::template exec<simdOps::Neg<T>>(this->_buffer, this->_shapeInfo, result._buffer, result._shapeInfo, nullptr, nullptr, nullptr);

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    // multiplication operator array*array
    template<typename T>
    NDArray<T> NDArray<T>::operator*(const NDArray<T>& other) const {
        
        if (other.lengthOf() == lengthOf()) {
            NDArray<T> result(this->_shapeInfo, this->_workspace);
            functions::pairwise_transforms::PairWiseTransform<T>::template exec<simdOps::Multiply<T>>(this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, result._buffer, result._shapeInfo, nullptr);
            return result;
        }

        return this->template applyTrueBroadcast<simdOps::Multiply<T>>(other);
    }

    ////////////////////////////////////////////////////////////////////////
    // multiplication operator array*scalar
    template<typename T>
    NDArray<T> NDArray<T>::operator*(const T scalar) const {
        
        NDArray<T> result(this->_shapeInfo, this->_workspace);
        functions::scalar::ScalarTransform<T>::template transform<simdOps::Multiply<T>>(this->_buffer, this->_shapeInfo, result._buffer, result._shapeInfo, scalar, nullptr);

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    // multiplication operator array1 *= array2
    template<typename T>
    void NDArray<T>::operator*=(const NDArray<T>& other) {    

        if (!this->isScalar() && other.isScalar()) {
            functions::scalar::ScalarTransform<T>::template transform<simdOps::Multiply<T>>(this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, other._buffer[0], nullptr);
        } else if (other.lengthOf() == lengthOf())
            functions::pairwise_transforms::PairWiseTransform<T>::template exec<simdOps::Multiply<T>>(this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, this->_buffer, this->_shapeInfo, nullptr);
        else
            *this = this->template applyTrueBroadcast<simdOps::Multiply<T>>(other);
    }

    ////////////////////////////////////////////////////////////////////////
    // multiplication operator array*scalar
    template<typename T>
    void NDArray<T>::operator*=(const T scalar) {
        functions::scalar::ScalarTransform<T>::template transform<simdOps::Multiply<T>>(this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, scalar, nullptr);
    }


    ////////////////////////////////////////////////////////////////////////
    // division operator array/array
    template<typename T>
    NDArray<T> NDArray<T>::operator/(const NDArray<T>& other) const {
        
        if (other.lengthOf() == lengthOf()) {
            NDArray<T> result(this->_shapeInfo, this->_workspace);
            functions::pairwise_transforms::PairWiseTransform<T>::template exec<simdOps::Divide<T>>(this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, result._buffer, result._shapeInfo, nullptr);
            return result;
        }

        return this->template applyTrueBroadcast<simdOps::Divide<T>>(other);
    }

    ////////////////////////////////////////////////////////////////////////
    // division operator array / scalar
    template<typename T>
    NDArray<T> NDArray<T>::operator/(const T scalar) const {

        if(scalar == (T)0.)
            throw "NDArray::operator/ (division operator) : division by zero !";
        
        NDArray<T> result(this->_shapeInfo, this->_workspace);
        functions::scalar::ScalarTransform<T>::template transform<simdOps::Divide<T>>(this->_buffer, this->_shapeInfo, result._buffer, result._shapeInfo, scalar, nullptr);

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    // division operator array1 /= array2
    template<typename T>
    void NDArray<T>::operator/=(const NDArray<T>& other) {    

        if (!this->isScalar() && other.isScalar()) {
            functions::scalar::ScalarTransform<T>::template transform<simdOps::Divide<T>>(this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, other._buffer[0], nullptr);
        } else if (other.lengthOf() == lengthOf())
            functions::pairwise_transforms::PairWiseTransform<T>::template exec<simdOps::Divide<T>>(this->_buffer, this->_shapeInfo, other._buffer, other._shapeInfo, this->_buffer, this->_shapeInfo, nullptr);
        else
            *this = this->template applyTrueBroadcast<simdOps::Divide<T>>(other);
    }

    ////////////////////////////////////////////////////////////////////////
    // division operator array /= scalar
    template<typename T>
    void NDArray<T>::operator/=(const T scalar) {
        
        functions::scalar::ScalarTransform<T>::template transform<simdOps::Divide<T>>(this->_buffer, this->_shapeInfo, this->_buffer, this->_shapeInfo, scalar, nullptr);
    }

    ////////////////////////////////////////////////////////////////////////
    // mathematical multiplication of two arrays
    template<typename T>
    NDArray<T> mmul(const NDArray<T>& left, const NDArray<T>& right) {

        NDArray<T>* ptr =  NDArrayFactory<T>::mmulHelper(const_cast<NDArray<T>*>(&left), const_cast<NDArray<T>*>(&right), nullptr, (T)1., (T)0.);
        NDArray<T> result(*ptr);
        delete ptr;
        return result;
    }

    template<typename T>
    DataType NDArray<T>::dataType() const {
        return _dataType;
    }

    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    void NDArray<T>::assign(const NDArray<T>& other, const Intervals& idx) {

        NDArray<T>* subarr = this->subarray(idx);
        subarr->assign(&other);
        delete subarr;
    }

    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    void NDArray<T>::setIdentity() {

        this->assign((T)0.);

        int  rank    = rankOf();
        int* shape   = shapeOf();
        int* strides = stridesOf();
        int  minDim  = 100000000;
        int indices[MAX_RANK];
        for(int j = 0; j < rank; ++j) 
                indices[j] = 1;
        
        Nd4jIndex offset = shape::getOffset(0, shape, strides, indices, rank);
        
        for(int i = 0; i < rank; ++i) 
            if(minDim > shape[i])
                minDim = shape[i];

#pragma omp parallel for if(minDim > Environment::getInstance()->elementwiseThreshold()) schedule(guided) 
        for(int i = 0; i < minDim; ++i)
            _buffer[i*offset] = (T)1.;                
    }

    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    void NDArray<T>::swapUnsafe(NDArray<T>& other) {
        
        if(_buffer == nullptr || other._buffer == nullptr)
            throw "NDArray::swapUnsafe method: input array should not be empty!";

        // if(_buffer == other._buffer)
        //     throw "NDArray::swapUnsafe method: the buffers of input arrays should not point on the same address!";

        if(lengthOf() != other.lengthOf())
            throw "NDArray::swapUnsafe method: input arrays should have the same length!";

        T temp;
#pragma omp parallel for schedule(static) private(temp)
        for (int i = 0; i < lengthOf(); ++i) {
            temp = (*this)(i);
            (*this)(i) = other(i);
            other(i) = temp;            
        }
    }

    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    NDArray<T>* NDArray<T>::diagonal(const char type) const {        
        
        const char order = ordering();
        const int  rank  = rankOf();
        int *outShapeInfo;
        ALLOCATE(outShapeInfo, _workspace, 8, int);
        outShapeInfo[0] = 2;
        outShapeInfo[5] = 0;

        if(isVector() || isScalar()) {
            
            outShapeInfo[1] = outShapeInfo[2] = outShapeInfo[3] = outShapeInfo[4] = 1;
            outShapeInfo[6] = 1;
            outShapeInfo[7] = (int)order;
        }
        else {            
            
            int diagSize  = 100000000;        
            int indices[MAX_RANK];     
                    
            for(int i = 0; i < rank; ++i) {    
                if(diagSize > shapeOf()[i])
                    diagSize = shapeOf()[i];
                indices[i] = 1;
            }
            
            int step = (int)shape::getOffset(0, shapeOf(), stridesOf(), indices, rank);            
                        
            if(type == 'c') {
                outShapeInfo[1] = diagSize;
                outShapeInfo[2] = 1;
            }
            else {
                outShapeInfo[1] = 1;
                outShapeInfo[2] = diagSize;                            
            }
            shape::updateStrides(outShapeInfo, order);
                        
            outShapeInfo[3] *= step;
            outShapeInfo[4] *= step;
            outShapeInfo[6] =  -1;
        }

        NDArray<T>* result = new NDArray<T>(this->_buffer, outShapeInfo, this->_workspace);
        result->_isShapeAlloc = true;
        return result;
    }

////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::setValueInDiagMatrix(const T& value, const int diag, const char direction) {

    if(rankOf() != 2)
       throw std::string("NDArray::setValueInDiagMatrix method: array must have rank = 2, but got " + toStringValue(rankOf()) + " instead !");

    const int rows = sizeAt(0);
    const int cols = sizeAt(1);
        
    switch(direction) {
            
        case 'u':                           // fill upper triangular block
#pragma omp parallel for if(rows > Environment::getInstance()->elementwiseThreshold()) schedule(guided)             
            for(int i = 0; i < rows; ++i) 
                for(int j = 0; j < cols; ++j)                                      
                    if (i + diag <= j)
                        (*this)(i, j) = value;    
                break;

        case 'l':                           // fill lower triangular block
#pragma omp parallel for if(rows > Environment::getInstance()->elementwiseThreshold()) schedule(guided)                         
            for(int i = 0; i < rows; ++i) 
                for(int j = 0; j < cols; ++j)                                      
                    if (i + diag >= j)
                        (*this)(i, j) = value;    
            break;

        default:
            throw std::string("NDArray::setValueInDiagMatrix method: wrong value of direction argument, expected is 'u' or 'l', but got " + std::string(1,direction) + " instead !");
    }  
}

    ////////////////////////////////////////////////////////////////////////
    // default destructor
    template<typename T>
    NDArray<T>::~NDArray() noexcept {
        if (_isBuffAlloc && _workspace == nullptr && _buffer != nullptr)
            delete[] _buffer;

        if (_isShapeAlloc  && _workspace == nullptr && _shapeInfo != nullptr)
            delete[] _shapeInfo;
    }
    
template<typename T>
void NDArray<T>::streamline(char o) {
    char order = o == 'a' ? this->ordering() : o;
    
    int *newShape;
    ALLOCATE(newShape, this->_workspace, shape::shapeInfoLength(this->rankOf()), int);

    T *newBuffer;
    ALLOCATE(newBuffer, this->_workspace, this->lengthOf(), T);

    std::vector<int> shape(this->rankOf());
    for (int e = 0; e < this->rankOf(); e++)
        shape[e] = this->sizeAt(e);

    if (order == 'c')
        shape::shapeBuffer(this->rankOf(), shape.data(), newShape);
    else
        shape::shapeBufferFortran(this->rankOf(), shape.data(), newShape);

    if (!isView()) {

        NativeOpExcutioner<T>::execPairwiseTransform(1, newBuffer, newShape, _buffer, _shapeInfo, newBuffer, newShape, nullptr);
        memcpy(_buffer, newBuffer, this->lengthOf() * sizeOfT());

        //if (_isBuffAlloc)
        //    RELEASE(this->_buffer, this->_workspace);
        if (_isShapeAlloc)
            RELEASE(this->_shapeInfo, this->_workspace);

        //this->_buffer = newBuffer;
        //this->_isBuffAlloc = true;

        RELEASE(newBuffer, this->_workspace);

        this->_shapeInfo = newShape;
        this->_isShapeAlloc = true;
    } else {
        NativeOpExcutioner<T>::execPairwiseTransform(1, newBuffer, newShape, _buffer, _shapeInfo, newBuffer, newShape, nullptr);

        if (_isBuffAlloc)
            RELEASE(this->_buffer, this->_workspace);
        if (_isShapeAlloc)
            RELEASE(this->_shapeInfo, this->_workspace);

        this->_buffer = newBuffer;
        this->_isBuffAlloc = true;

        this->_shapeInfo = newShape;
        this->_isShapeAlloc = true;
    }
}


////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::tileToShape(const std::vector<int>& shape, NDArray<T>* target) {

    if(target != nullptr) {
        this->tile(*target);
        return;
    }

    std::vector<int> thisShape(rankOf());
    for(int i = 0; i < rankOf(); ++i)
        thisShape[i] = sizeAt(i);

    if(!ShapeUtils<T>::areShapesBroadcastable(shape, thisShape))
        throw "NDArray::tileToShape method: the shape of this array and input shape are not suitable for broadcast operation !" ;

    const int newRank = shape.size();
    std::vector<int> repeats(newRank);

    for(int i = 1; i <= newRank; ++i) {
        if(i > rankOf())
            repeats[newRank-i] = shape[newRank - i];
        else
            repeats[newRank-i] = shape[newRank - i] / thisShape[rankOf() - i];
    }

    tilei(repeats);
}

////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::tileToShape(const std::initializer_list<int>& shape, NDArray<T>* target) {

    const std::vector<int> shapeV(shape);
    tileToShape(shapeV, target);
}

////////////////////////////////////////////////////////////////////////
template<typename T>
T NDArray<T>::getTrace() const {
    
    int  rank    = rankOf();
    int* shape   = shapeOf();
    int* strides = stridesOf();
    int  minDim  = 100000000;
    
    int indices[MAX_RANK];
    for(int j = 0; j < rank; ++j) 
        indices[j] = 1;
    
    Nd4jIndex offset = shape::getOffset(0, shape, strides, indices, rank);
    
    for(int i = 0; i < rank; ++i) 
        if(minDim > shape[i])
            minDim = shape[i];
    T sum = 0.;

#pragma omp parallel for reduction(sumT:sum) if(minDim > Environment::getInstance()->elementwiseThreshold()) schedule(guided) 
    for(int i = 0; i < minDim; ++i)
        sum += _buffer[i*offset];

    return sum;
}


template class ND4J_EXPORT NDArray<float>;
template class ND4J_EXPORT NDArray<float16>;
template class ND4J_EXPORT NDArray<double>;


template NDArray<float>* NDArray<float>::asT<float>();
template NDArray<float16>* NDArray<float>::asT<float16>();
template NDArray<double>* NDArray<float>::asT<double>();

template NDArray<float>* NDArray<float16>::asT<float>();
template NDArray<float16>* NDArray<float16>::asT<float16>();
template NDArray<double>* NDArray<float16>::asT<double>();

template NDArray<float>* NDArray<double>::asT<float>();
template NDArray<float16>* NDArray<double>::asT<float16>();
template NDArray<double>* NDArray<double>::asT<double>();


#ifndef __CLION_IDE__
#include "NDArray.macro"
#endif
}

#endif

