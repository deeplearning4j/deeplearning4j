/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef NDARRAY_CPP
#define NDARRAY_CPP

#include "../NDArray.h"
#include "../NDArrayFactory.h"
#include "NativeOpExecutioner.h"
#include <memory/Workspace.h>
#include <memory/MemoryRegistrator.h>
#include <ops.h>
#include <ops/gemm.h>
#include <pointercast.h>
#include <stdexcept>
#include <memory>
#include <helpers/logger.h>
#include <loops/pairwise_transform.h>
#include <loops/transform_same.h>
#include <loops/random.h>
#include <loops/broadcasting.h>
#include <indexing/NDIndex.h>
#include <indexing/IndicesList.h>
#include <helpers/ShapeUtils.h>
#include <sstream>
#include <helpers/ArrayUtils.h>
#include <MmulHelper.h>
#include <helpers/threshold.h>
#include <exceptions/datatype_exception.h>
#include <exceptions/cuda_exception.h>
#include <specials_cuda.h>

#include "../NDArray.hpp"

namespace nd4j {

////////////////////////////////////////////////////////////////////////
void* NDArray::operator new(size_t i) {
    if (nd4j::memory::MemoryRegistrator::getInstance()->hasWorkspaceAttached()) {
        nd4j::memory::Workspace* ws = nd4j::memory::MemoryRegistrator::getInstance()->getWorkspace();
        return ws->allocateBytes((Nd4jLong) i);
    } else {
        auto p = malloc(i);
        
        CHECK_ALLOC(p, "Failed to allocate new NDArray", i);
        return p;
    }
}

////////////////////////////////////////////////////////////////////////
void NDArray::operator delete(void* p) {
    
    if (!nd4j::memory::MemoryRegistrator::getInstance()->hasWorkspaceAttached())
        free(p);
}


////////////////////////////////////////////////////////////////////////
// copy constructor
NDArray::NDArray(const NDArray& other) {
    
    _context = other._context;    

    setShapeInfo(ShapeBuilders::copyShapeInfo(other._shapeInfo, false, _context->getWorkspace()));
    _isShapeAlloc = true;

    ALLOCATE_SPECIAL(_bufferD, _context->getWorkspace(), _length * sizeOfT(), int8_t);
    _isBuffDAlloc = true;

    if(other.isActualOnHostSide()) {
        auto res = cudaMemcpy(_bufferD, other._buffer, _length * sizeOfT(), cudaMemcpyHostToDevice);
        if (res != 0)
            throw cuda_exception::build("cudaMemcpy failed", res);
    } else {
        auto res = cudaMemcpy(_bufferD, other._bufferD, _length * sizeOfT(), cudaMemcpyDeviceToDevice);
        if (res != 0)
            throw cuda_exception::build("cudaMemcpy failed", res);
    }        

    tickWriteDevice();
}

////////////////////////////////////////////////////////////////////////
void NDArray::lazyAllocateBuffer() const {
    if (_buffer == nullptr && !this->isEmpty()) {
        int8_t* pB = const_cast<int8_t*>(_buffer);
        ALLOCATE(pB, _context->getWorkspace(), this->lengthOf() * this->sizeOfT(), int8_t);
        syncToHost();
    }
}

////////////////////////////////////////////////////////////////////////
// scalar constructor
NDArray::NDArray(nd4j::DataType dtype, nd4j::graph::LaunchContext* context) {

    setShapeInfo(ShapeBuilders::createScalarShapeInfo(dtype, context->getWorkspace()));

    ALLOCATE_SPECIAL(_bufferD, context->getWorkspace(), sizeOfT(), int8_t);
    _isBuffDAlloc = true;
    cudaMemset(_bufferD, 0, sizeOfT());    
    
    tickWriteDevice();
}

////////////////////////////////////////////////////////////////////////
// creates new NDArray using shape information from "shapeInfo" array, set all elements in new array to be zeros
NDArray::NDArray(Nd4jLong* shapeInfo, const bool copyStrides, nd4j::graph::LaunchContext* context, const bool isShapeAlloc) {
    
    if (shapeInfo == nullptr)
        throw std::runtime_error("NDArray constructor: can't be initalized without shapeinfo");

    if ((int) shapeInfo[0] > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    _context = context;        

    if(!isShapeAlloc) {
        setShapeInfo(ShapeBuilders::copyShapeInfo(shapeInfo, copyStrides, _context->getWorkspace()));       
    }
    else {
        setShapeInfo(shapeInfo);
        if(!copyStrides)
            shape::updateStrides(_shapeInfo, shape::order(shapeInfo));         
    }

    _isShapeAlloc = true;
   
    ALLOCATE_SPECIAL(_bufferD, _context->getWorkspace(), _length * sizeOfT(), int8_t);
    cudaMemset(_bufferD, 0, _length * sizeOfT());
    _isBuffDAlloc = true;        
   
    tickWriteDevice();  
}

////////////////////////////////////////////////////////////////////////
NDArray::NDArray(const char order, const std::vector<Nd4jLong> &shape, const std::vector<double>& data, nd4j::DataType dtype, nd4j::graph::LaunchContext* context) {

    if (shape.empty())
        throw std::runtime_error("NDArray constructor: input shape is empty !");

    if ((int) shape.size() > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    _context = context;

    setShapeInfo(ShapeBuilders::createShapeInfo(dtype, order, shape, _context->getWorkspace()));
    _isShapeAlloc = true;

    if (_length != data.size()) {
        nd4j_printf("NDArray constructor: data size [%i] doesn't match shape length [%i]\n", data.size(), _length);
        throw std::runtime_error("Data size doesn't match shape");
    }

    ALLOCATE(_buffer, _context->getWorkspace(), _length * DataTypeUtils::sizeOf(dtype), int8_t);
    _isBuffAlloc = true;
    
    ALLOCATE_SPECIAL(_bufferD, _context->getWorkspace(), _length * DataTypeUtils::sizeOf(dtype), int8_t);
    _isBuffDAlloc = true;    

    for(Nd4jLong i=0; i < _length; ++i) {
        BUILD_SINGLE_PARTIAL_SELECTOR(dtype, templatedDoubleAssign<, double>(_buffer, i, reinterpret_cast<const void *>(data.data()), i), LIBND4J_TYPES);
    }
        
    syncToDevice();
    tickReadHost();
}

////////////////////////////////////////////////////////////////////////
// do not allocate memory, memory for array is passed from outside
NDArray::NDArray(void *buffer, Nd4jLong *shapeInfo, graph::LaunchContext* context, const bool isBuffAlloc, const bool isShapeAlloc) {

    if (shapeInfo == nullptr)
        throw std::runtime_error("NDArray constructor: can't be initalized without shapeinfo");

    if ((int) shapeInfo[0] > MAX_RANK)
        throw std::invalid_argument("NDArray constructor: rank of NDArray can't exceed 32");

     if(!isShapeAlloc) 
        setShapeInfo(ShapeBuilders::copyShapeInfo(shapeInfo, true, _context->getWorkspace()));
    else 
        setShapeInfo(shapeInfo);
    
    _context = context;
    _isShapeAlloc = true;    
    
    _buffer = reinterpret_cast<int8_t *>(buffer);        
    ALLOCATE_SPECIAL(_bufferD, _context->getWorkspace(), _length * sizeOfT(), int8_t);
    _isBuffDAlloc = true;
    
    if(_buffer != nullptr)
        cudaMemcpy(_bufferD, _buffer, _length * sizeOfT(), cudaMemcpyHostToDevice);                    
    
    _isBuffAlloc = isBuffAlloc;
    
    tickWriteDevice();
    tickReadHost();
}

////////////////////////////////////////////////////////////////////////
NDArray::NDArray(const char order, const std::vector<Nd4jLong> &shape, nd4j::DataType dtype, nd4j::graph::LaunchContext* context) {

    if (shape.empty())
        throw std::runtime_error("NDArray constructor: input shape is empty !");

    if ((int) shape.size() > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    _context = context;

    setShapeInfo(ShapeBuilders::createShapeInfo(dtype, order, shape, _context->getWorkspace()));
    _isShapeAlloc = true;

    ALLOCATE_SPECIAL(_bufferD, _context->getWorkspace(), _length * sizeOfT(), int8_t);
    cudaMemset(_bufferD, '\0', _length * sizeOfT()); // zero all memory
    _isBuffDAlloc = true;    

    tickWriteDevice();
}

////////////////////////////////////////////////////////////////////////
NDArray::NDArray(const NDArray *other, const bool copyStrides, nd4j::graph::LaunchContext* context) {

    _context = context;
    
    setShapeInfo(ShapeBuilders::copyShapeInfo(other->_shapeInfo, copyStrides, _context->getWorkspace()));
    _isShapeAlloc = true;
    
    ALLOCATE_SPECIAL(_bufferD, _context->getWorkspace(), _length * sizeOfT(), int8_t);
    _isBuffDAlloc = true;    

    tickWriteDevice();
}

////////////////////////////////////////////////////////////////////////
NDArray::NDArray(void* buffer, const char order, const std::vector<Nd4jLong> &shape,  nd4j::DataType dtype, nd4j::graph::LaunchContext* context) {
    
    if (shape.empty())
        throw std::runtime_error("NDArray constructor: input shape is empty !");
        
    if ((int) shape.size() > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");
    
    _context = context;

    setShapeInfo(ShapeBuilders::createShapeInfo(dtype, order, shape, _context->getWorkspace()));
    _isShapeAlloc = true;

    _buffer = reinterpret_cast<int8_t *>(buffer);
    
    ALLOCATE_SPECIAL(_bufferD, _context->getWorkspace(), _length * sizeOfT(), int8_t);
    _isBuffDAlloc = true;
    
    if(_buffer != nullptr)
        cudaMemcpy(_bufferD, _buffer, _length * sizeOfT(), cudaMemcpyHostToDevice);
        
    tickWriteDevice();
    tickReadHost(); 
}

////////////////////////////////////////////////////////////////////////
// assignment operator
    NDArray& NDArray::operator=(const NDArray& other) {

    if (this == &other)
        return *this;    

    if (shape::equalsSoft(_shapeInfo, other._shapeInfo) && _dataType == other._dataType) {
        if(!isEmpty())
            this->assign(&other);
    }
    else {
        
        if(_isBuffAlloc && _context->getWorkspace() == nullptr) 
            delete []_buffer;

        if(_isShapeAlloc && _context->getWorkspace() == nullptr) 
            delete []_shapeInfo;
        
        if(_isBuffDAlloc)
            cudaFree(_bufferD);

        if(_isShapeDAlloc)
            cudaFree(_shapeInfoD);
               
        _context= other._context;
        _buffer = nullptr;
              
        setShapeInfo(ShapeBuilders::copyShapeInfo(other._shapeInfo, false, _context->getWorkspace()));    
        _isShapeAlloc = true;

        ALLOCATE_SPECIAL(_bufferD, _context->getWorkspace(), _length * sizeOfT(), int8_t);
        _isBuffDAlloc = true;        
                
        this->assign(&other);
    }

    return *this;
}

//////////////////////////////////////////////////////////////////////////
// perform array transformation
    // void NDArray::applyTransform(nd4j::transform::FloatOps op, void *extraParams) {
    //     applyTransform(op, this, extraParams);
    // }

    // void NDArray::applyTransform(nd4j::transform::AnyOps op, void *extraParams) {
    //     applyTransform(op, this, extraParams);
    // }

    // void NDArray::applyTransform(nd4j::transform::SameOps op, void *extraParams) {
    //     applyTransform(op, this, extraParams);
    // }

    // void NDArray::applyTransform(nd4j::transform::BoolOps op, void *extraParams) {
    //     applyTransform(op, this, extraParams);
    // }

    // void NDArray::applyTransform(nd4j::transform::StrictOps op, void *extraParams) {
    //     applyTransform(op, this, extraParams);
    // }

    // perform array transformation

/*
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
    */

    //////////////////////////////////////////////////////////////////////////
    void NDArray::applyTrueBroadcast(nd4j::BroadcastBoolOpsTuple op, const NDArray* other, NDArray* target, const bool checkTargetShape, ExtraArguments *extraArgs) const {
        if (isS())
            throw std::runtime_error("NDArray::applyTrueBroadcast bool: you can't use this method on String array!");
        if(target == nullptr || other == nullptr)
            throw std::runtime_error("NDArray::applyTrueBroadcast bool method: target or other = nullptr !");
        
        if (isScalar()) {
            NDArray temp(target->_shapeInfo, _dataType, false, _context);
            temp.assign(this);
            temp.applyPairwiseTransform(op.p, other, target,  extraArgs);
            return;
        }
        if (other->isScalar()) {
            this->applyScalarArr(op.s, other, target, extraArgs);
            return;
        }

        const NDArray* min(nullptr), *max(nullptr);
        if(this->rankOf() >= other->rankOf()) {
            max = this;
            min = other;
        }
        else {
            max = other;
            min = this;
        }

        if(checkTargetShape) {
            Nd4jLong* newShapeInfo = nullptr;
            if(!ShapeUtils::evalBroadcastShapeInfo(*max, *min, false, newShapeInfo, _context->getWorkspace()))          // the rank of target array must be equal to max->rankOf)()
                throw std::runtime_error("NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for broadcast operation !");
            if(!shape::equalsSoft(target->_shapeInfo, newShapeInfo) || target->_dataType != DataType::BOOL)
                throw std::runtime_error("NDArray::applyTrueBroadcast bool method: the shape or type of target array is wrong !");
            if(_dataType != other->_dataType)
                throw std::invalid_argument("NDArray::applyTrueBroadcast bool method: this and other arrays must have the same type !");

            // if workspace is not null - do not call delete.
            if (_context->getWorkspace() == nullptr)
                delete[] newShapeInfo;
        }

        NDArray* pTarget = (max->_dataType == target->_dataType) ? target : new NDArray(target->ordering(), target->getShapeAsVector(), max->_dataType, target->_context);
        // check whether max array has to be tiled
        if(!max->isSameShape(target)) {
            // evaluate repeating dimensions for tile operation
            std::vector<Nd4jLong> repeatMax(max->rankOf());
            for(int i = 1; i <= max->rankOf(); ++i)
                repeatMax[i-1] = (target->_shapeInfo[i] / max->_shapeInfo[i]);
            max->tile(repeatMax, *pTarget);
        }
        else
            pTarget->assign(max);

        // check whether min array has to be tiled
        std::vector<Nd4jLong> repeatMin(min->rankOf());
        int product = 1;
        for(int i = min->rankOf(); i >=1 ; --i) {
            repeatMin[i-1] = (target->_shapeInfo[target->rankOf() - min->rankOf() + i] / min->_shapeInfo[i]);
            product *= repeatMin[i-1];
        }

        auto pMin = const_cast<NDArray *>(min);
        if(product != 1 )
            pMin = new NDArray(min->tile(repeatMin));


        std::vector<int> sameDims = ShapeUtils::getDimsWithSameShape(*target, *pMin);

        if(max == this) {
            pTarget->applyBroadcast(op.b, sameDims, pMin, target, extraArgs);
        }
        else {
            auto dimsToExclude = ShapeUtils::evalDimsToExclude(target->rankOf(), sameDims);
            const auto numOfSubArrs = ShapeUtils::getNumOfSubArrs(target->_shapeInfo, dimsToExclude);

            for(Nd4jLong i = 0; i < numOfSubArrs; ++i) {
                NDArray targetSubArr = (*target)(i, dimsToExclude);
                if (pTarget == target)
                    pMin->applyPairwiseTransform(op.p, &targetSubArr, &targetSubArr, extraArgs);
                else {
                    NDArray pTargetSubArr = (*pTarget)(i, dimsToExclude);
                    pMin->applyPairwiseTransform(op.p, &pTargetSubArr, &targetSubArr, extraArgs);
                }
            }
        }

        if(pMin != min)
            delete pMin;
        if(pTarget != target)
            delete pTarget;
    }

    //////////////////////////////////////////////////////////////////////////
    void NDArray::applyTrueBroadcast(nd4j::BroadcastOpsTuple op, const NDArray* other, NDArray* target, const bool checkTargetShape, ExtraArguments *extraArgs) const {
        if (isS())
            throw std::runtime_error("NDArray::applyTrueBroadcast: you can't use this method on String array!");
        if(target == nullptr || other == nullptr)
            throw std::runtime_error("NDArray::applyTrueBroadcast method: target or other = nullptr !");
        if(((op.s == scalar::Divide || op.s == scalar::FloorDiv || op.s == scalar::FloorMod) && other->isB()) || (op.s == scalar::ReverseDivide && this->isB()))
            throw std::runtime_error("NDArray::applyTrueBroadcast method: you can't divide by bool array !");
        //NDArray::registerSpecialUse({target}, {this,other});
        if (isScalar()) {
            target->assign(this);
            target->applyPairwiseTransform(op.p, *other, extraArgs);
            return;
        }
        if (other->isScalar()) {
            const_cast<NDArray*>(this)->applyScalarArr(op.s, other, target, extraArgs);
            return;
        }

        const NDArray* min(nullptr), *max(nullptr);
        if(this->rankOf() >= other->rankOf()) {
            max = this;
            min = other;
        }
        else {
            max = other;
            min = this;
        }

        if(checkTargetShape) {
            Nd4jLong* newShapeInfo = nullptr;
            if(!ShapeUtils::evalBroadcastShapeInfo(*max, *min, false, newShapeInfo, _context->getWorkspace()))          // the rank of target array must be equal to max->rankOf)()
                throw std::runtime_error("NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for broadcast operation !");
            if(!shape::equalsTypesAndShapesSoft(target->getShapeInfo(), newShapeInfo))
                throw std::runtime_error("NDArray::applyTrueBroadcast method: the shape or type of target array is wrong !");

            // if workspace is not null - do not call delete.
            if (_context->getWorkspace() == nullptr)
                delete[] newShapeInfo;
        }

        NDArray* pTarget = (max->_dataType == target->_dataType) ? target : new NDArray(target->ordering(), target->getShapeAsVector(), max->_dataType, target->_context);
        // check whether max array has to be tiled
        if(!max->isSameShape(target)) {
            // evaluate repeating dimensions for tile operation
            std::vector<Nd4jLong> repeatMax(max->rankOf());
            for(int i = 1; i <= max->rankOf(); ++i)
                repeatMax[i-1] = (target->_shapeInfo[i] / max->_shapeInfo[i]);
            max->tile(repeatMax, *pTarget);
        }
        else
            pTarget->assign(max);


        // check whether min array has to be tiled
        std::vector<Nd4jLong> repeatMin(min->rankOf());
        int product = 1;
        for(int i = min->rankOf(); i >=1 ; --i) {
            repeatMin[i-1] = (target->_shapeInfo[target->rankOf() - min->rankOf() + i] / min->_shapeInfo[i]);
            product *= repeatMin[i-1];
        }

        auto pMin = const_cast<NDArray *>(min);
        if(product != 1 )
            pMin = new NDArray(min->tile(repeatMin));

        std::vector<int> sameDims = ShapeUtils::getDimsWithSameShape(*target, *pMin);

        if(max == this) {
            pTarget->applyBroadcast(op.b, sameDims, pMin, target, extraArgs);
        }
        else {
            auto dimsToExclude = ShapeUtils::evalDimsToExclude(target->rankOf(), sameDims);
            const auto numOfSubArrs = ShapeUtils::getNumOfSubArrs(target->_shapeInfo, dimsToExclude);

            for(Nd4jLong i = 0; i < numOfSubArrs; ++i) {
                auto targetSubArr = (*target)(i, dimsToExclude);
                if(pTarget == target)
                    pMin->applyPairwiseTransform(op.p, &targetSubArr, &targetSubArr, extraArgs);
                else {
                    auto pTargetSubArr = (*pTarget)(i, dimsToExclude);
                    pMin->applyPairwiseTransform(op.p, &pTargetSubArr, &targetSubArr, extraArgs);
                }
            }
        }

        if(pMin != min)
            delete pMin;
         if(pTarget != target)
            delete pTarget;
    }

    //////////////////////////////////////////////////////////////////////////
    // return array which is broadcasted from this and argument array
    NDArray* NDArray::broadcast(const NDArray& other) {
	    // the orders must be the same
	    char order = ordering();
	    if(order != other.ordering())
		    throw std::runtime_error("Broadcast method: arrays have different orders!");

	    // recognize shapes with smaller and bigger rank
	    Nd4jLong* biggerShapeInfo = nullptr;
	    Nd4jLong* smallerShapeInfo = nullptr;
	    int smallerRank, biggerRank;
	    if (rankOf() > other.rankOf()) {
		    biggerShapeInfo = _shapeInfo;
		    biggerRank = shape::rank(_shapeInfo);
		    smallerShapeInfo = other._shapeInfo;
		    smallerRank = shape::rank(other._shapeInfo);
	    }
	    else {
		    biggerShapeInfo = other._shapeInfo;
		    biggerRank = shape::rank(other._shapeInfo);
		    smallerShapeInfo = _shapeInfo;
		    smallerRank = shape::rank(_shapeInfo);
	    }

	    // check shapes on consistency
	    int diff = biggerRank - smallerRank;
	    for (int i = smallerRank; i<=1; --i)
		    if(biggerShapeInfo[diff+i] != smallerShapeInfo[i] && biggerShapeInfo[i] != 1 && smallerShapeInfo[i] != 1)
			    throw std::runtime_error("Broadcast method: arrays have incompatible shapes !");

		// create and fill ret shapeInfo
	    auto shapeInfoNew = new Nd4jLong[shape::shapeInfoLength(biggerRank)];
	    memcpy(shapeInfoNew, biggerShapeInfo, shape::shapeInfoByteLength(biggerRank));
	    for (int i = smallerRank; i>=1; --i)
		    if(shapeInfoNew[diff+i] == 1 || smallerShapeInfo[i] == 1)
			    shapeInfoNew[diff+i] *= smallerShapeInfo[i];

	    auto ret = new NDArray(shapeInfoNew, true, _context);
        ShapeUtils::updateStridesAndType(ret->getShapeInfo(), DataTypeUtils::pickPairwiseResultType(_dataType, other._dataType), order);
	    delete []shapeInfoNew;

    	return ret;
    }


    //////////////////////////////////////////////////////////////////////////
    // check whether array's rows (arg=0) or columns (arg=1) create orthogonal basis
    bool NDArray::hasOrthonormalBasis(const int arg) {
        if (isS())
            throw std::runtime_error("NDArray::hasOrthonormalBasis: you can't use this method on String array!");
	    if(rankOf() !=2 )
		    throw std::runtime_error("NDArray::hasOrthBasis method: rank of ndarray is not equal 2 !");

	    if(arg!=0  && arg!=1)
		    throw std::runtime_error("NDArray::hasOrthBasis method: input argument is not equal to 0 or 1 !");

	    const double eps = 1e-5;
        double dot = 0.f;

        if(arg) {					// check whether columns create orthogonal basis
		    for(int j=0; j<columns()-1; ++j)
			    for(int k=j+1; k<columns(); ++k) {
				    for(int i=0; i<rows(); ++i)
					    dot += e<double>(i,j)*e<double>(i,k);

				    if(nd4j::math::nd4j_abs(dot) > eps )
					    return false;

				    dot = 0.f;
			    }

			    for(int j=0; j<columns(); ++j)	{	// check whether norm of column vector = 1
			        for(int i=0; i<rows(); ++i)
				        dot += e<double>(i,j)*e<double>(i,j);
			    if(dot != 0.f && nd4j::math::nd4j_abs(nd4j::math::nd4j_sqrt<double, double>(dot) - 1.f) > eps)
				    return false;

			    dot = 0.f;
		    }
	    }
	    else {						// check whether rows create orthogonal basis
		    for(int i=0; i<rows()-1; ++i)
			    for(int k=i+1; k<rows(); ++k) {
				    for(int j=0; j<columns(); ++j)
					    dot += e<double>(i,j)*e<double>(k,j);

				    if(nd4j::math::nd4j_abs(dot) > eps )
					    return false;

				    dot = 0.;
			    }

		        for(int i=0; i<rows(); ++i) {		// check whether norm of row vector = 1
			        for(int j=0; j<columns(); ++j)
					    dot += e<double>(i,j)*e<double>(i,j);

			        if(dot!= 0. && nd4j::math::nd4j_abs(nd4j::math::nd4j_sqrt<double, double>(dot) - 1.) > eps)
				        return false;
			        dot = 0.;
		        }
	        }
	    return true;
    }

    template <typename T>
    std::vector<T> NDArray::asVectorT() {
        std::vector<T> result(this->lengthOf());

#pragma omp parallel for simd
        for (int e = 0; e < this->lengthOf(); e++)
            result[e] = this->e<T>(e);

        return result;
    }
    BUILD_SINGLE_TEMPLATE(template std::vector, NDArray::asVectorT(), LIBND4J_TYPES);


    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    void NDArray::setValueInDiagMatrix(const T& value, const int diag, const char direction) {
        if (isS())
            throw std::runtime_error("NDArray::setValueInDiagMatrix: you can't use this method on String array!");
        if(rankOf() != 2)
           throw std::string("NDArray::setValueInDiagMatrix method: array must have rank = 2, but got " + toStringValue(rankOf()) + " instead !");
    }
    template void NDArray::setValueInDiagMatrix(const double& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const float& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const float16& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const bfloat16& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const Nd4jLong& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const int& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const int16_t& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const uint8_t& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const int8_t& value, const int diag, const char direction);
    template void NDArray::setValueInDiagMatrix(const bool& value, const int diag, const char direction);

    


    //////////////////////////////////////////////////////////////////////////
// set new order and shape in case of suitable array length
    bool NDArray::reshapei(const char order, const std::vector<Nd4jLong>& cshape) {

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

        std::vector<Nd4jLong> shape(cshape);
        int rank = shape.size();

        // looking for negative in shape

        int numberNegativesOnes = 0;

        Nd4jLong* shape_ = shape.data();
        for (int i = 0; i < (int) shape.size(); i++) {
            if (shape[i] < 0) {
                if (numberNegativesOnes >= 1)
                    throw std::runtime_error("Only one dimension can be negative at once");

                numberNegativesOnes++;

                int shapeLength = 1;
                for (int j = 0; j < (int) shape.size(); j++)
                    if (i != j)
                        shapeLength *= shape_[j];

                Nd4jLong realShape = nd4j::math::nd4j_abs<int>(lengthOf() / shapeLength);
                auto thisNewShape = new Nd4jLong[shape.size()];

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
            throw std::runtime_error("Bad shape!");
        }

        int shapeLength = shape::shapeInfoLength(rank);
        // remember old values

        // we can do this only if there was no permute applied, or there are no weird strides
        if (shape::canReshape(this->rankOf(), this->_shapeInfo, shape.size(), shape.data(), order == 'f')) {
            Nd4jLong *shapeInfoNew;
            ALLOCATE(shapeInfoNew, _context->getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);

            shape::reshapeCF(this->rankOf(), this->_shapeInfo, shape.size(), shape.data(), order == 'f', shapeInfoNew);

            if (_isShapeAlloc)
                RELEASE(_shapeInfo, _context->getWorkspace());

            ArrayOptions::setDataType(shapeInfoNew, this->dataType());
            _shapeInfo = shapeInfoNew;
            _isShapeAlloc = true;
        } else {
            Nd4jLong *shapeInfoNew;
            ALLOCATE(shapeInfoNew, _context->getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);

            if (order == 'c')
                shape::shapeBuffer(shape.size(), dataType(), shape.data(), shapeInfoNew);
            else
                shape::shapeBufferFortran(shape.size(), dataType(), shape.data(), shapeInfoNew);

            int8_t *newBuffer;
            ALLOCATE(newBuffer, _context->getWorkspace(), this->lengthOf() * sizeOfT(), int8_t);

            NativeOpExecutioner::execTransformSame(nullptr, transform::Copy, _buffer, _shapeInfo, _bufferD, _shapeInfoD, newBuffer, shapeInfoNew, nullptr, nullptr, nullptr, nullptr, nullptr);

            if (_isBuffAlloc)
                RELEASE(_buffer, _context->getWorkspace());


            if (_isShapeAlloc)
                RELEASE(_shapeInfo, _context->getWorkspace());

            _buffer = newBuffer;
            _shapeInfo = shapeInfoNew;
            _isShapeAlloc = true;
            _isBuffAlloc = true;
        }

        return true;
    }

    ////////////////////////////////////////////////////////////////////////
    void NDArray::setIdentity() {
        if (isS())
            throw std::runtime_error("NDArray::setIdentity: you can't use this method on String array!");

        this->assign(0.);

        int  rank    = rankOf();
        auto shape   = shapeOf();
        auto strides = stridesOf();
        int  minDim  = 100000000;
        Nd4jLong indices[MAX_RANK];
        for(int j = 0; j < rank; ++j)
            indices[j] = 1;

        Nd4jLong offset = shape::getOffset(0, shape, strides, indices, rank);

        for(int i = 0; i < rank; ++i)
            if(minDim > shape[i])
                minDim = shape[i];

        float v = 1.0f;
#pragma omp parallel for if(minDim > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for(int i = 0; i < minDim; ++i)
            templatedSet<float>(_buffer, i*offset, this->dataType(), &v);
    }

    template <typename T>
    void NDArray::templatedSet(void *buffer, const Nd4jLong xOfsset, nd4j::DataType dtype, const void *value) {
        BUILD_SINGLE_PARTIAL_SELECTOR(dtype, templatedSet< , T>(buffer, xOfsset, value), LIBND4J_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template void NDArray::templatedSet, (void *buffer, const Nd4jLong xOfsset, nd4j::DataType dtype, const void *value), LIBND4J_TYPES);



    template <typename T>
    void NDArray::templatedSwap(void *xBuffer, void *yBuffer, Nd4jLong length) {
        auto x = reinterpret_cast<T *>(xBuffer);
        auto y = reinterpret_cast<T *>(yBuffer);

#pragma omp parallel for simd schedule(static)
        for (int i = 0; i < length; ++i) {
            auto temp = x[i];
            x[i] = y[i];
            y[i] = temp;
        }
    }
    BUILD_SINGLE_TEMPLATE(template void NDArray::templatedSwap, (void *xBuffer, void *yBuffer, Nd4jLong length), LIBND4J_TYPES);

    ////////////////////////////////////////////////////////////////////////
    void NDArray::swapUnsafe(NDArray& other) {
        auto xType = this->dataType();

        if (xType != other.dataType())
            throw std::runtime_error("NDArray::swapUnsage method: both arrays must have the same data type");

        if(_buffer == nullptr || other._buffer == nullptr)
            throw std::runtime_error("NDArray::swapUnsafe method: input array should not be empty!");

        // if(_buffer == other._buffer)
        //     throw std::runtime_error("NDArray::swapUnsafe method: the buffers of input arrays should not point on the same address!");

        if(lengthOf() != other.lengthOf())
            throw std::runtime_error("NDArray::swapUnsafe method: input arrays should have the same length!");

        BUILD_SINGLE_SELECTOR(xType, templatedSwap, (this->_buffer, other.buffer(), this->lengthOf()), LIBND4J_TYPES);
    }

    ////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::diagonal(const char type) const {

        if (isS())
            throw std::runtime_error("NDArray::diagonal: you can't use this method on String array!");

        const char order = ordering();
        const int  rank  = rankOf();
        Nd4jLong *outShapeInfo;
        ALLOCATE(outShapeInfo, _context->getWorkspace(), 8, Nd4jLong);
        outShapeInfo[0] = 2;
        outShapeInfo[5] = 0;

        if(isVector() || isScalar()) {

            outShapeInfo[1] = outShapeInfo[2] = outShapeInfo[3] = outShapeInfo[4] = 1;
            outShapeInfo[6] = 1;
            outShapeInfo[7] = (int)order;
        }
        else {

            int diagSize  = 100000000;
            Nd4jLong indices[MAX_RANK];

            for(int i = 0; i < rank; ++i) {
                if(diagSize > shapeOf()[i])
                    diagSize = shapeOf()[i];
                indices[i] = 1;
            }

            auto step = shape::getOffset(0, shapeOf(), stridesOf(), indices, rank);

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

        ArrayOptions::setDataType(outShapeInfo, this->dataType());

        auto result = new NDArray(this->_buffer, outShapeInfo, this->_context);
        result->_isShapeAlloc = true;
        return result;
    }

    void NDArray::streamline(char o) {
        char order = o == 'a' ? this->ordering() : o;

        Nd4jLong *newShape;
        ALLOCATE(newShape, this->_context->getWorkspace(), shape::shapeInfoLength(this->rankOf()), Nd4jLong);

        int8_t *newBuffer;
        ALLOCATE(newBuffer, this->_context->getWorkspace(), this->lengthOf() * sizeOfT(), int8_t);

        std::vector<Nd4jLong> shape(this->rankOf());
        for (int e = 0; e < this->rankOf(); e++)
            shape[e] = this->sizeAt(e);

        if (order == 'c')
            shape::shapeBuffer(this->rankOf(),dataType(),  shape.data(), newShape);
        else
            shape::shapeBufferFortran(this->rankOf(), dataType(), shape.data(), newShape);

        if (!isView()) {
            NativeOpExecutioner::execTransformSame(nullptr, transform::Copy, _buffer, _shapeInfo, nullptr, nullptr, newBuffer, newShape, nullptr, nullptr, nullptr, nullptr, nullptr);
            memcpy(_buffer, newBuffer, this->lengthOf() * sizeOfT());

            //if (_isBuffAlloc)
            //    RELEASE(this->_buffer, this->_workspace);
            if (_isShapeAlloc)
                RELEASE(this->_shapeInfo, this->_context->getWorkspace());

            //this->_buffer = newBuffer;
            //this->_isBuffAlloc = true;

            RELEASE(newBuffer, this->_context->getWorkspace());

            this->_shapeInfo = newShape;
            this->_isShapeAlloc = true;
        } else {
            NativeOpExecutioner::execTransformSame(nullptr, transform::Copy, _buffer, _shapeInfo, nullptr, nullptr, newBuffer, newShape, nullptr, nullptr, nullptr, nullptr, nullptr);

            if (_isBuffAlloc)
                RELEASE(this->_buffer, this->_context->getWorkspace());
            if (_isShapeAlloc)
                RELEASE(this->_shapeInfo, this->_context->getWorkspace());

            this->_buffer = newBuffer;
            this->_isBuffAlloc = true;

            this->_shapeInfo = newShape;
            this->_isShapeAlloc = true;
        }
    }

    void NDArray::applyPairwiseTransform(nd4j::pairwise::Ops op, const NDArray* other, NDArray *target, ExtraArguments *extraParams) const{
        if (isS())
            throw std::runtime_error("NDArray::applyPairwiseTransform: you can't use this method on String array!");
        if (other->lengthOf() != target->lengthOf())
            throw std::invalid_argument("NDArray::applyPairwiseTransform method - lengths of arrays are mismatched");
        if (target->_dataType != this->_dataType && target->_dataType != other->_dataType)
            throw std::invalid_argument("NDArray::applyPairwiseTransform method - type of target array must be the same as type of this or other array !");
        if (_context == nullptr)
            throw std::runtime_error("Launch context cannot be NULL!!!");
        if (_context->getCudaStream() == nullptr)
            throw std::runtime_error("CUDA stream cannot be NULL!!!");


        if (!this->isActualOnDeviceSide())
            this->syncToDevice();

        if (!other->isActualOnDeviceSide())
            other->syncToDevice();

        NativeOpExecutioner::execPairwiseTransform(_context, op, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, other->_buffer, other->_shapeInfo, other->_bufferD, other->_shapeInfoD, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, extraParams != nullptr ? extraParams->argumentAsT(target->dataType()) : nullptr);

        if(target)
            target->tickWriteDevice();
        else
            this->tickWriteDevice();

        if (extraParams != nullptr)
            this->synchronize();
    }

    void NDArray::syncToHost() const {
        cudaStreamSynchronize(*_context->getCudaStream());
        if (this->_buffer == nullptr) {
            NDArray* constThis =  const_cast<NDArray*>(this); // not recommended solution
            ALLOCATE(constThis->_buffer, constThis->_context->getWorkspace(), constThis->lengthOf() * constThis->sizeOfT(), int8_t);
            constThis->_isBuffAlloc = true;
        }
        cudaMemcpy(this->_buffer, this->_bufferD, this->lengthOf() * this->sizeOfT(), cudaMemcpyDeviceToHost);
        tickReadHost();
    }

    void NDArray::syncToDevice() const {
        cudaMemcpy(this->_bufferD, this->_buffer, this->lengthOf() * this->sizeOfT(), cudaMemcpyHostToDevice);
        tickReadDevice();
    }

    void NDArray::syncShape() const {
        cudaMemcpy(_shapeInfoD, _shapeInfo, shape::shapeInfoByteLength(_shapeInfo), cudaMemcpyHostToDevice);
    }

    template <typename X, typename Y>
    void NDArray::templatedDoubleAssign(void *xBuffer, const Nd4jLong xOffset, const void *yBuffer, const Nd4jLong yOffset) const {
        auto x = reinterpret_cast<X *>(xBuffer);
        const auto y = reinterpret_cast<const Y *>(yBuffer);

        x[xOffset] = static_cast<X>(y[yOffset]);
    }
    BUILD_DOUBLE_TEMPLATE(template void NDArray::templatedDoubleAssign, (void *xBuffer, const Nd4jLong xOffset, const void *yBuffer, const Nd4jLong yOffset) const, LIBND4J_TYPES, LIBND4J_TYPES);

    // This method assigns values of given NDArray to this one
    void NDArray::assign(const NDArray& other) {

        if (this == &other)
            return;

        if (!Environment::getInstance()->isExperimentalBuild() && (this->dataType() != other.dataType() && other.dataType() != DataType::BOOL)) {
            throw datatype_exception::build("NDArray::assign: cannot assign array of different types", this->dataType(), other.dataType());
        }

        if (other.isScalar()) {
            if(this->isScalar()) {
                if (!this->isEmpty() && !other.isEmpty()) {
                    BUILD_DOUBLE_SELECTOR(_dataType, other._dataType, templatedDoubleAssign,
                                          (_buffer, 0, other._buffer, 0), LIBND4J_TYPES, LIBND4J_TYPES);
                    syncToDevice();
                }
                else if (this->isEmpty() != other.isEmpty()) { // need assign non-empty scalar to empty
                    if (other.isEmpty()) {
                        ArrayOptions::setPropertyBit(this->_shapeInfo, ARRAY_EMPTY);
                        syncShape();
                    }
                    else
                        *this = other;
                }
            }
            else {
                syncToDevice();
                NativeOpExecutioner::execScalar(_context, scalar::CopyPws, _buffer, _shapeInfo, _bufferD, _shapeInfoD, _buffer, _shapeInfo, _bufferD, _shapeInfoD, other._buffer, other._shapeInfo, other._bufferD, other._shapeInfoD, nullptr);
            }
            tickWriteDevice();
            return;
        }

        if (other._length != _length) {
            auto shapeThis = ShapeUtils::shapeAsString(this);
            auto shapeThat = ShapeUtils::shapeAsString(&other);
            nd4j_printf("Can't assign new value to the array: this shape %s; other shape: %s\n", shapeThis.c_str(), shapeThat.c_str());
            throw std::runtime_error("Lengths of arrays are mismatched");
        }

        //syncToDevice();

        // memcpy is allowed only for same order && same ews (being equal to 1)
        if (ordering() == other.ordering() && _dataType == other._dataType && ews() == 1 && other.ews() == 1) {
            cudaMemcpy(_bufferD, other._bufferD, _length * sizeOfT(), cudaMemcpyDeviceToDevice);
        }
        else if(_dataType == other._dataType)
            NativeOpExecutioner::execTransformSame(_context, transform::Copy, other._buffer, other._shapeInfo, other._bufferD, other._shapeInfoD, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, nullptr, nullptr);
        else
            NativeOpExecutioner::execPairwiseTransform(_context, pairwise::CopyPws, _buffer, _shapeInfo, _bufferD, _shapeInfoD, other._buffer, other._shapeInfo, other._bufferD, other._shapeInfoD, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr);
        syncToHost();

        tickWriteDevice();
    }

    ////////////////////////////////////////////////////////////////////////
// This method returns new copy of this NDArray, optionally in different order
    NDArray* NDArray::dup(const char newOrder) {

        char order = newOrder == 'a' ? ordering() : newOrder;

        auto outShapeInfo = ShapeBuilders::createShapeInfo(_dataType, order, getShapeAsVector(), _context->getWorkspace());
        void* outBuffer = nullptr;
        //int8_t* outBufferD = nullptr;
        //Nd4jLong* outShapeD = nullptr;
        ALLOCATE(outBuffer, _context->getWorkspace(), _length * sizeOfT(), int8_t);
        auto result = new NDArray(outBuffer, outShapeInfo, _context, true, true);
        //result->setSpecialBuffers(outBufferD, outShapeD);
        //syncToDevice();
        result->assign(*this);

        return result;
    }

    void NDArray::synchronize() const {
        auto res = cudaStreamSynchronize(*(_context->getCudaStream()));
        if (res != 0)
            throw std::runtime_error("Synchronization failed");
    }

    void NDArray::registerSpecialUse(const std::initializer_list<const NDArray*>& writeList, const std::initializer_list<const NDArray*>& readList) {
        // no-op
        for (auto p:writeList) {
            if (!p->isActualOnDeviceSide())
                p->syncToDevice();

            p->tickWriteDevice();
        }

        for (auto p:readList) {
            if (!p->isActualOnDeviceSide())
                p->syncToDevice();

            p->tickReadDevice();
        }
    }
    

//////////////////////////////////////////////////////////////////////////
    template <>
    utf8string NDArray::e(const Nd4jLong i) const {
        if (i >= _length)
            throw std::invalid_argument("NDArray::e(i): input index is out of array length !");

        if (!isS())
            throw std::runtime_error("This method is available for String arrays only");

        lazyAllocateBuffer();
        if(!isActualOnHostSide()) 
            syncToHost();

        auto rp = getOffset(i);
        tickReadHost();
        return *(reinterpret_cast<utf8string**>(_buffer)[rp]);
    }

//////////////////////////////////////////////////////////////////////////
    template <>
    std::string NDArray::e(const Nd4jLong i) const {
        
        lazyAllocateBuffer();
        if(!isActualOnHostSide())
            syncToHost();

        auto u = e<utf8string>(i);
        std::string r(u._buffer);
        tickReadHost();
        return r;
    }

//////////////////////////////////////////////////////////////////////////
    template <typename T>
    T NDArray::e(const Nd4jLong i) const {

        if (i >= _length)
            throw std::invalid_argument("NDArray::e(i): input index is out of array length !");

        lazyAllocateBuffer();
        if(!isActualOnHostSide())
            syncToHost();

        auto rp = getOffset(i);

        BUILD_SINGLE_PARTIAL_SELECTOR(this->dataType(), return templatedGet<, T>(this->_buffer, rp), LIBND4J_TYPES);

        tickReadHost();
//        return static_cast<T>(119);
    }
    BUILD_SINGLE_UNCHAINED_TEMPLATE(template , NDArray::e(const Nd4jLong) const, LIBND4J_TYPES);
    //BUILD_DOUBLE_TEMPLATE(template void NDArray::templatedSet, (void *buffer, const Nd4jLong *indices, Y value), LIBND4J_TYPES, LIBND4J_TYPES);


//////////////////////////////////////////////////////////////////////////
// Returns value from 2D matrix by coordinates/indexes
    template <typename T>
    T NDArray::e(const Nd4jLong i, const Nd4jLong j) const {
        if (rankOf() != 2 || i >= shapeOf()[0] || j >= shapeOf()[1])
            throw std::invalid_argument("NDArray::e(i,j): one of input indexes is out of array length or rank!=2 !");

        lazyAllocateBuffer();
        if(!isActualOnHostSide()) 
            syncToHost();

        auto xType = this->dataType();
        Nd4jLong coords[2] = {i, j};
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        //return (*this)(i, j);
        BUILD_SINGLE_PARTIAL_SELECTOR(xType, return templatedGet<, T>(this->_buffer, xOffset), LIBND4J_TYPES);
        
        tickReadHost();
        return static_cast<T>(119);
    }
    BUILD_SINGLE_UNCHAINED_TEMPLATE(template , NDArray::e(const Nd4jLong, const Nd4jLong) const, LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
// returns value from 3D tensor by coordinates
    template <typename T>
    T NDArray::e(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k) const {
        //return (*this)(i, j, k);
        if (rankOf() != 3 || i >= shapeOf()[0] || j >= shapeOf()[1] || k >= shapeOf()[2])
            throw std::invalid_argument("NDArray::e(i,j,k): one of input indexes is out of array length or rank!=3 !");

        lazyAllocateBuffer();
        if(!isActualOnHostSide()) 
            syncToHost();

        auto xType = this->dataType();
        Nd4jLong coords[3] = {i, j, k};
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        BUILD_SINGLE_PARTIAL_SELECTOR(xType, return templatedGet<, T>(this->_buffer, xOffset), LIBND4J_TYPES);
        
        tickReadHost();
        return static_cast<T>(119);
    }
    BUILD_SINGLE_UNCHAINED_TEMPLATE(template , NDArray::e(const Nd4jLong, const Nd4jLong, const Nd4jLong) const, LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
    // returns value from 3D tensor by coordinates
    template <typename T>
    T NDArray::e(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l) const {
        //return (*this)(i, j, k);
        if (rankOf() != 4 || i >= shapeOf()[0] || j >= shapeOf()[1] || k >= shapeOf()[2] || l >= shapeOf()[3])
            throw std::invalid_argument("NDArray::e(i,j,k,l): one of input indexes is out of array length or rank!=4 !");

        lazyAllocateBuffer();
        if(!isActualOnHostSide()) 
            syncToHost();

        auto xType = this->dataType();
        Nd4jLong coords[4] = {i, j, k, l};
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        BUILD_SINGLE_PARTIAL_SELECTOR(xType, return templatedGet<, T>(this->_buffer, xOffset), LIBND4J_TYPES);

        tickReadHost();
        return static_cast<T>(119);
    }
    BUILD_SINGLE_UNCHAINED_TEMPLATE(template , NDArray::e(const Nd4jLong, const Nd4jLong, const Nd4jLong, const Nd4jLong) const, LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::e(const Nd4jLong i) const {
    if (i >= _length)
        throw std::invalid_argument("scalar NDArray::e(i): input index is out of array length !");
    
    NDArray scalar(_dataType, _context);
    
    if(isActualOnHostSide()) {
        cudaMemcpy(scalar._bufferD, bufferWithOffset(getOffset(i)), sizeOfT(), cudaMemcpyHostToDevice);
        tickReadHost();
    }
    else {
        cudaMemcpy(scalar._bufferD, specialBufferWithOffset(getOffset(i)), sizeOfT(), cudaMemcpyDeviceToDevice);
        tickReadDevice();
    }

    scalar.tickWriteDevice();
    return scalar;
}    

////////////////////////////////////////////////////////////////////////
#ifndef __JAVACPP_HACK__

    template<typename T>
    void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<T(T, T, T)>& func, NDArray* target) {
        if (target == nullptr)
            target = this;

        if (second == nullptr) {
            nd4j_printf("applyTriplewiseLambda requires three operands to be valid NDArrays, but Second is NULL\n","");
            throw std::runtime_error("second is null");
        }

        if (third == nullptr) {
            nd4j_printf("applyTriplewiseLambda requires three operands to be valid NDArrays, but Third is NULL\n","");
            throw std::runtime_error("third is null");
        }
        if(_dataType != DataTypeUtils::fromT<T>())
            throw std::runtime_error("NDArray::applyTriplewiseLambda<T> method: wrong template parameter T, its type should be the same as type of this array!");
        if(_dataType != second->_dataType || _dataType != third->_dataType || _dataType != target->_dataType)
            throw std::runtime_error("NDArray::applyTriplewiseLambda<T> method: bother four arrays (this, second, third, target) should have the same type !");

        if (this->lengthOf() != second->lengthOf() || this->lengthOf() != third->lengthOf() || !this->isSameShape(second) || !this->isSameShape(third)) {
            nd4j_printf("applyPairwiseLambda requires both operands to have the same shape\n","");
            throw std::runtime_error("Shapes mismach");
        }

        auto f = this->bufferAsT<T>();
        auto s = second->bufferAsT<T>();
        auto t = third->bufferAsT<T>();
        auto z = target->bufferAsT<T>();

        if (this->ordering() == second->ordering() && this->ordering() == third->ordering()  && this->ordering() == target->ordering() && (this->ews() == 1 && target->ews() == 1) && this->ews() == second->ews() && this->ews() == third->ews()) {
#pragma omp parallel for simd schedule(static)
            for (Nd4jLong e = 0; e < this->lengthOf(); e++)
                z[e] = func(f[e], s[e], t[e]);
        } else {
            if (f == z) {

#pragma omp parallel for schedule(guided)
                for (int e = 0; e < this->lengthOf(); e++) {

                    auto tOffset = this->getOffset(e);
                    auto uOffset = second->getOffset(e);
                    auto vOffset = third->getOffset(e);

                    f[tOffset] = func(f[tOffset], s[uOffset], t[vOffset]);
                }
            } else {

#pragma omp parallel for schedule(guided)
                for (int e = 0; e < this->lengthOf(); e++) {

                    auto tOffset = this->getOffset(e);
                    auto uOffset = second->getOffset(e);
                    auto vOffset = third->getOffset(e);
                    auto zOffset = target->getOffset(e);

                    z[zOffset] = func(f[tOffset], s[uOffset], t[vOffset]);
                }
            }
        }
        target->tickWriteDevice();
    }
    template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<double (double, double, double)>& func, NDArray* target);
    template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<float (float, float, float)>& func, NDArray* target);
    template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<float16 (float16, float16, float16)>& func, NDArray* target);
    template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<bfloat16 (bfloat16, bfloat16, bfloat16)>& func, NDArray* target);
    template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<Nd4jLong (Nd4jLong, Nd4jLong, Nd4jLong)>& func, NDArray* target);
    template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<int (int, int, int)>& func, NDArray* target);
    template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<int16_t (int16_t, int16_t, int16_t)>& func, NDArray* target);
    template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<uint8_t (uint8_t, uint8_t, uint8_t)>& func, NDArray* target);
    template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<int8_t (int8_t, int8_t, int8_t)>& func, NDArray* target);
    template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<bool (bool, bool, bool)>& func, NDArray* target);


    template<typename T>
    void NDArray::applyPairwiseLambda(NDArray* other, const std::function<T(T, T)>& func, NDArray* target) {
        if (target == nullptr)
            target = this;

        if (other == nullptr) {
            nd4j_printf("applyPairwiseLambda requires both operands to be valid NDArrays, but Y is NULL\n","");
            throw std::runtime_error("Other is null");
        }

        if(_dataType != DataTypeUtils::fromT<T>())
            throw std::runtime_error("NDArray::applyPairwiseLambda<T> method: wrong template parameter T, its type should be the same as type of this array!");
        if(_dataType != other->_dataType || _dataType != target->_dataType)
            throw std::runtime_error("NDArray::applyPairwiseLambda<T> method: all three arrays (this, other, target) must have the same type !");

        if (this->lengthOf() != other->lengthOf()) {
            nd4j_printf("applyPairwiseLambda requires both operands to have the same shape\n","");
            throw std::runtime_error("Shapes mismach");
        }

        auto f = this->bufferAsT<T>();
        auto s = other->bufferAsT<T>();
        auto z = target->bufferAsT<T>();

        if (this->ordering() == other->ordering() && this->ordering() == target->ordering() && (this->ews() == 1 && target->ews() == 1) && this->ews() == other->ews()) {
#pragma omp parallel for simd schedule(guided)
            for (int e = 0; e < this->lengthOf(); e++)
                z[e] = func(f[e], s[e]);
        } else {
            if (f == z) {

#pragma omp parallel for schedule(guided)
                for (int e = 0; e < this->lengthOf(); e++) {

                    auto xOffset = this->getOffset(e);
                    auto yOffset = other->getOffset(e);

                    f[xOffset] = func(f[xOffset], s[yOffset]);
                }
            } else {

#pragma omp parallel for schedule(guided)
                for (int e = 0; e < this->lengthOf(); e++) {

                    auto xOffset = this->getOffset(e);
                    auto yOffset = other->getOffset(e);
                    auto zOffset = target->getOffset(e);

                    z[zOffset] = func(f[xOffset], s[yOffset]);
                }
            }
        }
        target->tickWriteDevice();
    }
    template void NDArray::applyPairwiseLambda(NDArray* other, const std::function<double (double, double)>& func, NDArray* target);
    template void NDArray::applyPairwiseLambda(NDArray* other, const std::function<float (float, float)>& func, NDArray* target);
    template void NDArray::applyPairwiseLambda(NDArray* other, const std::function<float16 (float16, float16)>& func, NDArray* target);
    template void NDArray::applyPairwiseLambda(NDArray* other, const std::function<bfloat16 (bfloat16, bfloat16)>& func, NDArray* target);
    template void NDArray::applyPairwiseLambda(NDArray* other, const std::function<Nd4jLong (Nd4jLong, Nd4jLong)>& func, NDArray* target);
    template void NDArray::applyPairwiseLambda(NDArray* other, const std::function<int (int, int)>& func, NDArray* target);
    template void NDArray::applyPairwiseLambda(NDArray* other, const std::function<int16_t (int16_t, int16_t)>& func, NDArray* target);
    template void NDArray::applyPairwiseLambda(NDArray* other, const std::function<uint8_t (uint8_t, uint8_t)>& func, NDArray* target);
    template void NDArray::applyPairwiseLambda(NDArray* other, const std::function<int8_t (int8_t, int8_t)>& func, NDArray* target);
    template void NDArray::applyPairwiseLambda(NDArray* other, const std::function<bool (bool, bool)>& func, NDArray* target);


////////////////////////////////////////////////////////////////////////
    template<typename T>
    void NDArray::applyLambda(const std::function<T(T)>& func, NDArray* target) {
        if (target == nullptr)
            target = this;

        if(_dataType != DataTypeUtils::fromT<T>())
            throw std::runtime_error("NDArray::applyLambda<T> method: wrong template parameter T, its type should be the same as type of this array!");
        if(_dataType != target->_dataType)
            throw std::runtime_error("NDArray::applyLambda<T> method: types of this and target array should match !");

        auto f = this->bufferAsT<T>();
        auto z = target->bufferAsT<T>();

        if (this->ordering() == target->ordering() && (this->ews() == 1 && target->ews() == 1)) {
#pragma omp parallel for simd schedule(guided)
            for (int e = 0; e < this->lengthOf(); e++)
                z[e] = func(f[e]);
        } else {
            if (f == z) {

#pragma omp parallel for schedule(guided)
                for (int e = 0; e < this->lengthOf(); e++) {

                    auto xOffset = this->getOffset(e);

                    f[xOffset] = func(f[xOffset]);
                }
            } else {

#pragma omp parallel for schedule(guided)
                for (int e = 0; e < this->lengthOf(); e++) {

                    auto xOffset = this->getOffset(e);
                    auto zOffset = target->getOffset(e);

                    z[zOffset] = func(f[xOffset]);
                }
            }
        }
        target->tickWriteDevice();
    }
    template void NDArray::applyLambda(const std::function<double(double)>& func, NDArray* target);
    template void NDArray::applyLambda(const std::function<float(float)>& func, NDArray* target);
    template void NDArray::applyLambda(const std::function<float16(float16)>& func, NDArray* target);
    template void NDArray::applyLambda(const std::function<bfloat16(bfloat16)>& func, NDArray* target);
    template void NDArray::applyLambda(const std::function<Nd4jLong(Nd4jLong)>& func, NDArray* target);
    template void NDArray::applyLambda(const std::function<int16_t(int16_t)>& func, NDArray* target);
    template void NDArray::applyLambda(const std::function<int32_t(int32_t)>& func, NDArray* target);
    template void NDArray::applyLambda(const std::function<uint8_t(uint8_t)>& func, NDArray* target);
    template void NDArray::applyLambda(const std::function<int8_t(int8_t)>& func, NDArray* target);
    template void NDArray::applyLambda(const std::function<bool(bool)>& func, NDArray* target);

    template<typename T>
    void NDArray::applyIndexedLambda(const std::function<T(Nd4jLong, T)>& func, NDArray* target) {
        if (target == nullptr)
            target = this;

        if(_dataType != DataTypeUtils::fromT<T>())
            throw std::runtime_error("NDArray::applyIndexedLambda<T> method: wrong template parameter T, its type should be the same as type of this array!");
        if(_dataType != target->_dataType)
            throw std::runtime_error("NDArray::applyIndexedLambda<T> method: types of this and target array should match !");

        auto f = this->bufferAsT<T>();
        auto z = target->bufferAsT<T>();

        if (this->ordering() == target->ordering() && (this->ews() == 1 && target->ews() == 1)) {
#pragma omp parallel for simd schedule(guided)
            for (Nd4jLong e = 0; e < this->lengthOf(); e++)
                z[e] = func(e, f[e]);
        } else {
            if (f == z) {

#pragma omp parallel for schedule(guided)
                for (Nd4jLong e = 0; e < this->lengthOf(); e++) {

                    auto xOffset = this->getOffset(e);

                    f[xOffset] = func(e, f[xOffset]);
                }
            } else {

#pragma omp parallel for schedule(guided)
                for (Nd4jLong e = 0; e < this->lengthOf(); e++) {

                    auto xOffset = this->getOffset(e);
                    auto zOffset = target->getOffset(e);

                    z[zOffset] = func(e, f[xOffset]);
                }
            }
        }
        target->tickWriteDevice();
    }
    template void NDArray::applyIndexedLambda(const std::function<double(Nd4jLong, double)>& func, NDArray* target);
    template void NDArray::applyIndexedLambda(const std::function<float(Nd4jLong, float)>& func, NDArray* target);
    template void NDArray::applyIndexedLambda(const std::function<float16(Nd4jLong, float16)>& func, NDArray* target);
    template void NDArray::applyIndexedLambda(const std::function<bfloat16(Nd4jLong, bfloat16)>& func, NDArray* target);
    template void NDArray::applyIndexedLambda(const std::function<Nd4jLong(Nd4jLong, Nd4jLong)>& func, NDArray* target);
    template void NDArray::applyIndexedLambda(const std::function<int(Nd4jLong, int)>& func, NDArray* target);
    template void NDArray::applyIndexedLambda(const std::function<int16_t(Nd4jLong, int16_t)>& func, NDArray* target);
    template void NDArray::applyIndexedLambda(const std::function<uint8_t (Nd4jLong, uint8_t)>& func, NDArray* target);
    template void NDArray::applyIndexedLambda(const std::function<int8_t(Nd4jLong, int8_t)>& func, NDArray* target);
    template void NDArray::applyIndexedLambda(const std::function<bool(Nd4jLong, bool)>& func, NDArray* target);


    template<typename T>
    void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<T(Nd4jLong, T, T)>& func, NDArray* target) {
        if (target == nullptr)
            target = this;

        if (other == nullptr) {
            nd4j_printf("applyIndexedPairwiseLambda requires both operands to be valid NDArrays, but Y is NULL\n","");
            throw std::runtime_error("Other is null");
        }
        if(_dataType != DataTypeUtils::fromT<T>())
            throw std::runtime_error("NDArray::applyIndexedPairwiseLambda<T> method: wrong template parameter T, its type should be the same as type of this array!");
        if(_dataType != target->_dataType)
            throw std::runtime_error("NDArray::applyIndexedPairwiseLambda<T> method: types of this and target array should match !");
        if (this->lengthOf() != other->lengthOf()) {
            nd4j_printf("applyIndexedPairwiseLambda requires both operands to have the same shape\n","");
            throw std::runtime_error("Shapes mismach");
        }

        auto f = this->bufferAsT<T>();
        auto s = other->bufferAsT<T>();
        auto z = target->bufferAsT<T>();

        if (this->ordering() == other->ordering() && this->ordering() == target->ordering() && (this->ews() == 1 && target->ews() == 1) && this->ews() == other->ews()) {
#pragma omp parallel for simd schedule(guided)
            for (Nd4jLong e = 0; e < this->lengthOf(); e++)
                z[e] = func((Nd4jLong) e, f[e], s[e]);
        } else {
            if (f == z) {

#pragma omp parallel for schedule(guided)
                for (int e = 0; e < this->lengthOf(); e++) {

                    auto xOffset = this->getOffset(e);
                    auto yOffset = other->getOffset(e);

                    f[xOffset] = func((Nd4jLong) e, f[xOffset], s[yOffset]);
                }
            } else {

#pragma omp parallel for schedule(guided)
                for (int e = 0; e < this->lengthOf(); e++) {

                    auto xOffset = this->getOffset(e);
                    auto yOffset = other->getOffset(e);
                    auto zOffset = target->getOffset(e);

                    z[zOffset] = func((Nd4jLong) e, f[xOffset], s[yOffset]);
                }
            }
        }
        target->tickWriteDevice();
    }
    template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<double (Nd4jLong, double, double)>& func, NDArray* target);
    template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<float (Nd4jLong, float, float)>& func, NDArray* target);
    template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<float16 (Nd4jLong, float16, float16)>& func, NDArray* target);
    template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<bfloat16 (Nd4jLong, bfloat16, bfloat16)>& func, NDArray* target);
    template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<Nd4jLong (Nd4jLong, Nd4jLong, Nd4jLong)>& func, NDArray* target);
    template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<int (Nd4jLong, int, int)>& func, NDArray* target);
    template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<int16_t (Nd4jLong, int16_t, int16_t)>& func, NDArray* target);
    template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<uint8_t (Nd4jLong, uint8_t, uint8_t)>& func, NDArray* target);
    template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<int8_t (Nd4jLong, int8_t, int8_t)>& func, NDArray* target);
    template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<bool (Nd4jLong, bool, bool)>& func, NDArray* target);
#endif

//////////////////////////////////////////////////////////////////////////
// perform array transformation
    void NDArray::applyTransform(nd4j::transform::FloatOps op, NDArray *target, ExtraArguments *extraParams) {

        if (isS())
            throw std::runtime_error("NDArray::applyTransform FloatOps: you can't use this method on String array!");

        if (target == nullptr)
            target = this;

        if (!target->isR())
            throw std::runtime_error("NDArray::applyTransform FloatOps: target array must have one of FLOAT types");

        NativeOpExecutioner::execTransformFloat(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, extraParams != nullptr ? extraParams->argumentAsT(target->dataType()) : nullptr, nullptr, nullptr);
        if (target != nullptr)
            target->tickWriteDevice();
        else
            this->tickWriteDevice();
    }

    void NDArray::applyTransform(nd4j::transform::AnyOps op, NDArray *target, ExtraArguments *extraParams) {
        nd4j_printf("Float op %i transform:\n", (int)op);

        if (isS())
            throw std::runtime_error("NDArray::applyTransform FloatOps: you can't use this method on String array!");

        if (target == nullptr)
            target = this;

        NDArray::registerSpecialUse({target}, {this});
        NativeOpExecutioner::execTransformFloat(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, extraParams != nullptr ? extraParams->argumentAsT(target->dataType()) : nullptr, nullptr, nullptr);
        //target->tickWriteDevice();
    }

    void NDArray::applyTransform(nd4j::transform::SameOps op, NDArray *target, ExtraArguments *extraParams) {
        nd4j_printf("Same op %i transform:\n", (int)op);
        if (isS())
            throw std::runtime_error("NDArray::applyTransform SameOps: you can't use this method on String array!");

        if (target == nullptr)
            target = this;

        if (target->dataType() != this->dataType())
            throw std::runtime_error("NDArray::applyTransform SameOps: target array must have the same data type as original array");
        NDArray::registerSpecialUse({target}, {this});
        NativeOpExecutioner::execTransformSame(_context, op, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, extraParams != nullptr ? extraParams->argumentAsT(target->dataType()) : nullptr, nullptr, nullptr);
//        target->tickWriteDevice();
    }

    void NDArray::applyTransform(nd4j::transform::BoolOps op, NDArray *target, ExtraArguments *extraParams) {
        if (isS())
            throw std::runtime_error("NDArray::applyTransform BoolOps: you can't use this method on String array!");

        if (target == nullptr)
            target = this;

        if (!target->isB())
            throw std::runtime_error("NDArray::applyTransform BoolOps: target array must have one of BOOL types");

        NDArray::registerSpecialUse({target}, {this});
        NativeOpExecutioner::execTransformBool(_context, op, this->_buffer, this->_shapeInfo, _bufferD, _shapeInfoD, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, extraParams != nullptr ? extraParams->argumentAsT(this->dataType()) : nullptr, nullptr, nullptr);
    }

    void NDArray::applyTransform(nd4j::transform::StrictOps op, NDArray *target, ExtraArguments *extraParams) {
        if (isS())
            throw std::runtime_error("NDArray::applyTransform StrictOps: you can't use this method on String array!");

        if (target == nullptr)
            target = this;

        if (!this->isR() || !target->isR() || (this->dataType() != target->dataType()))
            throw std::runtime_error("NDArray::applyTransform StrictOps: both Source and Target array must have same FLOAT type !");

        NDArray::registerSpecialUse({target}, {this});
        NativeOpExecutioner::execTransformStrict(_context, op, this->_buffer, this->_shapeInfo, _bufferD, _shapeInfoD, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, extraParams != nullptr ? extraParams->argumentAsT(target->dataType()) : nullptr, nullptr, nullptr);
    }

//////////////////////////////////////////////////////////////////////////
// perform array transformation
    // void NDArray::applyTransform(nd4j::transform::FloatOps op, void *extraParams) {
    //     applyTransform(op, this, extraParams);
    // }

    // void NDArray::applyTransform(nd4j::transform::AnyOps op, void *extraParams) {
    //     applyTransform(op, this, extraParams);
    // }

    // void NDArray::applyTransform(nd4j::transform::SameOps op, void *extraParams) {
    //     applyTransform(op, this, extraParams);
    // }

    // void NDArray::applyTransform(nd4j::transform::BoolOps op, void *extraParams) {
    //     applyTransform(op, this, extraParams);
    // }

    // void NDArray::applyTransform(nd4j::transform::StrictOps op, void *extraParams) {
    //     applyTransform(op, this, extraParams);
    // }

    // perform array transformation
    NDArray NDArray::transform(nd4j::transform::FloatOps op, void *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::transform FloatOps: you can't use this method on String array!");

        NDArray result(this->ordering(), getShapeAsVector(), DataTypeUtils::pickFloatingType(dataType()), this->_context);
        NativeOpExecutioner::execTransformFloat(_context, op, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, extraParams, nullptr, nullptr);
        return result;
    }

    NDArray NDArray::transform(nd4j::transform::SameOps op, void *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::transform SameOps: you can't use this method on String array!");

        NDArray result(this->_shapeInfo, false, this->_context);
        NativeOpExecutioner::execTransformSame(_context, op, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, extraParams, nullptr, nullptr);
        return result;
    }

    NDArray NDArray::transform(nd4j::transform::StrictOps op, void *extraParams) const {
        if (!this->isR())
            throw std::runtime_error("Source array must have one of FLOAT types");

        NDArray result(this->_shapeInfo, false, this->_context);
        NativeOpExecutioner::execTransformStrict(_context, op, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, extraParams, nullptr, nullptr);
        return result;
    }

    NDArray NDArray::transform(nd4j::transform::BoolOps op, void *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::transform BoolOps: you can't use this method on String array!");

        NDArray result(this->ordering(), getShapeAsVector(), nd4j::DataType::BOOL, this->_context);
        NativeOpExecutioner::execTransformBool(_context, op, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, extraParams, nullptr, nullptr);
        return result;
    }

//////////////////////////////////////////////////////////////////////////
// perform pairwise transformation
    void NDArray::applyPairwiseTransform(nd4j::pairwise::Ops op, const NDArray& other, ExtraArguments *extraParams) {
        applyPairwiseTransform(op, &other, this, extraParams);
    }

    void NDArray::applyPairwiseTransform(nd4j::pairwise::BoolOps op, const NDArray *other, NDArray *target, ExtraArguments *extraParams) const{
        if (isS())
            throw std::runtime_error("NDArray::applyPairwiseTransform BoolOps: you can't use this method on String array!");
        if (other->lengthOf() != target->lengthOf())
            throw std::invalid_argument("NDArray::applyPairwiseTransform BoolOps method - lengths of arrays are mismatched");
        if (!target->isB())
            throw std::invalid_argument("NDArray::applyPairwiseTransform BoolOps method - result must have bool type");
        if (_dataType != other->_dataType)
            throw std::invalid_argument("NDArray::applyPairwiseTransform BoolOps method - this and other arrays must have the same type !");

        NDArray::registerSpecialUse({target}, {this,other});
        NativeOpExecutioner::execPairwiseBoolTransform(_context, op, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, other->_buffer, other->_shapeInfo, other->_bufferD, other->_shapeInfoD, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, extraParams != nullptr ? extraParams->argumentAsT(target->dataType()) : nullptr);
    }

//////////////////////////////////////////////////////////////////////////
    void NDArray::applyScalarArr(nd4j::scalar::BoolOps op, const NDArray* scalar, NDArray *target, ExtraArguments *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::applyScalarArr BoolOps: you can't use this method on String array!");
        if (target == nullptr || !target->isB())
            throw std::invalid_argument("NDArray::applyScalarArr bool method: target is nullptr or has not bool type!");
        if (_dataType != scalar->_dataType) {
            nd4j_printf("This dtype: [%i]; scalar dtype: [%i]\n", this->_dataType, scalar->_dataType);
            throw std::invalid_argument("NDArray::applyScalarArr bool method: this and scalar arrays must have the same type!");
        }
        if (!this->isActualOnDeviceSide())
            this->syncToDevice();

        if (!scalar->isActualOnDeviceSide())
            scalar->syncToDevice();
        NDArray::registerSpecialUse({target}, {this, scalar});
        NativeOpExecutioner::execScalarBool(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, scalar->_buffer, scalar->_shapeInfo, scalar->_bufferD, scalar->_shapeInfoD, extraParams != nullptr ? extraParams->argumentAsT(target->dataType()): nullptr);
    }

    template <typename T>
    void NDArray::applyScalar(nd4j::scalar::BoolOps op, const T scalar, NDArray *target, ExtraArguments *extraParams) const {

        auto scalarArr = NDArrayFactory::create<T>(scalar, _context);
        applyScalarArr(op, &scalarArr, target, extraParams);
    }

    template <> void NDArray::applyScalar(nd4j::scalar::BoolOps op, const NDArray* scalar, NDArray *target, ExtraArguments *extraParams) const { throw std::runtime_error("NDArray::applyScalar<NDArray*> method: do not use me!");}
    template void NDArray::applyScalar<double>(nd4j::scalar::BoolOps op, const double scalar, NDArray *target, ExtraArguments *extraParams) const;
    template void NDArray::applyScalar<float>(nd4j::scalar::BoolOps op, const float scalar, NDArray *target, ExtraArguments *extraParams) const;
    template void NDArray::applyScalar<float16>(nd4j::scalar::BoolOps op, const float16 scalar, NDArray *target, ExtraArguments *extraParams) const;
    template void NDArray::applyScalar<bfloat16>(nd4j::scalar::BoolOps op, const bfloat16 scalar, NDArray *target, ExtraArguments *extraParams) const;
    template void NDArray::applyScalar<Nd4jLong>(nd4j::scalar::BoolOps op, const Nd4jLong scalar, NDArray *target, ExtraArguments *extraParams) const;
    template void NDArray::applyScalar<int>(nd4j::scalar::BoolOps op, const int scalar, NDArray *target, ExtraArguments *extraParams) const;
    template void NDArray::applyScalar<int16_t>(nd4j::scalar::BoolOps op, const int16_t scalar, NDArray *target, ExtraArguments *extraParams) const;
    template void NDArray::applyScalar<int8_t>(nd4j::scalar::BoolOps op, const int8_t scalar, NDArray *target, ExtraArguments *extraParams) const;
    template void NDArray::applyScalar<uint8_t>(nd4j::scalar::BoolOps op, const uint8_t scalar, NDArray *target, ExtraArguments *extraParams) const;
    template void NDArray::applyScalar<bool>(nd4j::scalar::BoolOps op, const bool scalar, NDArray *target, ExtraArguments *extraParams) const;

//////////////////////////////////////////////////////////////////////////
    void NDArray::applyScalarArr(nd4j::scalar::Ops op, const NDArray* scalar, NDArray* target, ExtraArguments *extraParams) {
        if (isS())
            throw std::runtime_error("NDArray::applyScalarArr: you can't use this method on String array!");
        if (!scalar->isScalar())
            throw std::invalid_argument("NDArray::applyScalarArr method: operand is not a scalar!");
        if(target == nullptr)
            target = this;
        if(target->_dataType != DataTypeUtils::pickPairwiseResultType(_shapeInfo, scalar->_shapeInfo) && !(target->_dataType == this->_dataType || target->_dataType == scalar->_dataType))
            throw std::invalid_argument("NDArray::applyScalarArr method: wrong type of target array!");

        if (!this->isActualOnDeviceSide())
            this->syncToDevice();

        if (!scalar->isActualOnDeviceSide())
            scalar->syncToDevice();
        NDArray::registerSpecialUse({target}, {this,scalar});
        NativeOpExecutioner::execScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, scalar->getBuffer(), scalar->getShapeInfo(), scalar->_bufferD, scalar->_shapeInfoD, extraParams != nullptr ? extraParams->argumentAsT(target->dataType()) : nullptr);
    }

    template <typename T>
    void NDArray::applyScalar(nd4j::scalar::Ops op, const T scalar, NDArray *target, ExtraArguments *extraParams) {

        auto scalarArr = NDArrayFactory::create<T>(this->dataType(), scalar, this->_context);
        applyScalarArr(op, &scalarArr, target, extraParams);
    }

    template <> void NDArray::applyScalar(nd4j::scalar::Ops op, const NDArray* scalar, NDArray *target, ExtraArguments *extraParams) { throw std::runtime_error("NDArray::applyScalar<NDArray*> method: do not use me!");}
    template void NDArray::applyScalar(nd4j::scalar::Ops op, const double scalar, NDArray *target, ExtraArguments *extraParams);
    template void NDArray::applyScalar(nd4j::scalar::Ops op, const float scalar, NDArray *target, ExtraArguments *extraParams);
    template void NDArray::applyScalar(nd4j::scalar::Ops op, const float16 scalar, NDArray *target, ExtraArguments *extraParams);
    template void NDArray::applyScalar(nd4j::scalar::Ops op, const bfloat16 scalar, NDArray *target, ExtraArguments *extraParams);
    template void NDArray::applyScalar(nd4j::scalar::Ops op, const Nd4jLong scalar, NDArray *target, ExtraArguments *extraParams);
    template void NDArray::applyScalar(nd4j::scalar::Ops op, const int scalar, NDArray *target, ExtraArguments *extraParams);
    template void NDArray::applyScalar(nd4j::scalar::Ops op, const int16_t scalar, NDArray *target, ExtraArguments *extraParams);
    template void NDArray::applyScalar(nd4j::scalar::Ops op, const int8_t scalar, NDArray *target, ExtraArguments *extraParams);
    template void NDArray::applyScalar(nd4j::scalar::Ops op, const uint8_t scalar, NDArray *target, ExtraArguments *extraParams);
    template void NDArray::applyScalar(nd4j::scalar::Ops op, const bool scalar, NDArray *target, ExtraArguments *extraParams);

    //////////////////////////////////////////////////////////////////////////
    void NDArray::applyBroadcast(nd4j::broadcast::Ops op, const std::vector<int>& dimensions, const NDArray* tadArray, NDArray* target, ExtraArguments* extraArgs) {
        if (isS())
            throw std::runtime_error("NDArray::applyBroadcast: you can't use this method on String array!");
        if(((op == broadcast::Divide || op == broadcast::FloorDiv || op == broadcast::FloorMod) && tadArray->isB()) || (op == broadcast::ReverseDivide && this->isB()))
            throw std::runtime_error("NDArray::applyBroadcast: you can't divide by array!");

        if (dimensions.size() == 0)
            return;
        auto result = target == nullptr ? this : target;

        if(result->_dataType != DataTypeUtils::pickPairwiseResultType(_shapeInfo, tadArray->_shapeInfo))
            throw std::invalid_argument("NDArray::applyBroadcast method: wrong type of target array !");
        if(!result->isSameShape(this))
            throw std::invalid_argument("NDArray::applyBroadcast method: this and target arrays must have the same shape !");

        std::vector<int> copy(dimensions);

        //if (dimensions.size() > 1)
        //    std::sort(copy.begin(), copy.end());

        Nd4jLong tadLength = shape::tadLength(this->_shapeInfo, copy.data(), (int) copy.size());
        if (tadLength != tadArray->lengthOf())
            throw std::runtime_error("NDArray::applyBroadcast method: tad length mismatch !");

        shape::TAD tad(this->_shapeInfo, copy.data(), copy.size());
        tad.createTadOnlyShapeInfo();
        tad.createOffsets();
        if (!this->isActualOnDeviceSide())
            this->syncToDevice();

        if (!tadArray->isActualOnDeviceSide())
            tadArray->syncToDevice();

        // prepare input arrays for prepareDataForCuda function
        std::vector<std::pair<void*,size_t>> hostData;
        hostData.emplace_back(copy.data(), copy.size() * sizeof(int));							// 0 -- dimensions
        hostData.emplace_back(tad.tadOnlyShapeInfo, shape::shapeInfoByteLength(tad.tadOnlyShapeInfo));	// 1 -- xTadShapeInfo
        hostData.emplace_back(tad.tadOffsets, tad.numTads * sizeof(Nd4jLong));							// 2 -- xTadOffsets
        std::vector<void*> devicePtrs(hostData.size(), nullptr);

        // create cuda stream and LaunchContext
        cudaError_t cudaResult;
        //cudaStream_t stream;
        //cudaResult = cudaStreamCreate(&stream);	ASSERT_EQ(0, cudaResult);
        cudaStream_t* stream = this->getContext()->getCudaStream();
        // allocate required amount of global device memory and copy host data to it
//    cudaResult = allocateDeviceMem(*pLc, devicePtrs, hostData);	ASSERT_EQ(0, cudaResult);
        for(int i = 0; i < devicePtrs.size(); ++i) {

            cudaResult = cudaMalloc(reinterpret_cast<void **>(&devicePtrs[i]), hostData[i].second);
            if(cudaResult != 0) throw cuda_exception::build("Cannot allocate memory for tads on device", cudaResult);
            cudaMemcpyAsync(devicePtrs[i], hostData[i].first, hostData[i].second, cudaMemcpyHostToDevice, *stream);
        }

        // call cuda kernel which calculates result

        NDArray::registerSpecialUse({target}, {this, const_cast<NDArray*>(tadArray)});

        // TODO: eventually we want separate tads here
        NativeOpExecutioner::execBroadcast(_context, op, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, tadArray->_buffer, tadArray->_shapeInfo, tadArray->_bufferD, tadArray->_shapeInfoD, result->_buffer, result->_shapeInfo, result->_bufferD, result->_shapeInfoD, (int*)devicePtrs[0], (int)copy.size(), (Nd4jLong*)devicePtrs[1], (Nd4jLong*)devicePtrs[2], nullptr, nullptr);
    }

    //////////////////////////////////////////////////////////////////////////
    void NDArray::applyBroadcast(nd4j::broadcast::BoolOps op, const std::vector<int>& dimensions, const NDArray* tadArray, NDArray* target, ExtraArguments* extraArgs) {
        if (isS())
            throw std::runtime_error("NDArray::applyBroadcast BoolOps: you can't use this method on String array!");

        if (dimensions.size() == 0)
            return;

        auto result = target == nullptr ? this : target;

        if(result->_dataType != DataType::BOOL)
            throw std::invalid_argument("NDArray::applyBroadcast bool method: type of target array must be BOOL!");
        if(!result->isSameShape(this))
            throw std::invalid_argument("NDArray::applyBroadcast bool method: this and other arrays must have the same shape !");
        if(_dataType != tadArray->_dataType)
            throw std::invalid_argument("NDArray::applyBroadcast bool method: this and tad arrays must have the same type !");

        std::vector<int> copy(dimensions);

        if (dimensions.size() > 1)
            std::sort(copy.begin(), copy.end());

        Nd4jLong tadLength = shape::tadLength(this->_shapeInfo, copy.data(), (int) copy.size());
        if (tadLength != tadArray->lengthOf())
            throw std::runtime_error("Tad length mismatch");

        shape::TAD tad(this->_shapeInfo, copy.data(), copy.size());
        tad.createTadOnlyShapeInfo();
        tad.createOffsets();
        if (!this->isActualOnDeviceSide())
            this->syncToDevice();

        if (!tadArray->isActualOnDeviceSide())
            tadArray->syncToDevice();

        NDArray::registerSpecialUse({target}, {this, tadArray});
        // TODO: eventually we want separate tads here
        NativeOpExecutioner::execBroadcastBool(_context, op, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD,
                                               tadArray->_buffer, tadArray->_shapeInfo, tadArray->_bufferD, tadArray->_shapeInfoD,
                                               result->_buffer, result->_shapeInfo, result->_bufferD, result->_shapeInfoD, copy.data(), (int)copy.size(), tad.tadOnlyShapeInfo, tad.tadOffsets, nullptr, nullptr);
    }

    //////////////////////////////////////////////////////////////////////////
    NDArray NDArray::applyTrueBroadcast(nd4j::BroadcastOpsTuple op, const NDArray& other, ExtraArguments *extraArgs) const {
        Nd4jLong* newShapeInfo = nullptr;
        if(!ShapeUtils::evalBroadcastShapeInfo(*this, &other, true, newShapeInfo, _context->getWorkspace()))          // the rank of new array = max->rankOf)()
            throw std::runtime_error("NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for broadcast operation !");
        NDArray result(newShapeInfo, true, this->_context);

        // if workspace is not null - do not call delete.
        if (_context->getWorkspace() == nullptr)
            delete[] newShapeInfo;

        this->applyTrueBroadcast(op, &other, &result, false, extraArgs);

        return result;
    }
  
    ////////////////////////////////////////////////////////////////////////
    void NDArray::applyIndexReduce(nd4j::indexreduce::Ops op, NDArray* target, const std::vector<int>& dimensions, const ExtraArguments *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::applyIndexReduce: you can't use this method on String array!");

        if (target->dataType() != nd4j::DataType::INT64)
            throw std::runtime_error("NDArray::applyIndexReduce operations return INT64");
        
        void* params = extraParams != nullptr ? const_cast<ExtraArguments*>(extraParams)->argumentAsT(this->dataType()) : nullptr;

        if (target->isScalar()) {
            //target->_buffer[0] = functions::indexreduce::IndexReduce<T>::template execScalar<OpName>(_buffer, _shapeInfo, const_cast<T*>(extraParams));
            NativeOpExecutioner::execIndexReduceScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, params, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD);

            auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());        
            if (cudaResult != 0) throw cuda_exception::build("NDArray::applyIndexReduce cuda failed !", cudaResult);            
        } 
        else {

            std::vector<int> copy(dimensions);
            shape::checkDimensions(rankOf(), copy);

            shape::TAD xTad(_shapeInfo, copy.data(), copy.size());
            xTad.createTadOnlyShapeInfo();
            xTad.createOffsets();

            // device memory allocation for tads
            Nd4jLong *xTadShapeInfo, *xTadOffsets;
            int* dims;
            
            auto cudaResult = cudaMalloc(reinterpret_cast<void **>(&dims), copy.size() * sizeof(int));
            if (cudaResult != 0) throw cuda_exception::build("NDArray::applyIndexReduce cuda memory allocation failed !", cudaResult);
            
            cudaResult = cudaMalloc(reinterpret_cast<void **>(&xTadShapeInfo), shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo));
            if (cudaResult != 0) throw cuda_exception::build("NDArray::applyIndexReduce cuda memory allocation failed !", cudaResult);
            
            cudaResult = cudaMalloc(reinterpret_cast<void **>(&xTadOffsets), xTad.numTads * sizeof(Nd4jLong));
            if (cudaResult != 0) throw cuda_exception::build("NDArray::applyIndexReduce cuda memory allocation failed !", cudaResult);

            cudaMemcpyAsync(dims, copy.data(), copy.size() * sizeof(int), cudaMemcpyHostToDevice, *_context->getCudaStream());
            cudaMemcpyAsync(xTadShapeInfo, xTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo), cudaMemcpyHostToDevice, *_context->getCudaStream());
            cudaMemcpyAsync(xTadOffsets, xTad.tadOffsets, xTad.numTads * sizeof(Nd4jLong), cudaMemcpyHostToDevice, *_context->getCudaStream());

            NativeOpExecutioner::execIndexReduce(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, params,target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD,dims, copy.size(),xTadShapeInfo, xTadOffsets);

            cudaResult = cudaStreamSynchronize(*_context->getCudaStream());        
            if (cudaResult != 0) throw cuda_exception::build("NDArray::applyIndexReduce cuda failed !", cudaResult);

            cudaFree(dims); cudaFree(xTadShapeInfo); cudaFree(xTadOffsets);
        }
        NDArray::registerSpecialUse({target}, {this});
    }
    
    ////////////////////////////////////////////////////////////////////////
    // reduce dimensions in this array relying on index operations
    NDArray* NDArray::applyIndexReduce(nd4j::indexreduce::Ops op,const std::vector<int>& dimensions, const ExtraArguments* extraParams ) const {
        if (isS())
            throw std::runtime_error("NDArray::applyIndexReduce: you can't use this method on String array!");

        std::vector<int> copy(dimensions);
        shape::checkDimensions(rankOf(), copy);

        auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, false, false, _context->getWorkspace());
        ArrayOptions::setDataType(newShape, nd4j::INT64);
        auto result = new NDArray(newShape, true, _context, true);

        void* params = extraParams != nullptr ? const_cast<ExtraArguments*>(extraParams)->argumentAsT(this->dataType()) : nullptr;        

        if (rankOf() == copy.size()) {
            NativeOpExecutioner::execIndexReduceScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, params, result->_buffer, result->_shapeInfo, result->_bufferD, result->_shapeInfoD);

            auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());        
            if (cudaResult != 0) throw cuda_exception::build("NDArray::applyIndexReduce cuda failed !", cudaResult);            
        } 
        else {            

            shape::TAD xTad(_shapeInfo, copy.data(), copy.size());
            xTad.createTadOnlyShapeInfo();
            xTad.createOffsets();

            // device memory allocation for tads
            Nd4jLong *xTadShapeInfo, *xTadOffsets;
            int* dims;
            
            auto cudaResult = cudaMalloc(reinterpret_cast<void **>(&dims), copy.size() * sizeof(int));
            if (cudaResult != 0) throw cuda_exception::build("NDArray::applyIndexReduce cuda memory allocation failed !", cudaResult);
            
            cudaResult = cudaMalloc(reinterpret_cast<void **>(&xTadShapeInfo), shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo));
            if (cudaResult != 0) throw cuda_exception::build("NDArray::applyIndexReduce cuda memory allocation failed !", cudaResult);
            
            cudaResult = cudaMalloc(reinterpret_cast<void **>(&xTadOffsets), xTad.numTads * sizeof(Nd4jLong));
            if (cudaResult != 0) throw cuda_exception::build("NDArray::applyIndexReduce cuda memory allocation failed !", cudaResult);

            cudaMemcpyAsync(dims, copy.data(), copy.size() * sizeof(int), cudaMemcpyHostToDevice, *_context->getCudaStream());
            cudaMemcpyAsync(xTadShapeInfo, xTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo), cudaMemcpyHostToDevice, *_context->getCudaStream());
            cudaMemcpyAsync(xTadOffsets, xTad.tadOffsets, xTad.numTads * sizeof(Nd4jLong), cudaMemcpyHostToDevice, *_context->getCudaStream());

            NativeOpExecutioner::execIndexReduce(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD,
                                                params,
                                                result->_buffer, result->_shapeInfo, result->_bufferD, result->_shapeInfoD,
                                                dims, copy.size(),
                                                xTadShapeInfo, xTadOffsets);

            cudaResult = cudaStreamSynchronize(*_context->getCudaStream());        
            if (cudaResult != 0) throw cuda_exception::build("NDArray::applyIndexReduce cuda failed !", cudaResult);

            cudaFree(dims); cudaFree(xTadShapeInfo); cudaFree(xTadOffsets);
        }
        
        NDArray::registerSpecialUse({result}, {this});
        
        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    // apply reduce3 operations to this and other array, return result in new output array
    NDArray* NDArray::applyReduce3(nd4j::reduce3::Ops op, const NDArray* other, const ExtraArguments* extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::applyReduce3 method: you can't use this method on String array!");
        if(_dataType != other->_dataType)
            throw std::runtime_error("NDArray::applyReduce3 method: the types of this and other arrays must be the same !");
        // check shapes consistency
        if(!isSameShape(other))
            throw std::runtime_error("NDArray::applyReduce3 method: the shapes of this and other arrays must be the same !");
        // create shapeInfo for scalar
        auto newShape = ShapeBuilders::createScalarShapeInfo(DataTypeUtils::pickFloatingType(_dataType), _context->getWorkspace());
        // create output array (scalar)
        auto result = new NDArray(newShape, true, _context, true);
        // create dynamic array of extra parameters if array extraParams is empty (==nullptr)
        void* params = extraParams != nullptr ? const_cast<ExtraArguments*>(extraParams)->argumentAsT(this->dataType()) : nullptr;        

        NativeOpExecutioner::execReduce3Scalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, params, other->_buffer, other->_shapeInfo, other->_bufferD, other->_shapeInfoD, result->_buffer, result->_shapeInfo, result->_bufferD, result->_shapeInfoD);

        auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());        
        if (cudaResult != 0) throw cuda_exception::build("NDArray::applyReduce3 cuda failed !", cudaResult);

        NDArray::registerSpecialUse({result}, {this, other});

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    // apply reduce3 (exec) operations to this and other array, return result in new output array
    NDArray* NDArray::applyReduce3(nd4j::reduce3::Ops op, const NDArray* other, const std::vector<int>& dimensions, const ExtraArguments* extraParams) const {
        
        if (isS())
            throw std::runtime_error("NDArray::applyReduce3: you can't use this method on String array!");
        if(_dataType != other->_dataType)
            throw std::runtime_error("NDArray::applyReduce3 method: the types of this and other arrays must be the same !");

        std::vector<int> copy(dimensions);
        shape::checkDimensions(rankOf(), copy);
        shape::checkDimensions(other->rankOf(), copy);

        if(!isActualOnDeviceSide()) 
            syncToDevice();

        if(!other->isActualOnDeviceSide())
            other->syncToDevice();

        auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, false, false, _context->getWorkspace());
        ArrayOptions::setDataType(newShape, DataTypeUtils::pickFloatingType(_dataType));
        auto result = new NDArray(newShape, true, _context, true);
        // create temporary dynamic array of extra parameters if array extraParams is empty (==nullptr)
        void* params = extraParams != nullptr ? const_cast<ExtraArguments*>(extraParams)->argumentAsT(this->dataType()) : nullptr;

        // perform calculations
        if(rankOf() == copy.size() && other->rankOf() == copy.size()) {
            NativeOpExecutioner::execReduce3Scalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, params, other->_buffer, other->_shapeInfo, other->_bufferD, other->_shapeInfoD, result->_buffer, result->shapeInfo(), result->specialBuffer(), result->specialShapeInfo());
            auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());        
            if (cudaResult != 0) throw cuda_exception::build("NDArray::applyReduce3 cuda failed !", cudaResult);
        }
        else {
            
            // evaluate xTad data 
            shape::TAD xTad(_shapeInfo, copy.data(), copy.size());
            xTad.createTadOnlyShapeInfo();
            xTad.createOffsets();

            // evaluate yTad data
            shape::TAD yTad(other->_shapeInfo, copy.data(), copy.size());         
            yTad.createTadOnlyShapeInfo();
            yTad.createOffsets();

            if(!shape::equalsSoft(xTad.tadOnlyShapeInfo, yTad.tadOnlyShapeInfo) || (xTad.numTads != yTad.numTads && xTad.numTads != 1 && yTad.numTads != 1))
                throw std::runtime_error("NDArray::applyReduce3 cuda method: arrays tads are inconsistent !");

            // device memory allocation for tads
            Nd4jLong *xTadShapeInfo, *xTadOffsets, *yTadShapeInfo, *yTadOffsets;
            int *dims;

            auto cudaResult = cudaMalloc(reinterpret_cast<void **>(&dims), copy.size() * sizeof(int));
            if (cudaResult != 0) throw cuda_exception::build("NDArray::applyReduce3 cuda memory allocation failed !", cudaResult);
            
            cudaResult = cudaMalloc(reinterpret_cast<void **>(&xTadShapeInfo), shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo));
            if (cudaResult != 0) throw cuda_exception::build("NDArray::applyReduce3 cuda memory allocation failed !", cudaResult);

            cudaResult = cudaMalloc(reinterpret_cast<void **>(&yTadShapeInfo), shape::shapeInfoByteLength(yTad.tadOnlyShapeInfo));
            if (cudaResult != 0) throw cuda_exception::build("NDArray::applyReduce3 cuda memory allocation failed !", cudaResult);

            cudaResult = cudaMalloc(reinterpret_cast<void **>(&xTadOffsets), xTad.numTads * sizeof(Nd4jLong));
            if (cudaResult != 0) throw cuda_exception::build("NDArray::applyReduce3 cuda memory allocation failed !", cudaResult);

            cudaResult = cudaMalloc(reinterpret_cast<void **>(&yTadOffsets), yTad.numTads * sizeof(Nd4jLong));
            if (cudaResult != 0) throw cuda_exception::build("NDArray::applyReduce3 cuda memory allocation failed !", cudaResult);

            cudaMemcpyAsync(dims, copy.data(), copy.size() * sizeof(int), cudaMemcpyHostToDevice, *_context->getCudaStream());
            cudaMemcpyAsync(xTadShapeInfo, xTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo), cudaMemcpyHostToDevice, *_context->getCudaStream());
            cudaMemcpyAsync(yTadShapeInfo, yTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(yTad.tadOnlyShapeInfo), cudaMemcpyHostToDevice, *_context->getCudaStream());
            cudaMemcpyAsync(xTadOffsets, xTad.tadOffsets, xTad.numTads * sizeof(Nd4jLong), cudaMemcpyHostToDevice, *_context->getCudaStream());
            cudaMemcpyAsync(yTadOffsets, yTad.tadOffsets, yTad.numTads * sizeof(Nd4jLong), cudaMemcpyHostToDevice, *_context->getCudaStream());

            NativeOpExecutioner::execReduce3(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, params, other->_buffer, other->_shapeInfo, other->_bufferD, other->_shapeInfoD, result->_buffer, result->_shapeInfo, result->_bufferD, result->_shapeInfoD, dims, copy.size(), xTadShapeInfo, xTadOffsets,  yTadShapeInfo, yTadOffsets);

            cudaResult = cudaStreamSynchronize(*_context->getCudaStream());        
            if (cudaResult != 0) throw cuda_exception::build("NDArray::applyReduce3 cuda failed !", cudaResult);

            cudaFree(dims); cudaFree(xTadShapeInfo); cudaFree(xTadOffsets); cudaFree(yTadShapeInfo); cudaFree(yTadOffsets);            
        }        

        NDArray::registerSpecialUse({result}, {this, other});

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    // apply reduce3 (execAll) operations to this and other array, return result in new output array
    NDArray* NDArray::applyAllReduce3(nd4j::reduce3::Ops op, const NDArray *other, const std::vector<int>& dimensions, const ExtraArguments* extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::applyAllReduce3: you can't use this method on String array!");
        if(_dataType != other->_dataType)
            throw std::runtime_error("NDArray::applyAllReduce3 method: the types of this and other arrays must be the same !");

        // be careful, copy array may undergo changes (sort, transformation of negative dimensions to positive, duplicates removing )
        std::vector<int> copy(dimensions);
        shape::checkDimensions(rankOf(), copy);
        shape::checkDimensions(other->rankOf(), copy);
        
        // create tads
        shape::TAD xTad(_shapeInfo, copy.data(), copy.size());
        xTad.createTadOnlyShapeInfo();
        xTad.createOffsets();

        shape::TAD yTad(other->_shapeInfo, copy.data(), copy.size());
        yTad.createTadOnlyShapeInfo();
        yTad.createOffsets();

        // check tads shapes
        if(!shape::equalsSoft(xTad.tadOnlyShapeInfo, yTad.tadOnlyShapeInfo))
            throw std::runtime_error("NDArray::applyAllReduce3 method: the shapes of array tads are different !");

        // set newShape for output array
        Nd4jLong *newShape = nullptr;
        ALLOCATE(newShape, _context->getWorkspace(), 8, Nd4jLong);
        newShape[0] = 2;        // output rank is always equal to 2 for execAll case
        newShape[1] = xTad.numTads;
        newShape[2] = yTad.numTads;
        ShapeUtils::updateStridesAndType(newShape, DataTypeUtils::pickFloatingType(_dataType), 'c');
        // create output array
        auto result = new NDArray(newShape, true, _context, true);

        // create dynamic array of extra parameters if array extraParams is empty (==nullptr)
        void* params = extraParams != nullptr ? const_cast<ExtraArguments*>(extraParams)->argumentAsT(this->dataType()) : nullptr;

         // device memory allocation for tads
        Nd4jLong *xTadShapeInfo, *xTadOffsets, *yTadShapeInfo, *yTadOffsets;
        int *dims;     
        
        auto cudaResult = cudaMalloc(reinterpret_cast<void **>(&dims), copy.size() * sizeof(int));
        if (cudaResult != 0) throw cuda_exception::build("NDArray::applyAllReduce3 cuda memory allocation failed !", cudaResult);

        cudaResult = cudaMalloc(reinterpret_cast<void **>(&xTadShapeInfo), shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo));
        if (cudaResult != 0) throw cuda_exception::build("NDArray::applyAllReduce3 cuda memory allocation failed !", cudaResult);

        cudaResult = cudaMalloc(reinterpret_cast<void **>(&yTadShapeInfo), shape::shapeInfoByteLength(yTad.tadOnlyShapeInfo));
        if (cudaResult != 0) throw cuda_exception::build("NDArray::applyAllReduce3 cuda memory allocation failed !", cudaResult);

        cudaResult = cudaMalloc(reinterpret_cast<void **>(&xTadOffsets), xTad.numTads * sizeof(Nd4jLong));
        if (cudaResult != 0) throw cuda_exception::build("NDArray::applyAllReduce3 cuda memory allocation failed !", cudaResult);

        cudaResult = cudaMalloc(reinterpret_cast<void **>(&yTadOffsets), yTad.numTads * sizeof(Nd4jLong));
        if (cudaResult != 0) throw cuda_exception::build("NDArray::applyAllReduce3 cuda memory allocation failed !", cudaResult);

        cudaMemcpyAsync(dims, copy.data(), copy.size() * sizeof(int), cudaMemcpyHostToDevice, *_context->getCudaStream());  
        cudaMemcpyAsync(xTadShapeInfo, xTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo), cudaMemcpyHostToDevice, *_context->getCudaStream());
        cudaMemcpyAsync(yTadShapeInfo, yTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(yTad.tadOnlyShapeInfo), cudaMemcpyHostToDevice, *_context->getCudaStream());
        cudaMemcpyAsync(xTadOffsets, xTad.tadOffsets, xTad.numTads * sizeof(Nd4jLong), cudaMemcpyHostToDevice, *_context->getCudaStream());
        cudaMemcpyAsync(yTadOffsets, yTad.tadOffsets, yTad.numTads * sizeof(Nd4jLong), cudaMemcpyHostToDevice, *_context->getCudaStream());

        NativeOpExecutioner::execReduce3All(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, params,other->_buffer, other->_shapeInfo, other->_bufferD, other->_shapeInfoD,result->_buffer,result->_shapeInfo, result->_bufferD, result->_shapeInfoD,dims, copy.size(), xTadShapeInfo, xTadOffsets, yTadShapeInfo, yTadOffsets);

        cudaResult = cudaStreamSynchronize(*_context->getCudaStream());        
        if (cudaResult != 0) throw cuda_exception::build("NDArray::applyAllReduce3 cuda failed !", cudaResult);

        cudaFree(dims); cudaFree(xTadShapeInfo); cudaFree(xTadOffsets); cudaFree(yTadShapeInfo); cudaFree(yTadOffsets);            

        NDArray::registerSpecialUse({result}, {this, other});
        
        return result;
    }

    
////////////////////////////////////////////////////////////////////////
// default destructor
NDArray::~NDArray() noexcept {

    if (_isBuffAlloc && _context->getWorkspace() == nullptr && _buffer != nullptr) {
        if (!isS()) {
            delete[] _buffer;
        } else {
            for (int e = 0; e < lengthOf(); e++) {
                auto t = reinterpret_cast<utf8string**>(_buffer);
                delete t[e];
            };
            delete[] _buffer;
        }
    }

    if (_isShapeAlloc  && _context->getWorkspace() == nullptr && _shapeInfo != nullptr)
        delete[] _shapeInfo;

    if (_isShapeDAlloc)
        RELEASE_SPECIAL(_shapeInfoD, _context->getWorkspace());

    if (_isBuffDAlloc)
        RELEASE_SPECIAL(_bufferD, _context->getWorkspace());
}

//////////////////////////////////////////////////////////////////////////
void NDArray::setShapeInfo(Nd4jLong *shapeInfo) {

    if(_isShapeAlloc && _context->getWorkspace() == nullptr)                
        delete []_shapeInfo;

    if(_isShapeDAlloc)
        cudaFree(_shapeInfoD);
        
    _shapeInfo = shapeInfo;

    if (shapeInfo != nullptr) {

        if(ArrayOptions::arrayType(shapeInfo) == ArrayType::EMPTY)
            _length = 0;
        else
            _length = shape::length(shapeInfo);
        
        _dataType = ArrayOptions::dataType(shapeInfo);
        ALLOCATE_SPECIAL(_shapeInfoD, _context->getWorkspace(), shape::shapeInfoLength(_shapeInfo), Nd4jLong);
        _isShapeDAlloc = true;
        syncShape();
    } 
    else {
        this->_dataType = nd4j::DataType::INHERIT;    
        _shapeInfoD = nullptr;
        _isShapeDAlloc = false;
    }
}

////////////////////////////////////////////////////////////////////////
    // This method returns true if two arrays are equal, with custom or default Eps value of 1e-5, false otherwise
    bool NDArray::equalsTo(const NDArray *other, double eps) const {
        if (this->dataType() != other->dataType() || lengthOf() != other->lengthOf())
            return false;

        // we need to be able to compare [1, len] to [len]
        if ((rankOf() == 1 && other->rankOf() == 2) || (rankOf() == 2 && other->rankOf() == 1)) {
            // FIXME: do something here?
        } else if (!shape::equalsSoft(_shapeInfo, other->_shapeInfo))
            return false;

        NDArray tmp(nd4j::DataType::FLOAT32, _context); // scalar = 0

        if(!isActualOnDeviceSide()) 
            syncToDevice();

        if(!other->isActualOnDeviceSide())
            other->syncToDevice();

        ExtraArguments extras({eps}); 
        NativeOpExecutioner::execReduce3Scalar(_context, reduce3::EqualsWithEps, _buffer, _shapeInfo, _bufferD, _shapeInfoD, extras.argumentAsT(DataType::FLOAT32), other->_buffer, other->_shapeInfo, other->_bufferD, other->_shapeInfoD, tmp.buffer(), tmp.shapeInfo(), tmp._bufferD, tmp._shapeInfoD);

        auto res = cudaStreamSynchronize(*_context->getCudaStream());
        if (res != 0)
            throw cuda_exception::build("NDArray::equalsTo failed", res);

        if (tmp.e<int>(0) > 0)
            return false;

        return true;
    }

    //////////////////////////////////////////////////////////////////////////
    bool NDArray::permutei(const int* dimensions, const int rank) {

        // check if current object is _shapeInfo owner
        if (!_isShapeAlloc) {             // if _shapeInfo is not its own
            _shapeInfo = ShapeUtils::evalPermShapeInfo(dimensions, rank, *this, _context->getWorkspace());
            _isShapeAlloc = true;
        } else {
            if (!nonNull() || rank != rankOf())
                throw std::runtime_error("NDArray::permutei method: wrong arguments in permutei method: either array is nullptr or rank is not suitable!");
            shape::doPermuteShapeInfo(_shapeInfo, dimensions);
        }

        syncShape();

        return true;
    }

    //////////////////////////////////////////////////////////////////////////
    bool NDArray::permutei(const Nd4jLong* dimensions, const int rank) {

        // check if current object is _shapeInfo owner
        if (!_isShapeAlloc) {             // if _shapeInfo is not its own
            _shapeInfo = ShapeUtils::evalPermShapeInfo(dimensions, rank, *this, _context->getWorkspace());
            _isShapeAlloc = true;
        } else {
            if (!nonNull() || rank != rankOf())
                throw std::runtime_error("NDArray::permutei method: wrong arguments in permutei method: either array is nullptr or rank is not suitable!");
            shape::doPermuteShapeInfo(_shapeInfo, dimensions);
        }

        syncShape();
        
        return true;
    }

//////////////////////////////////////////////////////////////////////////
// method reduces array by excluding its shapes along axes present in dimensions vector
void NDArray::reduceAlongDimension(nd4j::reduce::FloatOps op, NDArray* target, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes, const bool checkTargetShape) const {

    if (isS())
        throw std::runtime_error("NDArray::reduceAlongDimension FloatOps: you can't use this method on String array!");
    if (target == nullptr || !target->isR())
        throw std::invalid_argument("NDArray::reduceAlongDimension FloatOps: requires target array to be present and have type form real space!");

    std::vector<int> copy(dimensions);
    shape::checkDimensions(rankOf(), copy);

    if(checkTargetShape) {
        auto newShape = ShapeUtils::evalReduceShapeInfo(target->ordering(), copy, *this, keepDims, supportOldShapes, _context->getWorkspace());
        if(!shape::shapeEquals(newShape, target->getShapeInfo()))
            throw std::runtime_error("NDArray::reduceAlongDimension FloatOps: wrong target shape!");
        RELEASE(newShape, _context->getWorkspace());
    }

    if(rankOf() == copy.size() || copy.empty()) {        
        NativeOpExecutioner::execReduceFloatScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD,nullptr, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD);        

        auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
        if (cudaResult != 0) throw cuda_exception::build("NDArray::applyAllReduce3 cuda failed !", cudaResult);
    }
    else {

        shape::TAD xTad(_shapeInfo, copy.data(), copy.size());
        xTad.createTadOnlyShapeInfo();
        xTad.createOffsets();

         // device memory allocation for tads
        Nd4jLong *xTadShapeInfo, *xTadOffsets;
        int *dims;     
        
        auto cudaResult = cudaMalloc(reinterpret_cast<void **>(&dims), copy.size() * sizeof(int));
        if (cudaResult != 0) throw cuda_exception::build("NDArray::applyAllReduce3 cuda memory allocation failed !", cudaResult);

        cudaResult = cudaMalloc(reinterpret_cast<void **>(&xTadShapeInfo), shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo));
        if (cudaResult != 0) throw cuda_exception::build("NDArray::applyAllReduce3 cuda memory allocation failed !", cudaResult);

        cudaResult = cudaMalloc(reinterpret_cast<void **>(&xTadOffsets), xTad.numTads * sizeof(Nd4jLong));
        if (cudaResult != 0) throw cuda_exception::build("NDArray::applyAllReduce3 cuda memory allocation failed !", cudaResult);

        cudaMemcpyAsync(dims, copy.data(), copy.size() * sizeof(int), cudaMemcpyHostToDevice, *_context->getCudaStream());  
        cudaMemcpyAsync(xTadShapeInfo, xTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo), cudaMemcpyHostToDevice, *_context->getCudaStream());
        cudaMemcpyAsync(xTadOffsets, xTad.tadOffsets, xTad.numTads * sizeof(Nd4jLong), cudaMemcpyHostToDevice, *_context->getCudaStream());

        NativeOpExecutioner::execReduceFloat(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, dims, copy.size(), xTadShapeInfo, xTadOffsets);

        cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
        if (cudaResult != 0) throw cuda_exception::build("NDArray::applyAllReduce3 cuda failed !", cudaResult);

        cudaFree(dims); cudaFree(xTadShapeInfo); cudaFree(xTadOffsets);
    }
    NDArray::registerSpecialUse({target}, {this});
}

//////////////////////////////////////////////////////////////////////////
// This method sets value in linear buffer to position i
    template <typename T>
    void NDArray::p(const Nd4jLong i, const T value) {
        lazyAllocateBuffer();
        if (isActualOnDeviceSide() && !isActualOnHostSide())
            syncToHost();
        if (i >= _length)
            throw std::invalid_argument("NDArray::p(i, value): input index is out of array length !");

        auto rp = getOffset(i);
        const void *pV = reinterpret_cast<const void*>(const_cast<T *>(&value));
        BUILD_SINGLE_PARTIAL_SELECTOR(this->dataType(), templatedSet<, T>(this->_buffer, rp, pV), LIBND4J_TYPES);
        syncToDevice();
    }
    template void NDArray::p(const Nd4jLong i, const double value);
    template void NDArray::p(const Nd4jLong i, const float value);
    template void NDArray::p(const Nd4jLong i, const float16 value);
    template void NDArray::p(const Nd4jLong i, const bfloat16 value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong value);
    template void NDArray::p(const Nd4jLong i, const int value);
    template void NDArray::p(const Nd4jLong i, const int8_t value);
    template void NDArray::p(const Nd4jLong i, const uint8_t value);
    template void NDArray::p(const Nd4jLong i, const int16_t value);
    template void NDArray::p(const Nd4jLong i, const bool value);

    void NDArray::p(const Nd4jLong i, const NDArray& scalar) {
        if(!scalar.isScalar())
            throw std::invalid_argument("NDArray::p method: input array must be scalar!");
        if (i >= _length)
            throw std::invalid_argument("NDArray::p(i, NDArray_scalar): input index is out of array length !");
        // probably wrong args order
        lazyAllocateBuffer();
        auto rp = getOffset(i);
        BUILD_SINGLE_SELECTOR(scalar.dataType(), templatedSet, (_buffer, rp, scalar.dataType(), scalar.getBuffer()), LIBND4J_TYPES);
        // void NDArray::templatedSet(void *buffer, const Nd4jLong xOfsset, nd4j::DataType dtype, void *value)
        syncToDevice();
    }


//////////////////////////////////////////////////////////////////////////
// This method sets value in 2D matrix to position i, j

    template <typename T>
    void NDArray::p(const Nd4jLong i, const Nd4jLong j, const T value) {
        //(*this)(i,j) = value;
        if (rankOf() != 2 || i >= shapeOf()[0] || j >= shapeOf()[1])
            throw std::invalid_argument("NDArray:pe(i,j, value): one of input indexes is out of array length or rank!=2 !");

        lazyAllocateBuffer();
        void *p = reinterpret_cast<void *>(const_cast<T *>(&value));
        auto xType = this->dataType();
        Nd4jLong coords[2] = {i, j};
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        BUILD_SINGLE_PARTIAL_SELECTOR(xType, templatedSet<, T>(this->_buffer, xOffset, p), LIBND4J_TYPES);
        syncToDevice();
    }
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const double value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const float value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const float16 value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const bfloat16 value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const int value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const int8_t value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const uint8_t value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const int16_t value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const bool value);
    // template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const utf8string value);

//////////////////////////////////////////////////////////////////////////
// This method sets value in 3D matrix to position i,j,k
    template <typename T>
    void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const T value) {
        //(*this)(i,j,k) = value;
        if (rankOf() != 3 || i >= shapeOf()[0] || j >= shapeOf()[1] || k >= shapeOf()[2])
            throw std::invalid_argument("NDArray:pe(i,j,k, value): one of input indexes is out of array length or rank!=3 !");
        void *p = reinterpret_cast<void *>(const_cast<T *>(&value));
        auto xType = this->dataType();
        Nd4jLong coords[3] = {i, j, k};
        lazyAllocateBuffer();
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        BUILD_SINGLE_PARTIAL_SELECTOR(xType, templatedSet<, T>(this->_buffer, xOffset, p), LIBND4J_TYPES);
        syncToDevice();
    }
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const double value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const float value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const float16 value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const bfloat16 value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const int value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const int8_t value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const uint8_t value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const int16_t value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const bool value);

//////////////////////////////////////////////////////////////////////////
    template <typename T>
    void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const T value) {
        //(*this)(i,j,k) = value;
        if (rankOf() != 4 || i >= shapeOf()[0] || j >= shapeOf()[1] || k >= shapeOf()[2] || l >= shapeOf()[3])
            throw std::invalid_argument("NDArray::p(i,j,k,l, value): one of input indexes is out of array length or rank!=4 !");
        void *p = reinterpret_cast<void *>(const_cast<T *>(&value));
        auto xType = this->dataType();
        Nd4jLong coords[4] = {i, j, k, l};
        lazyAllocateBuffer();
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        BUILD_SINGLE_PARTIAL_SELECTOR(xType, templatedSet<, T>(this->_buffer, xOffset, p), LIBND4J_TYPES);
        syncToDevice();
    }
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const double value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const float value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const float16 value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const bfloat16 value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const Nd4jLong value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const int value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const int8_t value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const uint8_t value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const int16_t value);
    template void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong l, const bool value);

//////////////////////////////////////////////////////////////////////////
    void* NDArray::specialBufferWithOffset(Nd4jLong offset) const {
        return _bufferD + (offset * sizeOfT());
    }

//////////////////////////////////////////////////////////////////////////    
    NDArray* NDArray::tensorAlongDimension(Nd4jLong index, const std::vector<int>& dimensions) const {
        std::vector<int> copy(dimensions);
        shape::checkDimensions(rankOf(), copy);

        Nd4jLong tadLength = shape::tadLength(this->_shapeInfo, copy.data(), copy.size());
        Nd4jLong numTads = this->lengthOf() / tadLength;

        if (index >= numTads)
            throw std::runtime_error("Can't get index higher than total number of TADs");

        shape::TAD tad(this->_shapeInfo, copy.data(), copy.size());
        tad.createTadOnlyShapeInfo();
        tad.createOffsets();

        Nd4jLong* shapeInfo;
        if (_context->getWorkspace() == nullptr) {
            shapeInfo = new Nd4jLong[shape::shapeInfoLength(tad.tadOnlyShapeInfo)];
        } else {
            shapeInfo = reinterpret_cast<Nd4jLong *>(_context->getWorkspace()->allocateBytes(shape::shapeInfoByteLength(tad.tadOnlyShapeInfo)));
        }
        std::memcpy(shapeInfo, tad.tadOnlyShapeInfo, shape::shapeInfoByteLength(tad.tadOnlyShapeInfo));
        auto array = new NDArray(shapeInfo, _context);
        //cudaFree(array->_bufferD);
        array->_bufferD = (int8_t*)specialBufferWithOffset(tad.tadOffsets[index]); //, array->lengthOf() * DataTypeUtils::sizeOf(dataType()), cudaMemcpyDeviceToDevice);
        array->_isBuffAlloc = false;
        array->_isBuffDAlloc = false;
        array->_isShapeAlloc = true;
        array->_isShapeDAlloc = true;
        array->_isView = true;

        return array;
    }




}



#endif

