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
#include <loops/special_kernels.h>
#include <PointersManager.h>
#include "../NDArray.hpp"
#include <ConstantShapeHelper.h>

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

    setShapeInfo(ShapeDescriptor(other.dataType(), other.ordering(), other.shapeOf(), other.rankOf()));

    ALLOCATE_SPECIAL(_bufferD, _context->getWorkspace(), _length * sizeOfT(), int8_t);
    triggerAllocationFlag(true);

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
    
    if (lengthOf() > 0 && _buffer == nullptr && !isEmpty()) {
        
        NDArray* constThis = const_cast<NDArray*>(this);
        // ALLOCATE(constThis->_buffer, _context->getWorkspace(), this->lengthOf() * this->sizeOfT(), int8_t);
        ALLOCATE(constThis->_buffer, _context->getWorkspace(), (getOffset(_length - 1) + 1) * sizeOfT(), int8_t);
        constThis->triggerAllocationFlag(true);
    }
}   

////////////////////////////////////////////////////////////////////////
// scalar constructor
NDArray::NDArray(nd4j::DataType dtype, nd4j::LaunchContext * context, const bool isScalar) {

    //auto shapeInfo = ShapeBuilders::createScalarShapeInfo(dtype, context->getWorkspace());
   // setShapeInfo(shapeInfo);
    _context = context;
    _isAttached = _context->getWorkspace() != nullptr;

    if (isScalar) {
        setShapeInfo(ShapeDescriptor::scalarDescriptor(dtype));
        ALLOCATE_SPECIAL(_bufferD, context->getWorkspace(), sizeOfT(), int8_t);
        _isBuffDAlloc = true;
        cudaMemset(_bufferD, 0, sizeOfT());
        tickWriteDevice();
    }
    else {
        setShapeInfo(ConstantShapeHelper::getInstance()->emptyShapeInfo(dtype));
    }
}

////////////////////////////////////////////////////////////////////////
// creates new NDArray using shape information from "shapeInfo" array, set all elements in new array to be zeros
NDArray::NDArray(Nd4jLong* shapeInfo, const nd4j::DataType dtype, const bool copyStrides, nd4j::LaunchContext * context) {

    if (shapeInfo == nullptr)
        throw std::runtime_error("NDArray constructor: can't be initalized without shapeinfo");

    if ((int) shapeInfo[0] > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    _context = context;
    if (copyStrides)
        setShapeInfo(ShapeDescriptor(shapeInfo, dtype));
    else
        setShapeInfo(ShapeDescriptor(dtype, shape::order(shapeInfo), shape::shapeOf(shapeInfo), shape::rank(shapeInfo)));

    ALLOCATE_SPECIAL(_bufferD, _context->getWorkspace(), _length * sizeOfT(), int8_t);
    cudaMemset(_bufferD, 0, _length * sizeOfT());
    _isBuffDAlloc = true;
   
    tickWriteDevice();
}

////////////////////////////////////////////////////////////////////////
NDArray::NDArray(const char order, const std::vector<Nd4jLong> &shape, const std::vector<double>& data, nd4j::DataType dtype, nd4j::LaunchContext * context) {

    if (shape.empty())
        throw std::runtime_error("NDArray constructor: input shape is empty !");

    if ((int) shape.size() > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    _context = context;

    setShapeInfo(ShapeDescriptor(dtype, order, shape));

    if (_length != data.size()) {
        nd4j_printf("NDArray constructor: data size [%i] doesn't match shape length [%i]\n", data.size(), _length);
        throw std::runtime_error("Data size doesn't match shape");
    }

    ALLOCATE(_buffer, _context->getWorkspace(), _length * DataTypeUtils::sizeOf(dtype), int8_t);
    triggerAllocationFlag(true);
    
    ALLOCATE_SPECIAL(_bufferD, _context->getWorkspace(), _length * DataTypeUtils::sizeOf(dtype), int8_t);
    _isBuffDAlloc = true;    

    for(Nd4jLong i=0; i < _length; ++i) {
        BUILD_SINGLE_PARTIAL_SELECTOR(dtype, templatedDoubleAssign<, double>(_buffer, i, reinterpret_cast<const void *>(data.data()), i), LIBND4J_TYPES);
    }    
    syncToDevice();
    tickReadHost();
}

////////////////////////////////////////////////////////////////////////
NDArray::NDArray(const char order, const std::vector<Nd4jLong> &shape, nd4j::DataType dtype, nd4j::LaunchContext * context) {

    if (shape.empty())
        throw std::runtime_error("NDArray constructor: input shape is empty !");

    if ((int) shape.size() > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    _context = context;    

    setShapeInfo(ShapeDescriptor(dtype, order, shape));

    ALLOCATE_SPECIAL(_bufferD, _context->getWorkspace(), _length * sizeOfT(), int8_t);
    cudaMemset(_bufferD, '\0', _length * sizeOfT()); // zero all memory
    _isBuffDAlloc = true;

    tickWriteDevice();
}

////////////////////////////////////////////////////////////////////////
NDArray::NDArray(const NDArray *other, const bool copyStrides, nd4j::LaunchContext * context) {
    _context = context;
    
    if (copyStrides)
        setShapeInfo(ShapeDescriptor(other->_shapeInfo, other->dataType()));
    else
        setShapeInfo(ShapeDescriptor(other->dataType(), other->ordering(), other->getShapeAsVector()));
    
    ALLOCATE_SPECIAL(_bufferD, _context->getWorkspace(), _length * sizeOfT(), int8_t);
    _isBuffDAlloc = true;    

    tickWriteDevice();
}

////////////////////////////////////////////////////////////////////////
NDArray::NDArray(void* buffer, const char order, const std::vector<Nd4jLong> &shape,  nd4j::DataType dtype, nd4j::LaunchContext * context) {
    
    if (shape.empty())
        throw std::runtime_error("NDArray constructor: input shape is empty !");
        
    if ((int) shape.size() > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");
    
    _context = context;

    setShapeInfo(ShapeDescriptor(dtype, order, shape));

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
    nd4j_printf("Assignment operator...\n","");
    if (this == &other)
        return *this;    

    if (shape::equalsSoft(_shapeInfo, other._shapeInfo) && _dataType == other._dataType) {
        if(!isEmpty())
            this->assign(&other);
    }
    else {
        
        if(_context->getWorkspace() == nullptr) {
            if(_isBuffAlloc) delete []_buffer;
            if(_isBuffDAlloc)  RELEASE_SPECIAL(_bufferD, nullptr);
        }
               
        _context= other._context;
        _buffer = nullptr;

        //setShapeInfo(ShapeDescriptor(other.dataType(), other.ordering(), other.getShapeInfoAsVector()));
        setShapeInfo(ShapeDescriptor(other.dataType(), other.ordering(), other.shapeOf(), other.rankOf()));
        ALLOCATE_SPECIAL(_bufferD, _context->getWorkspace(), _length * sizeOfT(), int8_t);
        _isBuffDAlloc = true;        
                
        this->assign(&other);
    }
    nd4j_printf("Assigned array with shape rank %lld.\n", this->getShapeInfo()[0]);

    return *this;
}

    //////////////////////////////////////////////////////////////////////////
    void NDArray::applyTrueBroadcast(nd4j::BroadcastBoolOpsTuple op, const NDArray* other, NDArray* target, const bool checkTargetShape, ExtraArguments *extraArgs) const {
        if (isS())
            throw std::runtime_error("NDArray::applyTrueBroadcast bool: you can't use this method on String array!");
        if(target == nullptr || other == nullptr)
            throw std::runtime_error("NDArray::applyTrueBroadcast bool method: target or other = nullptr !");

        NDArray::prepareSpecialUse({target}, {this, other});

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

        NDArray::prepareSpecialUse({target}, {this, other});

        if (isScalar()) {
            target->assign(this);
            target->applyPairwiseTransform(op.p, *other, extraArgs);
            return;
        }
        if (other->isScalar()) {
            const_cast<NDArray*>(this)->applyScalarArr(op.s, other, target, extraArgs);
            return;
        }

        const NDArray* min(other);
        const NDArray* max(this);

        if(this->rankOf() < other->rankOf()) {
            max = other;
            min = this;
        }
        if(checkTargetShape) {
            Nd4jLong* newShapeInfo = nullptr;
            if(!ShapeUtils::evalBroadcastShapeInfo(*max, *min, false, newShapeInfo, _context->getWorkspace()))          // the rank of target array must be equal to max->rankOf)()
                throw std::runtime_error("NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for broadcast operation !");
            if(!shape::equalsTypesAndShapesSoft(target->getShapeInfo(), newShapeInfo))
                throw std::runtime_error("NDArray::applyTrueBroadcast method: the shape or type of target array is wrong !");
        }

        NDArray* pTarget = (max->_dataType == target->_dataType) ? target : new NDArray(target->ordering(), target->getShapeAsVector(), max->_dataType, target->_context);

        // check whether max array has to be tiled
        if(!max->isSameShape(target)) {
            // evaluate repeating dimensions for tile operation
            std::vector<Nd4jLong> repeatMax(max->rankOf());
            for(int i = 1; i <= max->rankOf(); ++i) {
                repeatMax[i - 1] = (target->_shapeInfo[i] / max->_shapeInfo[i]);
                //nd4j_printf("repeatMax[%i] = %i\n", i - 1, repeatMax[i - 1]);
            }
            max->tile(repeatMax, *pTarget);
        }
        else {
            pTarget->assign(max);
        }

        // check whether min array has to be tiled
        std::vector<Nd4jLong> repeatMin(min->rankOf());
        int product = 1;
        for(int i = min->rankOf(); i >=1 ; --i) {
            repeatMin[i-1] = (target->_shapeInfo[target->rankOf() - min->rankOf() + i] / min->_shapeInfo[i]);
            product *= repeatMin[i-1];
        }
        auto pMin = const_cast<NDArray *>(min);
        if(product != 1 ) {
            auto localMin = min->tile(repeatMin);
            pMin = new NDArray(localMin);
        }

        std::vector<int> sameDims = ShapeUtils::getDimsWithSameShape(*target, *pMin);
        //max->syncToDevice();
        //pMin->syncToDevice(); // tile has a problem with syncing data to device        
//        if (sameDims.size() == max->rankOf()) {
//            target->syncToDevice();
//            max->applyPairwiseTransform(op.p, pMin, target, extraArgs);
//        }
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

    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    std::vector<T> NDArray::asVectorT() {
        std::vector<T> result(this->lengthOf());
        lazyAllocateBuffer();

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
           throw std::runtime_error("NDArray::setValueInDiagMatrix method: array must have rank = 2, but got " + toStringValue(rankOf()) + " instead !");
        cudaStream_t* stream = _context->getCudaStream();
        const auto rows = sizeAt(0);
        const auto cols = sizeAt(1);
        if (!isActualOnDeviceSide())
            syncToDevice();
        NDArray val = NDArrayFactory::create(value, _context);
        switch(direction) {
            case 'u':                           // fill upper triangular block
                BUILD_SINGLE_SELECTOR(_dataType, setDiagonalValueUpper, ((void*)_bufferD, _shapeInfoD, val, diag, rows, cols,  *stream), LIBND4J_TYPES);
                break;

            case 'l':                           // fill lower triangular block
                BUILD_SINGLE_SELECTOR(_dataType, setDiagonalValueLower, ((void*)_bufferD, _shapeInfoD, val, diag, rows, cols, *stream), LIBND4J_TYPES);
                break;
            default:
                throw std::string("NDArray::setValueInDiagMatrix method: wrong value of direction argument, expected is 'u' or 'l', but got " + std::string(1,direction) + " instead !");
        }
        tickWriteDevice();
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
        if(order == ordering() && shape::shapeEquals(rankOf(), shapeOf(), cshape.size(), cshape.data()))         
            return true;

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

        if(_bufferD==nullptr || arrLength != this->lengthOf()) {
            this->printShapeInfo("Mismatched shape");
            nd4j::Logger::printv("Shape requested: ", shape);
            nd4j_debug("Requested length in reshape: %i; Existing length: %i;\n", arrLength, this->lengthOf());
            throw std::runtime_error("Bad shape!");
        }

        Nd4jLong *shapeInfoNew;
        ALLOCATE(shapeInfoNew, _context->getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);
       
        bool canReshape = shape::reshapeC(this->rankOf(), this->_shapeInfo, shape.size(), shape.data(), shapeInfoNew);
        
        // we can do this only if there was no permute applied, or there are no weird strides
        if (canReshape) {
            if(ordering() == 'c' && order == 'f')
                throw std::invalid_argument("NDArray::reshapei(order, shape): in case of reshapeC it doesn't make sense to reshape from c order to f order !");

            shape::setEws(shapeInfoNew, arrLength);
            setShapeInfo(ShapeDescriptor(shapeInfoNew, dataType()));
        } 
        else {
        
            NDArray temp(order, shape, dataType(), _context);
            this->applyTransform(transform::Copy, &temp, nullptr);            
            temp.tickWriteDevice();
            *this = std::move(temp);
        }

        RELEASE(shapeInfoNew, _context->getWorkspace());

        return true;
    }

    ////////////////////////////////////////////////////////////////////////
    void NDArray::setIdentity() {
        if (isS())
            throw std::runtime_error("NDArray::setIdentity: you can't use this method on String array!");

        if (rankOf() != 2)
            throw std::runtime_error("NDArray::setIdentity: method should work only for 2D tensors. But " + toStringValue(rankOf()) + " was given.");

        this->assign(1.);

        setValueInDiagMatrix(0.f, 1, 'u');
        setValueInDiagMatrix(0.f, -1, 'l');        
    }

    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    void NDArray::templatedSet(void *buffer, const Nd4jLong xOfsset, nd4j::DataType dtype, const void *value) {
        BUILD_SINGLE_PARTIAL_SELECTOR(dtype, templatedSet< , T>(buffer, xOfsset, value), LIBND4J_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template void NDArray::templatedSet, (void *buffer, const Nd4jLong xOfsset, nd4j::DataType dtype, const void *value), LIBND4J_TYPES);


    ////////////////////////////////////////////////////////////////////////
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

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

        BUILD_SINGLE_SELECTOR(xType, templatedSwapUnsafe, (this->_bufferD, this->_shapeInfoD, other.specialBuffer(), other.specialShapeInfo(), _context->getCudaStream()), LIBND4J_TYPES);
    }

    ////////////////////////////////////////////////////////////////////////
    void NDArray::streamline(char o) {
        char order = o == 'a' ? this->ordering() : o;
        if (!isActualOnDeviceSide())
            syncToDevice();

        Nd4jLong rank = this->rankOf();

        int8_t *newBuffer = nullptr;
        int8_t* newBufferD;
        ALLOCATE(newBuffer, this->_context->getWorkspace(), this->lengthOf() * sizeOfT(), int8_t);
        ALLOCATE_SPECIAL(newBufferD, this->_context->getWorkspace(), this->lengthOf() * sizeOfT(), int8_t);

        std::vector<Nd4jLong> shape(this->rankOf());
        for (int e = 0; e < this->rankOf(); e++)
            shape[e] = this->sizeAt(e);

        auto theNewShape = ShapeBuilders::createShapeInfo(dataType(), order, shape, _context->getWorkspace());

        ShapeDescriptor descriptor(dataType(), order, shape); //theNewShape);
        auto shapeBuffer = ConstantShapeHelper::getInstance()->bufferForShapeInfo(descriptor);

        if (!isView()) {
            NativeOpExecutioner::execTransformSame(_context, transform::Copy, _buffer, _shapeInfo, _bufferD, _shapeInfoD, newBuffer, (Nd4jLong*)shapeBuffer.primary(), newBufferD, (Nd4jLong*)shapeBuffer.special(), nullptr, nullptr, nullptr);
            //memcpy(_buffer, newBuffer, this->lengthOf() * sizeOfT());

            if (_isBuffAlloc) {
                RELEASE(this->_buffer, this->_context->getWorkspace());
            }
            if (_isBuffDAlloc) {
                RELEASE_SPECIAL(_bufferD, this->_context->getWorkspace());
            }

            _buffer == nullptr;
            //_shapeInfo = reinterpret_cast<Nd4jLong *>(shapeBuffer.primary());
            setShapeInfo(descriptor);
            setSpecialBuffer(newBufferD);//, reinterpret_cast<Nd4jLong *>(shapeBuffer.special()));
            //this->_buffer = newBuffer;
            this->_isBuffAlloc = false;
            this->_isBuffDAlloc = true;
        } else {
            NativeOpExecutioner::execTransformSame(_context, transform::Copy, _buffer, _shapeInfo, _bufferD, _shapeInfoD, newBuffer, (Nd4jLong*)shapeBuffer.primary(), newBufferD, (Nd4jLong*)shapeBuffer.special(), nullptr, nullptr, nullptr);

            if (_isBuffAlloc)
                RELEASE(_buffer, this->_context->getWorkspace());

            if (_isBuffDAlloc)
                RELEASE_SPECIAL(_bufferD, this->_context->getWorkspace());

            _buffer = nullptr;
            //setBuffer(newBuffer);
            setShapeInfo(descriptor);
            //_shapeInfo = reinterpret_cast<Nd4jLong *>(shapeBuffer.primary());
            setSpecialBuffer(newBufferD); //, reinterpret_cast<Nd4jLong *>(shapeBuffer.special()));
            this->_isBuffAlloc = false;
            this->_isBuffDAlloc = true;
        }

        //tickReadHost();
        tickWriteDevice();
    }

    ////////////////////////////////////////////////////////////////////////
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

        prepareSpecialUse({target}, {this, other});

        NativeOpExecutioner::execPairwiseTransform(_context, op, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, other->_buffer, other->_shapeInfo, other->_bufferD, other->_shapeInfoD, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, extraParams != nullptr ? extraParams->argumentsAsT(target->dataType()) : nullptr);

        registerSpecialUse({target}, {this, other});

        if (extraParams != nullptr)
            this->synchronize();
    }

////////////////////////////////////////////////////////////////////////
    void NDArray::syncToHost() const {
        
        if(isEmpty() || lengthOf() == 0) return;
        
        lazyAllocateBuffer();        
        auto stream = _context->getCudaStream();
        auto res = cudaStreamSynchronize(*stream);
        if (res != 0)
            throw cuda_exception::build("syncToHost failed to to some previous kernel failre", res);

        if (ews() != 1) {
            // FIXME: ^$%@#$%@#$!@#!!!!!!!!!!!
            for (Nd4jLong i = 0; i < _length; i++) {
                auto offset = getOffset(i) * sizeOfT();
                cudaMemcpy(_buffer + offset, _bufferD + offset, sizeOfT(), cudaMemcpyDeviceToHost);
            }
        }
        else
            cudaMemcpy(_buffer, _bufferD, _length * sizeOfT(), cudaMemcpyDeviceToHost);
        
        tickReadHost();
    }

////////////////////////////////////////////////////////////////////////
    void NDArray::syncToDevice() const {
        
        if(isEmpty()) return;

        if (_bufferD == nullptr) {
            NDArray* constThis =  const_cast<NDArray*>(this); // not recommended solution
            void* p = constThis->_bufferD;
            ALLOCATE_SPECIAL(p, _context->getWorkspace(), (getOffset(_length - 1) + 1) * sizeOfT(), int8_t);
            constThis->_isBuffDAlloc = true;
        }

         if (ews() != 1) {
            for (Nd4jLong i = 0; i < _length; i++) {
                auto offset = getOffset(i) * sizeOfT();
                cudaMemcpy(_bufferD + offset, _buffer + offset, sizeOfT(), cudaMemcpyHostToDevice);
            }
        }
        else
            cudaMemcpy(_bufferD, _buffer, _length * sizeOfT(), cudaMemcpyHostToDevice);
                
        tickReadDevice();        
    }

    void NDArray::syncShape() const {
        cudaMemcpy(_shapeInfoD, _shapeInfo, shape::shapeInfoByteLength(_shapeInfo), cudaMemcpyHostToDevice);
    }

    ////////////////////////////////////////////////////////////////////////
    template <typename X, typename Y>
    void NDArray::templatedDoubleAssign(void *xBuffer, const Nd4jLong xOffset, const void *yBuffer, const Nd4jLong yOffset) const {
        auto x = reinterpret_cast<X *>(xBuffer);
        const auto y = reinterpret_cast<const Y *>(yBuffer);
        if (x && y)
        *(reinterpret_cast<X*>(xBuffer) + xOffset) = static_cast<X>(*(reinterpret_cast<Y const*>(yBuffer) + yOffset));
    }
    BUILD_DOUBLE_TEMPLATE(template void NDArray::templatedDoubleAssign, (void *xBuffer, const Nd4jLong xOffset, const void *yBuffer, const Nd4jLong yOffset) const, LIBND4J_TYPES, LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
    // This method assigns values of given NDArray to this one
    void NDArray::assign(const NDArray& other) {

        if (this == &other)
            return;

        if (other.isEmpty()) {
            if (!isEmpty()) {
                 ArrayOptions::setPropertyBit(this->_shapeInfo, ARRAY_EMPTY);
                 syncShape();
            }
            return;
        }

        if(isEmpty()) {
            *this = other;
            return;
        }

        // if (!Environment::getInstance()->isExperimentalBuild() && (this->dataType() != other.dataType() && other.dataType() != DataType::BOOL)) {
        //     throw datatype_exception::build("NDArray::assign: cannot assign array of different types", this->dataType(), other.dataType());
        // }

        if (other.isScalar()) {
            if(this->isScalar()) {
                lazyAllocateBuffer();
                preparePrimaryUse({this}, {&other});
                BUILD_DOUBLE_SELECTOR(_dataType, other._dataType, templatedDoubleAssign, (_buffer, 0, other._buffer, 0), LIBND4J_TYPES, LIBND4J_TYPES);
                registerPrimaryUse({this}, {&other});
            }
            else {
                prepareSpecialUse({this}, {&other});
                NativeOpExecutioner::execScalar(_context, scalar::CopyPws, _buffer, _shapeInfo, _bufferD, _shapeInfoD, _buffer, _shapeInfo, _bufferD, _shapeInfoD, other._buffer, other._shapeInfo, other._bufferD, other._shapeInfoD, nullptr);
                registerSpecialUse({this}, {&other});
            }
            return;
        }

        if (other._length != _length) {
            auto shapeThis = ShapeUtils::shapeAsString(this);
            auto shapeThat = ShapeUtils::shapeAsString(&other);
            nd4j_printf("Can't assign new value to the array: this shape %s; other shape: %s\n", shapeThis.c_str(), shapeThat.c_str());
            throw std::runtime_error("Lengths of arrays are mismatched");
        }

        prepareSpecialUse({this}, {&other});

        // memcpy is allowed only for same order && same ews (being equal to 1)
        if (ordering() == other.ordering() && _dataType == other._dataType && ews() == 1 && other.ews() == 1) 
            cudaMemcpy(_bufferD, other._bufferD, _length * sizeOfT(), cudaMemcpyDeviceToDevice);
        else 
            NativeOpExecutioner::execTransformAny(_context, transform::Assign, nullptr, other._shapeInfo, other._bufferD, other._shapeInfoD, nullptr, _shapeInfo, _bufferD, _shapeInfoD, nullptr, nullptr, nullptr);

        registerSpecialUse({this}, {&other});
    }

    ////////////////////////////////////////////////////////////////////////
// This method returns new copy of this NDArray, optionally in different order
    NDArray* NDArray::dup(const char newOrder) {

        char order = newOrder == 'a' ? ordering() : newOrder;

        if (this->dataType() == DataType::UTF8) {
            std::vector<std::string> strings(_length);
            for (int e = 0; e < _length; e++)
                strings[e] = this->e<std::string>(e);

            auto result = NDArrayFactory::string_(order, this->getShapeAsVector(), strings, _context);
            return result;
        } else {
            auto result = new NDArray(order, getShapeAsVector(), _dataType, _context);//new NDArray(reinterpret_cast<Nd4jLong *>(outShapeInfo.primary()), true, _context);
            result->assign(*this);

            return result;
        }
    }

    void NDArray::synchronize() const {
        auto res = cudaStreamSynchronize(*(_context->getCudaStream()));
        if (res != 0)
            throw std::runtime_error("Synchronization failed");
    }


//////////////////////////////////////////////////////////////////////////
    template <>
    std::string NDArray::e(const Nd4jLong i) const {

        if (!isS())
            throw std::runtime_error("Can't get std::string out of non-string array");
        
        preparePrimaryUse({}, {this});

        // getting "virtual" offset. it's not real though,since it doesn't take lengths into account
        auto offset = getOffset(i);
        auto offsets = reinterpret_cast<Nd4jLong *>(_buffer);
        auto offsetsLength = ShapeUtils::stringBufferHeaderRequirements(this->lengthOf());
        auto start = offsets[offset];
        auto end = offsets[offset + 1];
        auto data = _buffer + offsetsLength + start;

        std::string r(reinterpret_cast<const char *>(data), (end - start));
        
        registerPrimaryUse({}, {this});
        
        return r;
    }

//////////////////////////////////////////////////////////////////////////
    template <typename T>
    T NDArray::e(const Nd4jLong i) const {

        if (i >= _length)
            throw std::invalid_argument("NDArray::e(i): input index is out of array length !");

        preparePrimaryUse({}, {this});
        
        auto rp = getOffset(i);
        registerPrimaryUse({}, {this});
        BUILD_SINGLE_PARTIAL_SELECTOR(this->dataType(), return templatedGet<, T>(this->_buffer, rp), LIBND4J_TYPES);
        
    }
    BUILD_SINGLE_UNCHAINED_TEMPLATE(template , NDArray::e(const Nd4jLong) const, LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
// Returns value from 2D matrix by coordinates/indexes
    template <typename T>
    T NDArray::e(const Nd4jLong i, const Nd4jLong j) const {
        if (rankOf() != 2 || i >= shapeOf()[0] || j >= shapeOf()[1])
            throw std::invalid_argument("NDArray::e(i,j): one of input indexes is out of array length or rank!=2 !");

        preparePrimaryUse({}, {this});

        auto xType = this->dataType();
        Nd4jLong coords[2] = {i, j};
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        registerPrimaryUse({}, {this});
        
        BUILD_SINGLE_PARTIAL_SELECTOR(xType, return templatedGet<, T>(this->_buffer, xOffset), LIBND4J_TYPES);
                
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

        preparePrimaryUse({}, {this});

        auto xType = this->dataType();
        Nd4jLong coords[3] = {i, j, k};
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        registerPrimaryUse({}, {this});

        BUILD_SINGLE_PARTIAL_SELECTOR(xType, return templatedGet<, T>(this->_buffer, xOffset), LIBND4J_TYPES);
                
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

        preparePrimaryUse({}, {this});

        auto xType = this->dataType();
        Nd4jLong coords[4] = {i, j, k, l};
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        registerPrimaryUse({}, {this});
        BUILD_SINGLE_PARTIAL_SELECTOR(xType, return templatedGet<, T>(this->_buffer, xOffset), LIBND4J_TYPES);
        
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

//////////////////////////////////////////////////////////////////////////
// perform array transformation
    void NDArray::applyTransform(nd4j::transform::FloatOps op, NDArray *target, ExtraArguments *extraParams) {

        if (isS())
            throw std::runtime_error("NDArray::applyTransform FloatOps: you can't use this method on String array!");

        if (target == nullptr)
            target = this;

        if (!target->isR())
            throw std::runtime_error("NDArray::applyTransform FloatOps: target array must have one of FLOAT types");

        prepareSpecialUse({target}, {this});
        NativeOpExecutioner::execTransformFloat(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, extraParams != nullptr ? extraParams->argumentsAsT(target->dataType()) : nullptr, nullptr, nullptr);
        registerSpecialUse({target}, {this});
    }

    ////////////////////////////////////////////////////////////////////////
    void NDArray::applyTransform(nd4j::transform::AnyOps op, NDArray *target, ExtraArguments *extraParams) {

        if (isS())
            throw std::runtime_error("NDArray::applyTransform AnyOps: you can't use this method on String array!");

        if (target == nullptr)
            target = this;

        prepareSpecialUse({target}, {this});
        NativeOpExecutioner::execTransformAny(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, extraParams != nullptr ? extraParams->argumentsAsT(target->dataType()) : nullptr, nullptr, nullptr);
        registerSpecialUse({target}, {this});
    }

    ////////////////////////////////////////////////////////////////////////
    void NDArray::applyTransform(nd4j::transform::SameOps op, NDArray *target, ExtraArguments *extraParams) {
        //nd4j_printf("Same op %i transform:\n", (int)op);
        if (isS())
            throw std::runtime_error("NDArray::applyTransform SameOps: you can't use this method on String array!");

        if (target == nullptr)
            target = this;

        if (target->dataType() != this->dataType())
            throw std::runtime_error("NDArray::applyTransform SameOps: target array must have the same data type as original array");
        
        prepareSpecialUse({target}, {this});
        NativeOpExecutioner::execTransformSame(_context, op, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, extraParams != nullptr ? extraParams->argumentsAsT(target->dataType()) : nullptr, nullptr, nullptr);
        registerSpecialUse({target}, {this});
    }

    ////////////////////////////////////////////////////////////////////////
    void NDArray::applyTransform(nd4j::transform::BoolOps op, NDArray *target, ExtraArguments *extraParams) {
        if (isS())
            throw std::runtime_error("NDArray::applyTransform BoolOps: you can't use this method on String array!");

        if (target == nullptr)
            target = this;

        if (!target->isB())
            throw std::runtime_error("NDArray::applyTransform BoolOps: target array must have one of BOOL types");

        prepareSpecialUse({target}, {this});        
        NativeOpExecutioner::execTransformBool(_context, op, this->_buffer, this->_shapeInfo, _bufferD, _shapeInfoD, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, extraParams != nullptr ? extraParams->argumentsAsT(this->dataType()) : nullptr, nullptr, nullptr);
        registerSpecialUse({target}, {this});
    }

    ////////////////////////////////////////////////////////////////////////
    void NDArray::applyTransform(nd4j::transform::StrictOps op, NDArray *target, ExtraArguments *extraParams) {
        if (isS())
            throw std::runtime_error("NDArray::applyTransform StrictOps: you can't use this method on String array!");

        if (target == nullptr)
            target = this;

        if (!this->isR() || !target->isR() || (this->dataType() != target->dataType()))
            throw std::runtime_error("NDArray::applyTransform StrictOps: both Source and Target array must have same FLOAT type !");

        registerSpecialUse({target}, {this});
        NativeOpExecutioner::execTransformStrict(_context, op, this->_buffer, this->_shapeInfo, _bufferD, _shapeInfoD, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, extraParams != nullptr ? extraParams->argumentsAsT(target->dataType()) : nullptr, nullptr, nullptr);
        prepareSpecialUse({target}, {this});
    }

    // perform array transformation
    ////////////////////////////////////////////////////////////////////////
    NDArray NDArray::transform(nd4j::transform::FloatOps op, void *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::transform FloatOps: you can't use this method on String array!");

        NDArray result(this->ordering(), getShapeAsVector(), DataTypeUtils::pickFloatingType(dataType()), this->_context);

        registerSpecialUse({&result}, {this});
        NativeOpExecutioner::execTransformFloat(_context, op, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, extraParams, nullptr, nullptr);
        prepareSpecialUse({&result}, {this});
        
        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    NDArray NDArray::transform(nd4j::transform::SameOps op, void *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::transform SameOps: you can't use this method on String array!");

        NDArray result(this->_shapeInfo, false, this->_context);

        prepareSpecialUse({&result}, {this});
        NativeOpExecutioner::execTransformSame(_context, op, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, extraParams, nullptr, nullptr);
        registerSpecialUse({&result}, {this});

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    NDArray NDArray::transform(nd4j::transform::StrictOps op, void *extraParams) const {
        if (!this->isR())
            throw std::runtime_error("Source array must have one of FLOAT types");

        NDArray result(this->_shapeInfo, false, this->_context);

        prepareSpecialUse({&result}, {this});
        NativeOpExecutioner::execTransformStrict(_context, op, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, extraParams, nullptr, nullptr);
        registerSpecialUse({&result}, {this});
        
        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    NDArray NDArray::transform(nd4j::transform::BoolOps op, void *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::transform BoolOps: you can't use this method on String array!");

        NDArray result(this->ordering(), getShapeAsVector(), nd4j::DataType::BOOL, this->_context);

        prepareSpecialUse({&result}, {this});
        NativeOpExecutioner::execTransformBool(_context, op, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, extraParams, nullptr, nullptr);
        registerSpecialUse({&result}, {this});

        return result;
    }

//////////////////////////////////////////////////////////////////////////
// perform pairwise transformation
    void NDArray::applyPairwiseTransform(nd4j::pairwise::Ops op, const NDArray& other, ExtraArguments *extraParams) {
        applyPairwiseTransform(op, &other, this, extraParams);
    }

    ////////////////////////////////////////////////////////////////////////
    void NDArray::applyPairwiseTransform(nd4j::pairwise::BoolOps op, const NDArray *other, NDArray *target, ExtraArguments *extraParams) const{
        if (isS())
            throw std::runtime_error("NDArray::applyPairwiseTransform BoolOps: you can't use this method on String array!");
        if (other->lengthOf() != target->lengthOf())
            throw std::invalid_argument("NDArray::applyPairwiseTransform BoolOps method - lengths of arrays are mismatched");
        if (!target->isB())
            throw std::invalid_argument("NDArray::applyPairwiseTransform BoolOps method - result must have bool type");
        if (_dataType != other->_dataType)
            throw std::invalid_argument("NDArray::applyPairwiseTransform BoolOps method - this and other arrays must have the same type !");

        prepareSpecialUse({target}, {this, other});
        NativeOpExecutioner::execPairwiseBoolTransform(_context, op, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, other->_buffer, other->_shapeInfo, other->_bufferD, other->_shapeInfoD, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, extraParams != nullptr ? extraParams->argumentsAsT(target->dataType()) : nullptr);
        registerSpecialUse({target}, {this, other});
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
        
        prepareSpecialUse({target}, {this, scalar});        
        NativeOpExecutioner::execScalarBool(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, scalar->_buffer, scalar->_shapeInfo, scalar->_bufferD, scalar->_shapeInfoD, extraParams != nullptr ? extraParams->argumentsAsT(target->dataType()): nullptr);
        registerSpecialUse({target}, {this, scalar});
    }
    
    ////////////////////////////////////////////////////////////////////////
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
        
        prepareSpecialUse({target}, {this, scalar});
        NativeOpExecutioner::execScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, scalar->getBuffer(), scalar->getShapeInfo(), scalar->_bufferD, scalar->_shapeInfoD, extraParams != nullptr ? extraParams->argumentsAsT(target->dataType()) : nullptr);
        registerSpecialUse({target}, {this, scalar});
    }

    ////////////////////////////////////////////////////////////////////////
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
        auto result = (NDArray*)this;// == nullptr ? this : target;
        if (target != nullptr)
            result = target;

        if(result->_dataType != DataTypeUtils::pickPairwiseResultType(_shapeInfo, tadArray->_shapeInfo))
            throw std::invalid_argument("NDArray::applyBroadcast method: wrong type of target array !");
        if(!result->isSameShape(this))
            throw std::invalid_argument("NDArray::applyBroadcast method: this and target arrays must have the same shape !");

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, dimensions);
        auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(result->_shapeInfo, dimensions);

        auto tadLength = shape::length(packX.primaryShapeInfo());

        if (tadLength != tadArray->lengthOf())
            throw std::runtime_error("NDArray::applyBroadcast method: tad length mismatch !");

        NDArray::prepareSpecialUse({result}, {this, tadArray});

        NativeOpExecutioner::execBroadcast(_context, op, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD, tadArray->_buffer, tadArray->_shapeInfo, tadArray->_bufferD, tadArray->_shapeInfoD, result->_buffer, result->_shapeInfo, result->_bufferD, result->_shapeInfoD,
                                           const_cast<int *>(dimensions.data()), (int)dimensions.size(), packX.specialShapeInfo(), packX.specialOffsets(), packZ.specialShapeInfo(), packZ.specialOffsets());

        NDArray::registerSpecialUse({result}, {this, tadArray});
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

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, dimensions);
        auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(result->_shapeInfo, dimensions);

        auto tadLength = shape::length(packX.primaryShapeInfo());
        if (tadLength != tadArray->lengthOf())
            throw std::runtime_error("Tad length mismatch");

        NDArray::prepareSpecialUse({result}, {this, tadArray});

        NativeOpExecutioner::execBroadcastBool(_context, op, this->_buffer, this->_shapeInfo, this->_bufferD, this->_shapeInfoD,
                                               tadArray->_buffer, tadArray->_shapeInfo, tadArray->_bufferD, tadArray->_shapeInfoD,
                                               result->_buffer, result->_shapeInfo, result->_bufferD, result->_shapeInfoD,
                                               nullptr, (int)dimensions.size(), packX.specialShapeInfo(), packX.specialOffsets(), packZ.specialShapeInfo(), packZ.specialOffsets());

        NDArray::registerSpecialUse({result}, {this, tadArray});
    }

    //////////////////////////////////////////////////////////////////////////
    NDArray NDArray::applyTrueBroadcast(nd4j::BroadcastOpsTuple op, const NDArray& other, ExtraArguments *extraArgs) const {
        Nd4jLong* newShapeInfo = nullptr;
        if(!ShapeUtils::evalBroadcastShapeInfo(*this, other, true, newShapeInfo, _context->getWorkspace()))          // the rank of new array = max->rankOf)()
            throw std::runtime_error("NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for broadcast operation !");
        NDArray result(newShapeInfo, true, this->_context);

        this->applyTrueBroadcast(op, &other, &result, false, extraArgs);

        return result;
    }
  
    ////////////////////////////////////////////////////////////////////////
    void NDArray::applyIndexReduce(nd4j::indexreduce::Ops op, NDArray* target, const std::vector<int>& dimensions, const ExtraArguments *extraParams) const {
        if (isS())
            throw std::runtime_error("NDArray::applyIndexReduce: you can't use this method on String array!");

        if (target->dataType() != nd4j::DataType::INT64)
            throw std::runtime_error("NDArray::applyIndexReduce operations return INT64");
        
        void* params = extraParams != nullptr ? const_cast<ExtraArguments*>(extraParams)->argumentsAsT(this->dataType()) : nullptr;

        NDArray::prepareSpecialUse({target}, {this});

        if (target->isScalar()) {
            //target->_buffer[0] = functions::indexreduce::IndexReduce<T>::template execScalar<OpName>(_buffer, _shapeInfo, const_cast<T*>(extraParams));
            NativeOpExecutioner::execIndexReduceScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, params, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD);
        }
        else {

            std::vector<int> copy(dimensions);
            shape::checkDimensions(rankOf(), copy);

            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, copy);

            NativeOpExecutioner::execIndexReduce(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, params,target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, nullptr, copy.size(), packX.specialShapeInfo(), packX.specialOffsets());
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

        auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, DataType::INT64, false, false, _context->getWorkspace());
        auto result = new NDArray(newShape, true, _context);

        void* params = extraParams != nullptr ? const_cast<ExtraArguments*>(extraParams)->argumentsAsT(this->dataType()) : nullptr;

        NDArray::prepareSpecialUse({result}, {this});

        if (rankOf() == copy.size()) {
            NativeOpExecutioner::execIndexReduceScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, params, result->_buffer, result->_shapeInfo, result->_bufferD, result->_shapeInfoD);

            auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
            if (cudaResult != 0)
                throw cuda_exception::build("NDArray::applyIndexReduce cuda failed !", cudaResult);
        }
        else {
            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, copy);

            NativeOpExecutioner::execIndexReduce(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD,
                                                params,
                                                result->_buffer, result->_shapeInfo, result->_bufferD, result->_shapeInfoD,
                                                nullptr, copy.size(),
                                                packX.specialShapeInfo(), packX.specialOffsets());

            auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
            if (cudaResult != 0) throw cuda_exception::build("NDArray::applyIndexReduce cuda failed !", cudaResult);
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
        auto result = new NDArray(newShape, true, _context);
        // create dynamic array of extra parameters if array extraParams is empty (==nullptr)
        void* params = extraParams != nullptr ? const_cast<ExtraArguments*>(extraParams)->argumentsAsT(this->dataType()) : nullptr;        

        NDArray::prepareSpecialUse({result}, {this, other});

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

        auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, DataTypeUtils::pickFloatingType(_dataType), false, false, _context->getWorkspace());
        auto result = new NDArray(newShape, true, _context);
        // create temporary dynamic array of extra parameters if array extraParams is empty (==nullptr)
        void* params = extraParams != nullptr ? const_cast<ExtraArguments*>(extraParams)->argumentsAsT(this->dataType()) : nullptr;

        NDArray::prepareSpecialUse({result}, {this, other});

        // perform calculations
        if(rankOf() == copy.size() && other->rankOf() == copy.size()) {
            NativeOpExecutioner::execReduce3Scalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, params, other->_buffer, other->_shapeInfo, other->_bufferD, other->_shapeInfoD, result->_buffer, result->shapeInfo(), result->specialBuffer(), result->specialShapeInfo());

            auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
            if (cudaResult != 0)
                throw cuda_exception::build("NDArray::applyReduce3 cuda failed !", cudaResult);
        }
        else {

            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(_shapeInfo, copy);
            auto packY = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(other->getShapeInfo(), copy);

            if(!shape::equalsSoft(packX.primaryShapeInfo(), packY.primaryShapeInfo()) || (packX.numberOfTads() != packY.numberOfTads() && packX.numberOfTads() != 1 && packY.numberOfTads() != 1))
                throw std::runtime_error("NDArray::applyReduce3 cuda method: arrays tads are inconsistent !");

            NativeOpExecutioner::execReduce3(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, params, other->_buffer, other->_shapeInfo, other->_bufferD, other->_shapeInfoD, result->_buffer, result->_shapeInfo, result->_bufferD, result->_shapeInfoD, nullptr, copy.size(), packX.specialShapeInfo(), packX.specialOffsets(),  packY.specialShapeInfo(), packY.specialOffsets());

            auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
            if (cudaResult != 0)
                throw cuda_exception::build("NDArray::applyReduce3 cuda failed !", cudaResult);
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

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(getShapeInfo(), copy);
        auto packY = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(other->getShapeInfo(), copy);

        // check tads shapes
        if(!shape::equalsSoft(packX.primaryShapeInfo(), packY.primaryShapeInfo()))
            throw std::runtime_error("NDArray::applyAllReduce3 method: the shapes of array tads are different !");

        // set newShape for output array
        Nd4jLong *newShape = nullptr;
        ALLOCATE(newShape, _context->getWorkspace(), 8, Nd4jLong);
        newShape[0] = 2;        // output rank is always equal to 2 for execAll case
        newShape[1] = packX.numberOfTads();
        newShape[2] = packY.numberOfTads();
        ShapeUtils::updateStridesAndType(newShape, DataTypeUtils::pickFloatingType(_dataType), 'c');
        // create output array
        auto result = new NDArray(newShape, true, _context);
        RELEASE(newShape, _context->getWorkspace());

        NDArray::prepareSpecialUse({result}, {const_cast<NDArray*>(this), const_cast<NDArray*>(other)});

        // create dynamic array of extra parameters if array extraParams is empty (==nullptr)
        void* params = extraParams != nullptr ? const_cast<ExtraArguments*>(extraParams)->argumentsAsT(this->dataType()) : nullptr;

        NativeOpExecutioner::execReduce3All(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, params,other->_buffer, other->_shapeInfo, other->_bufferD, other->_shapeInfoD,result->_buffer,result->_shapeInfo, result->_bufferD, result->_shapeInfoD,
                                            nullptr, copy.size(), packX.specialShapeInfo(), packX.specialOffsets(), packY.specialShapeInfo(), packY.specialOffsets());

        auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
        if (cudaResult != 0) throw cuda_exception::build("NDArray::applyAllReduce3 cuda failed !", cudaResult);

        NDArray::registerSpecialUse({result}, {this, other});
        
        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    void NDArray::prepareSpecialUse(const std::initializer_list<const NDArray*>& writeList, const std::initializer_list<const NDArray*>& readList, bool synchronizeWritables) {

        for (const auto& a : readList) {
            if (!a->isActualOnDeviceSide())
                a->syncToDevice();
            a->tickReadDevice();
        }

        for (const auto& a : writeList) {
            if (synchronizeWritables && !a->isActualOnDeviceSide())
                a->syncToDevice();
            a->tickWriteDevice();
        }
    }

    ////////////////////////////////////////////////////////////////////////
    void NDArray::registerSpecialUse(const std::initializer_list<const NDArray*>& writeList, const std::initializer_list<const NDArray*>& readList) {                

		for (const auto& p : readList) 
			p->tickReadDevice();

        for (const auto& p : writeList)
            p->tickWriteDevice();
    }

////////////////////////////////////////////////////////////////////////
void NDArray::preparePrimaryUse(const std::initializer_list<const NDArray*>& writeList, const std::initializer_list<const NDArray*>& readList, bool synchronizeWritables) {
        
    for (const auto& a : writeList) {    
        if (synchronizeWritables && !a->isActualOnHostSide())
            a->syncToHost();
        a->tickWriteHost();
    }

    for (const auto& a : readList) {
        if (!a->isActualOnHostSide())
            a->syncToHost();            
        a->tickReadHost();
    }
}

////////////////////////////////////////////////////////////////////////
void NDArray::registerPrimaryUse(const std::initializer_list<const NDArray*>& writeList, const std::initializer_list<const NDArray*>& readList) {
        
    for (const auto& p : writeList)         
        p->tickWriteHost();        

    for (const auto& p : readList) 
        p->tickReadHost();        
}
        
////////////////////////////////////////////////////////////////////////
// default destructor
NDArray::~NDArray() noexcept {
    if (isS()) {
        delete[] _buffer;
    }
    else
    if (_isBuffAlloc)
        RELEASE(_buffer, _context->getWorkspace());

    if (_isBuffDAlloc) {
        RELEASE_SPECIAL(_bufferD, _context->getWorkspace());
    }
}


////////////////////////////////////////////////////////////////////////
    NDArray* NDArray::varianceAlongDimension(nd4j::variance::Ops op, const bool biasCorrected, const std::vector<int>& dimensions) const {
        if (isS())
            throw std::runtime_error("NDArray::varianceAlongDimension: you can't use this method on String array!");

        std::vector<int> copy(dimensions);
        if (copy.size() > 1)
            std::sort(copy.begin(), copy.end());

        auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, DataTypeUtils::pickFloatingType(_dataType), false, false, _context->getWorkspace());
        auto result = new NDArray(newShape, true, _context);

        NDArray::prepareSpecialUse({result}, {this});

        if(rankOf() == copy.size() || copy.empty())
            NativeOpExecutioner::execSummaryStatsScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, result->buffer(), result->shapeInfo(), result->specialBuffer(), result->specialShapeInfo(), biasCorrected);
        else {
            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), copy);

            NativeOpExecutioner::execSummaryStats(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, result->_buffer, result->_shapeInfo, result->_bufferD, result->_shapeInfoD,
                                                  nullptr, copy.size(), packX.specialShapeInfo(), packX.specialOffsets(), biasCorrected);

            auto res = cudaStreamSynchronize(*_context->getCudaStream());
            if (res != 0)
                throw cuda_exception::build("varianceAlongDimension failed", res);
        }

        NDArray::registerSpecialUse({result}, {this});

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    NDArray NDArray::varianceAlongDims(nd4j::variance::Ops op, const bool biasCorrected, const std::vector<int>& dimensions) const {
        if (isS())
            throw std::runtime_error("NDArray::varianceAlongDimension: you can't use this method on String array!");

        std::vector<int> copy(dimensions);
        if (copy.size() > 1)
            std::sort(copy.begin(), copy.end());

        auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, DataTypeUtils::pickFloatingType(_dataType), false, false, _context->getWorkspace());
        NDArray result(newShape, true, _context);

        NDArray::prepareSpecialUse({&result}, {this});

        if(rankOf() == copy.size() || copy.empty())
            NativeOpExecutioner::execSummaryStatsScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, result.getBuffer(), result.getShapeInfo(), result.getSpecialBuffer(), result.getSpecialShapeInfo(), biasCorrected);
        else {
            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), copy);

            NativeOpExecutioner::execSummaryStats(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, result._buffer, result._shapeInfo, result._bufferD, result._shapeInfoD, nullptr, copy.size(), packX.specialShapeInfo(), packX.specialOffsets(), biasCorrected);

            auto res = cudaStreamSynchronize(*_context->getCudaStream());
            if (res != 0)
                throw cuda_exception::build("varianceAlongDimension failed", res);
        }

        NDArray::registerSpecialUse({&result}, {this});

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    void NDArray::varianceAlongDimension(nd4j::variance::Ops op, const NDArray *target, const bool biasCorrected, const std::vector<int>& dimensions) {
        if (isS())
            throw std::runtime_error("NDArray::varianceAlongDimension: you can't use this method on String array!");

        std::vector<int> copy(dimensions);

        if (!target->isR())
            throw std::runtime_error("NDArray::varianceAlongDimension: target array must have FLOAT type");

        NDArray::prepareSpecialUse({target}, {this});

        if(rankOf() == copy.size() || copy.empty())
            NativeOpExecutioner::execSummaryStatsScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, target->getBuffer(), target->getShapeInfo(), target->getSpecialBuffer(), target->getSpecialShapeInfo(), biasCorrected);
        else {
            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), copy);

            NativeOpExecutioner::execSummaryStats(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, nullptr, copy.size(), packX.specialShapeInfo(), packX.specialOffsets(), biasCorrected);

            auto res = cudaStreamSynchronize(*_context->getCudaStream());
            if (res != 0)
                throw cuda_exception::build("varianceAlongDimension failed", res);
        }

        NDArray::registerSpecialUse({target}, {this});
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

        NDArray tmp = NDArrayFactory::create<float>(0.f, _context); // scalar = 0
        NDArray::prepareSpecialUse({&tmp}, {this, other});

        ExtraArguments extras({eps});
        NativeOpExecutioner::execReduce3Scalar(_context, reduce3::EqualsWithEps, _buffer, _shapeInfo, _bufferD, _shapeInfoD, extras.argumentsAsT(DataType::FLOAT32), other->_buffer, other->_shapeInfo, other->_bufferD, other->_shapeInfoD, tmp._buffer, tmp._shapeInfo, tmp._bufferD, tmp._shapeInfoD);

        NDArray::registerSpecialUse({&tmp}, {this, other});

        auto res = cudaStreamSynchronize(*_context->getCudaStream());
        if (res != 0)
            throw cuda_exception::build("NDArray::equalsTo failed", res);

        auto r = tmp.e<int>(0);
        //nd4j_printf("equalsTo result: [%lld]\n", r);
        if (r > 0LL)
            return false;

        return true;
    }

    //////////////////////////////////////////////////////////////////////////
    bool NDArray::permutei(const int* dimensions, const int rank) {

        // check if current object is _shapeInfo owner
        auto shapeInfo = ShapeUtils::evalPermShapeInfo(dimensions, rank, *this, _context->getWorkspace());

        ShapeDescriptor descriptor(shapeInfo, dataType());
        setShapeInfo(descriptor);

        return true;
    }

    //////////////////////////////////////////////////////////////////////////
    bool NDArray::permutei(const Nd4jLong* dimensions, const int rank) {

        // check if current object is _shapeInfo owner

        auto shapeInfo = ShapeUtils::evalPermShapeInfo(dimensions, rank, *this, _context->getWorkspace());
        ShapeDescriptor descriptor(shapeInfo, dataType());

        return true;
    }

//////////////////////////////////////////////////////////////////////////
// method reduces array by excluding its shapes along axes present in dimensions vector
void NDArray::reduceAlongDimension(nd4j::reduce::FloatOps op, NDArray* target, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes, const bool checkTargetShape) const {

    if (isS())
        throw std::runtime_error("NDArray::reduceAlongDimension FloatOps cuda: you can't use this method on String array!");
    if (target == nullptr || !target->isR())
        throw std::invalid_argument("NDArray::reduceAlongDimension FloatOps cuda: requires target array to be present and have type form real space!");

    std::vector<int> copy(dimensions);
    if (copy.size())
    shape::checkDimensions(rankOf(), copy);

    if(checkTargetShape) {
        auto newShape = ShapeUtils::evalReduceShapeInfo(target->ordering(), copy, *this, keepDims, supportOldShapes, _context->getWorkspace());
        if(!shape::shapeEquals(newShape, target->getShapeInfo()))
            throw std::runtime_error("NDArray::reduceAlongDimension FloatOps cuda: wrong target shape!");
    }

    NDArray::prepareSpecialUse({target}, {this});

    if(rankOf() == copy.size() || copy.empty()) {        
        NativeOpExecutioner::execReduceFloatScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD,nullptr, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD);        

        auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
        if (cudaResult != 0)
            throw cuda_exception::build("NDArray::reduceAlongDimension FloatOps cuda failed !", cudaResult);
    }
    else {
        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), copy);

        NativeOpExecutioner::execReduceFloat(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, nullptr, copy.size(), packX.specialShapeInfo(), packX.specialOffsets());

        auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
        if (cudaResult != 0)
            throw cuda_exception::build("NDArray::reduceAlongDimension FloatOps cuda failed !", cudaResult);
    }

    NDArray::registerSpecialUse({target}, {this});
}

//////////////////////////////////////////////////////////////////////////
// method reduces array by excluding its shapes along axes present in dimensions vector
void NDArray::reduceAlongDimension(nd4j::reduce::SameOps op, NDArray* target, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes, const bool checkTargetShape) const {

    if (isS())
        throw std::runtime_error("NDArray::reduceAlongDimension SameOps cuda: you can't use this method on String array!");
    if (target == nullptr || target->_dataType != _dataType)
        throw std::runtime_error("NDArray::reduceAlongDimension SameOps cuda: requires target array to be present and have same dtype as input");

    std::vector<int> copy(dimensions);
    if (copy.size())
    shape::checkDimensions(rankOf(), copy);

    if(checkTargetShape) {
        auto newShape = ShapeUtils::evalReduceShapeInfo(target->ordering(), copy, *this, keepDims, supportOldShapes, _context->getWorkspace());
        if(!shape::shapeEquals(newShape, target->getShapeInfo()))
            throw std::runtime_error("NDArray::reduceAlongDimension SameOps cuda: wrong target shape!");
    }

    NDArray::prepareSpecialUse({target}, {this});

    if(rankOf() == copy.size() || copy.empty()) {        
        NativeOpExecutioner::execReduceSameScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD);        

        auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
        if (cudaResult != 0) throw cuda_exception::build("NDArray::reduceAlongDimension SameOps cuda failed !", cudaResult);
    }
    else {
        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), copy);

        NativeOpExecutioner::execReduceSame(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, nullptr, copy.size(), packX.specialShapeInfo(), packX.specialOffsets());

        auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
        if (cudaResult != 0)
            throw cuda_exception::build("NDArray::reduceAlongDimension SameOps cuda failed !", cudaResult);
    }
    NDArray::registerSpecialUse({target}, {this});
}

//////////////////////////////////////////////////////////////////////////
// method reduces array by excluding its shapes along axes present in dimensions vector
void NDArray::reduceAlongDimension(nd4j::reduce::BoolOps op, NDArray* target, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes, const bool checkTargetShape) const {

    if (isS())
        throw std::runtime_error("NDArray::reduceAlongDimension BoolOps cuda: you can't use this method on String array!");
    if (target == nullptr || !target->isB())
        throw std::invalid_argument("NDArray::reduceAlongDimension BoolOps cuda: requires target array to be present and have BOOL type!");    

    std::vector<int> copy(dimensions);
    shape::checkDimensions(rankOf(), copy);

    if(checkTargetShape) {
        auto newShape = ShapeUtils::evalReduceShapeInfo(target->ordering(), copy, *this, keepDims, supportOldShapes, _context->getWorkspace());
        if(!shape::shapeEquals(newShape, target->getShapeInfo()))
            throw std::runtime_error("NDArray::reduceAlongDimension BoolOps cuda: wrong target shape!");
    }

    NDArray::prepareSpecialUse({target}, {this});

    if(rankOf() == copy.size() || copy.empty()) {        
        NativeOpExecutioner::execReduceBoolScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD);        

        auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
        if (cudaResult != 0) throw cuda_exception::build("NDArray::reduceAlongDimension BoolOps cuda failed !", cudaResult);
    }
    else {
        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), copy);

        NativeOpExecutioner::execReduceBool(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, nullptr, copy.size(), packX.specialShapeInfo(), packX.specialOffsets());

        auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
        if (cudaResult != 0)
            throw cuda_exception::build("NDArray::reduceAlongDimension BoolOps cuda failed !", cudaResult);
    }
    NDArray::registerSpecialUse({target}, {this});
}

//////////////////////////////////////////////////////////////////////////
// method reduces array by excluding its shapes along axes present in dimensions vector
void NDArray::reduceAlongDimension(nd4j::reduce::LongOps op, NDArray* target, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes, const bool checkTargetShape) const {

    if (isS())
        throw std::runtime_error("NDArray::reduceAlongDimension LongOps cuda: you can't use this method on String array!");
    if (target == nullptr || target->_dataType != DataType::INT64)
        throw std::runtime_error("NDArray::reduceAlongDimension LongOps cuda: requires target array to be present and have type of INT64");

    std::vector<int> copy(dimensions);
    shape::checkDimensions(rankOf(), copy);

    if(checkTargetShape) {
        auto newShape = ShapeUtils::evalReduceShapeInfo(target->ordering(), copy, *this, keepDims, supportOldShapes, _context->getWorkspace());
        if(!shape::shapeEquals(newShape, target->getShapeInfo()))
            throw std::runtime_error("NDArray::reduceAlongDimension LongOps cuda: wrong target shape!");
    }

    NDArray::prepareSpecialUse({target}, {this});

    if(rankOf() == copy.size() || copy.empty()) {        
        NativeOpExecutioner::execReduceLongScalar(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD);        

        auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
        if (cudaResult != 0) throw cuda_exception::build("NDArray::reduceAlongDimension LongOps cuda failed !", cudaResult);
    }
    else {
        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), copy);

        NativeOpExecutioner::execReduceLong(_context, op, _buffer, _shapeInfo, _bufferD, _shapeInfoD, nullptr, target->_buffer, target->_shapeInfo, target->_bufferD, target->_shapeInfoD, nullptr, copy.size(), packX.specialShapeInfo(), packX.specialOffsets());

        auto cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
        if (cudaResult != 0)
            throw cuda_exception::build("NDArray::reduceAlongDimension LongOps cuda failed !", cudaResult);
    }
    NDArray::registerSpecialUse({target}, {this});
}

//////////////////////////////////////////////////////////////////////////
// This method sets value in linear buffer to position i
    template <typename T>
    void NDArray::p(const Nd4jLong i, const T value) {

        preparePrimaryUse({this}, {}, true);

        if (i >= _length)
            throw std::invalid_argument("NDArray::p(i, value): input index is out of array length !");

        auto rp = getOffset(i);
        const void *pV = reinterpret_cast<const void*>(const_cast<T *>(&value));        
    
        BUILD_SINGLE_PARTIAL_SELECTOR(this->dataType(), templatedSet<, T>(this->_buffer, rp, pV), LIBND4J_TYPES);

        registerPrimaryUse({this}, {});
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

    ////////////////////////////////////////////////////////////////////////
    void NDArray::p(const Nd4jLong i, const NDArray& scalar) {
        
        if(!scalar.isScalar())
            throw std::invalid_argument("NDArray::p method: input array must be scalar!");
        if (i >= _length)
            throw std::invalid_argument("NDArray::p(i, NDArray_scalar): input index is out of array length !");

        preparePrimaryUse({this}, {&scalar}, true);
        auto rp = getOffset(i);        
        BUILD_SINGLE_SELECTOR(scalar.dataType(), templatedSet, (_buffer, rp, scalar.dataType(), scalar.getBuffer()), LIBND4J_TYPES);
        registerPrimaryUse({this}, {&scalar});
    }

//////////////////////////////////////////////////////////////////////////
// This method sets value in 2D matrix to position i, j

    template <typename T>
    void NDArray::p(const Nd4jLong i, const Nd4jLong j, const T value) {
        //(*this)(i,j) = value;
        if (rankOf() != 2 || i >= shapeOf()[0] || j >= shapeOf()[1])
            throw std::invalid_argument("NDArray:pe(i,j, value): one of input indexes is out of array length or rank!=2 !");

        preparePrimaryUse({this}, {}, true);

        void *p = reinterpret_cast<void *>(const_cast<T *>(&value));        
        Nd4jLong coords[2] = {i, j};
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        BUILD_SINGLE_PARTIAL_SELECTOR(dataType(), templatedSet<, T>(this->_buffer, xOffset, p), LIBND4J_TYPES);
        registerPrimaryUse({this}, {});
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

//////////////////////////////////////////////////////////////////////////
// This method sets value in 3D matrix to position i,j,k
    template <typename T>
    void NDArray::p(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const T value) {
        //(*this)(i,j,k) = value;
        if (rankOf() != 3 || i >= shapeOf()[0] || j >= shapeOf()[1] || k >= shapeOf()[2])
            throw std::invalid_argument("NDArray:pe(i,j,k, value): one of input indexes is out of array length or rank!=3 !");

        preparePrimaryUse({this}, {}, true);
        
        void *p = reinterpret_cast<void *>(const_cast<T *>(&value));        
        Nd4jLong coords[3] = {i, j, k};        
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        BUILD_SINGLE_PARTIAL_SELECTOR(dataType(), templatedSet<, T>(this->_buffer, xOffset, p), LIBND4J_TYPES);
        registerPrimaryUse({this}, {});
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

        preparePrimaryUse({this}, {}, true);
        
        void *p = reinterpret_cast<void *>(const_cast<T *>(&value));        
        Nd4jLong coords[4] = {i, j, k, l};                
        auto xOffset = shape::getOffset(0, shapeOf(), stridesOf(), coords, rankOf());
        BUILD_SINGLE_PARTIAL_SELECTOR(dataType(), templatedSet<, T>(this->_buffer, xOffset, p), LIBND4J_TYPES);
        registerPrimaryUse({this}, {});
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
        return _bufferD != nullptr ? _bufferD + (offset * sizeOfT()) : nullptr;
    }

//////////////////////////////////////////////////////////////////////////    
    NDArray* NDArray::tensorAlongDimension(Nd4jLong index, const std::vector<int>& dimensions) const {
        std::vector<int> copy(dimensions);
        shape::checkDimensions(rankOf(), copy);

        Nd4jLong tadLength = shape::tadLength(this->_shapeInfo, copy.data(), copy.size());
        Nd4jLong numTads = this->lengthOf() / tadLength;

        if (index >= numTads)
            throw std::runtime_error("Can't get index higher than total number of TADs");

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), copy);

        auto array = new NDArray(bufferWithOffset(packX.primaryOffsets()[index]), specialBufferWithOffset(packX.primaryOffsets()[index]), packX.primaryShapeInfo(), _context);        
        array->_isView = true;
        array->copyBufferStatus(*this);

        return array;
    }

//////////////////////////////////////////////////////////////////////////
    void NDArray::addRowVector(const NDArray *row, NDArray *target) const {

        if (isS())
            throw std::runtime_error("NDArray::addRowVector: you can't use this method on String array!");
        if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() || !row->isRowVector() || columns() != row->lengthOf())
            throw std::invalid_argument("NDArray::addRowVector: wrong arguments !");
        if(target->_dataType !=  DataTypeUtils::pickPairwiseResultType(_dataType, row->_dataType) && !(isR() && row->isR() && target->isR()))
            throw std::invalid_argument("NDArray::addRowVector: wrong type of target array !");

        int dimension = 1;

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), dimension);

        NDArray::prepareSpecialUse({target}, {this, row});
        NativeOpExecutioner::execBroadcast(_context, nd4j::broadcast::Ops::Add, _buffer, _shapeInfo, _bufferD, _shapeInfoD, row->_buffer, row->_shapeInfo, row->_bufferD, row->_shapeInfoD, target->getBuffer(), target->getShapeInfo(), target->getSpecialBuffer(), target->getSpecialShapeInfo(), nullptr, 1, packX.specialShapeInfo(), packX.specialOffsets(), nullptr, nullptr);
        NDArray::registerSpecialUse({target}, {this, row});
    }

//////////////////////////////////////////////////////////////////////////
    void NDArray::subRowVector(const NDArray *row, NDArray * target) const {

        if (isS())
            throw std::runtime_error("NDArray::subRowVector: you can't use this method on String array!");
        if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() || !row->isRowVector() || columns() != row->columns())
            throw std::invalid_argument("NDArray::subRowVector: wrong arguments !");
        if(target->_dataType !=  DataTypeUtils::pickPairwiseResultType(_dataType, row->_dataType))
            throw std::invalid_argument("NDArray::subRowVector: wrong type of target array !");

        int dimension = 1;

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), dimension);

        NDArray::prepareSpecialUse({target}, {this, row});
        NativeOpExecutioner::execBroadcast(_context, nd4j::broadcast::Ops::Subtract, _buffer, _shapeInfo, _bufferD, _shapeInfoD, row->_buffer, row->_shapeInfo, row->_bufferD, row->_shapeInfoD, target->getBuffer(), target->getShapeInfo(), target->getSpecialBuffer(), target->getSpecialShapeInfo(), nullptr, 1, packX.specialShapeInfo(), packX.specialOffsets(), nullptr, nullptr);
        NDArray::registerSpecialUse({target}, {this, row});
    }

//////////////////////////////////////////////////////////////////////////
    void NDArray::mulRowVector(const NDArray *row, NDArray *target) const {

        if (isS())
            throw std::runtime_error("NDArray::mulRowVector: you can't use this method on String array!");
        if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() || !row->isRowVector() || columns() != row->columns())
            throw std::invalid_argument("NDArray::divRowVector: wrong arguments !");
        if(target->_dataType !=  DataTypeUtils::pickPairwiseResultType(_dataType, row->_dataType))
            throw std::invalid_argument("NDArray::mulRowVector: wrong type of target array !");

        int dimension = 1;

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), dimension);

        NDArray::prepareSpecialUse({target}, {this, row});
        NativeOpExecutioner::execBroadcast(_context, nd4j::broadcast::Ops::Multiply, _buffer, _shapeInfo, _bufferD, _shapeInfoD, row->_buffer, row->_shapeInfo, row->_bufferD, row->_shapeInfoD, target->getBuffer(), target->getShapeInfo(), target->getSpecialBuffer(), target->getSpecialShapeInfo(), nullptr, 1, packX.specialShapeInfo(), packX.specialOffsets(), nullptr, nullptr);
        NDArray::registerSpecialUse({target}, {this, row});
    }

//////////////////////////////////////////////////////////////////////////
    void NDArray::divRowVector(const NDArray *row, NDArray *target) const {

        if (isS())
            throw std::runtime_error("NDArray::divRowVector: you can't use this method on String array!");
        if (row->isB())
            throw std::runtime_error("NDArray::divRowVector: you can't divide by bool row!");
        if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() || !row->isRowVector() || columns() != row->columns())
            throw std::invalid_argument("NDArray::divRowVector: wrong arguments !");
        if(target->_dataType !=  DataTypeUtils::pickPairwiseResultType(_dataType, row->_dataType))
            throw std::invalid_argument("NDArray::divRowVector: wrong type of target array !");

        int dimension = 1;

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), dimension);

        NDArray::prepareSpecialUse({target}, {this, row});
        NativeOpExecutioner::execBroadcast(_context, nd4j::broadcast::Divide, _buffer, _shapeInfo, _bufferD, _shapeInfoD, row->_buffer, row->_shapeInfo, row->_bufferD, row->_shapeInfoD, target->getBuffer(), target->getShapeInfo(), target->getSpecialBuffer(), target->getSpecialShapeInfo(), nullptr, 1, packX.specialShapeInfo(), packX.specialOffsets(), nullptr, nullptr);
        NDArray::registerSpecialUse({target}, {this, row});
    }

//////////////////////////////////////////////////////////////////////////
// This method adds given row to all rows in this NDArray, this array becomes affected
    void NDArray::addiRowVector(const NDArray *row) {

        if (isS())
            throw std::runtime_error("NDArray::addiRowVector: you can't use this method on String array!");
        if (rankOf() != 2 || !row->isRowVector() || columns() != row->lengthOf())
            throw std::invalid_argument("NDArray::addiRowVector: wrong arguments !");

        int dimension = 1;

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), dimension);

        NDArray::prepareSpecialUse({this}, {row});
        NativeOpExecutioner::execBroadcast(_context, nd4j::broadcast::Ops::Add, _buffer, _shapeInfo, _bufferD, _shapeInfoD, row->_buffer, row->_shapeInfo, row->_bufferD, row->_shapeInfoD, this->buffer(), this->shapeInfo(), this->specialBuffer(), this->specialShapeInfo(), nullptr, 1, packX.specialShapeInfo(), packX.specialOffsets(), nullptr, nullptr);
        NDArray::registerSpecialUse({this}, {row});
    }

//////////////////////////////////////////////////////////////////////////
    void NDArray::addColumnVector(const NDArray *column, NDArray *target) const {
        if (isS())
            throw std::runtime_error("NDArray::addColumnVector: you can't use this method on String array!");
        if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() || !column->isColumnVector() || rows() != column->lengthOf())
            throw std::invalid_argument("NDArray::addColumnVector: wrong arguments !");
        if(target->_dataType !=  DataTypeUtils::pickPairwiseResultType(_dataType, column->_dataType))
            throw std::invalid_argument("NDArray::addColumnVector: wrong type of target array !");

        int dimension = 0;

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), dimension);

        NDArray::prepareSpecialUse({target}, {this, column});
        NativeOpExecutioner::execBroadcast(_context, nd4j::broadcast::Ops::Add, _buffer, _shapeInfo, _bufferD, _shapeInfoD, column->_buffer, column->_shapeInfo, column->_bufferD, column->_shapeInfoD, target->getBuffer(), target->getShapeInfo(), target->getSpecialBuffer(), target->getSpecialShapeInfo(), nullptr, 1, packX.specialShapeInfo(), packX.specialOffsets(), nullptr, nullptr);
        NDArray::registerSpecialUse({target}, {this, column});
    }

//////////////////////////////////////////////////////////////////////////
// This method adds given column to all columns in this NDArray, this array becomes affected
    void NDArray::addiColumnVector(const NDArray *column) {
        if (isS())
            throw std::runtime_error("NDArray::addiColumnVector: you can't use this method on String array!");
        if (rankOf() != 2 || !column->isColumnVector() || rows() != column->lengthOf())
            throw std::invalid_argument("NDArray::addiColumnVector: wrong arguments !");

        int dimension = 0;

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), dimension);

        NDArray::prepareSpecialUse({this}, {column});
        NativeOpExecutioner::execBroadcast(_context, nd4j::broadcast::Ops::Add, _buffer, _shapeInfo, _bufferD, _shapeInfoD, column->_buffer, column->_shapeInfo, column->_bufferD, column->_shapeInfoD, this->buffer(), this->shapeInfo(), this->specialBuffer(), this->specialShapeInfo(), nullptr, 1, packX.specialShapeInfo(), packX.specialOffsets(), nullptr, nullptr);
        NDArray::registerSpecialUse({this}, {column});
    }

//////////////////////////////////////////////////////////////////////////
// This method multiplies each column of this array by given argument-column, this array becomes affected
    void NDArray::muliColumnVector(const NDArray *column) {
        if (isS())
            throw std::runtime_error("NDArray::muliColumnVector: you can't use this method on String array!");
        if (rankOf() != 2 || !column->isColumnVector() || rows() != column->lengthOf())
            throw std::invalid_argument("NDArray::muliColumnVector: wrong arguments !");

        int dimension = 0;

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), dimension);

        NDArray::prepareSpecialUse({this}, {column});
        NativeOpExecutioner::execBroadcast(_context, nd4j::broadcast::Ops::Multiply, _buffer, _shapeInfo, _bufferD, _shapeInfoD, column->_buffer, column->_shapeInfo, column->_bufferD, column->_shapeInfoD, this->buffer(), this->shapeInfo(), this->specialBuffer(), this->specialShapeInfo(), nullptr, 1, packX.specialShapeInfo(), packX.specialOffsets(), nullptr, nullptr);
        NDArray::registerSpecialUse({this}, {column});
    }
    
    //////////////////////////////////////////////////////////////////////////
    // change an array by repeating it the number of times given by reps.
    NDArray NDArray::tile(const std::vector<Nd4jLong>& reps) const {
        int dim = reps.size();
        int product = 1;
        for(const auto& item : reps)
            product *= item;
        if(product == 0)
            throw std::runtime_error("NDArray::tile method: one of the elements in reps array is zero !");

        int rankOld = rankOf();
        int diff = rankOld - dim;
        if(product==1) {        // in this case 2 possibilities are present: just reshape or nothing to do
            NDArray result(*this);
            if(diff < 0) {      // reshape to higher dimension
                std::vector<Nd4jLong> shapeNew = reps;               // need to have unities at first "diff" positions of new shape
                memcpy(&shapeNew[-diff], result._shapeInfo+1, rankOld * sizeof(Nd4jLong));   // put old shape numbers at rest of positions
                result.reshapei(ordering(), shapeNew);
            }
            return result;             // nothing to do, if diff >= 0 -> identity tile
        }        

        // evaluate shapeInfo for resulting array
        auto newShapeInfo = ShapeUtils::evalTileShapeInfo(*this, reps, _context->getWorkspace());
        // create new buffer, in any case the memory amount new buffer points to is bigger then those for old _buffer
        int8_t * newBuff = nullptr;
        ALLOCATE(newBuff, _context->getWorkspace(), shape::length(newShapeInfo) * sizeOfT(), int8_t);
        // assign new shape and new buffer to resulting array
        NDArray result(newBuff, newShapeInfo, _context, true);

        prepareSpecialUse({&result}, {this});

        // fill newBuff, loop through all elements of newBuff
        // looping through _buffer goes automatically by means of getSubArrayIndex applying
        const auto resultLen = result.lengthOf();
        auto xType = this->dataType();
        auto stream = _context->getCudaStream();
        BUILD_SINGLE_SELECTOR(xType, tileKernelH, (this->_bufferD, this->_shapeInfoD, result._bufferD, result._shapeInfoD, resultLen, *stream), LIBND4J_TYPES);
        registerSpecialUse({&result}, {this});

        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    void NDArray::templatedAssign(void *xBuffer, Nd4jLong xOffset, const void *yBuffer, const Nd4jLong yOffset) const {
        if (xBuffer != nullptr && yBuffer != nullptr)
            *(reinterpret_cast<T*>(xBuffer) + xOffset) = *(reinterpret_cast<T const*>(yBuffer) + yOffset);
    }
    BUILD_SINGLE_TEMPLATE(template void NDArray::templatedAssign, (void *xBuffer, const Nd4jLong xOffset, const void *yBuffer, const Nd4jLong yOffset) const, LIBND4J_TYPES);


    //////////////////////////////////////////////////////////////////////////
    // change an array by repeating it the number of times given by reps.
    void NDArray::tile(const std::vector<Nd4jLong>& reps, NDArray& target) const {

        // evaluate true tile shapeInfo for comparison with target shapeInfo
        auto newShapeInfo = ShapeUtils::evalTileShapeInfo(*this, reps, _context->getWorkspace());
        if(!shape::equalsSoft(newShapeInfo, target.getShapeInfo()))  {
            delete []newShapeInfo;
            throw std::runtime_error("NDArray::tile method - shapeInfo of target array is not suitable for tile operation !");
        }

        prepareSpecialUse({&target}, {this});

        // fill newBuff, loop through all elements of newBuff
        // looping through _buffer goes automatically by means of getSubArrayIndex applying
        const int ews = target.ews();
        const int targetLen = target.lengthOf();
        auto stream = _context->getCudaStream();
        BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), tileKernelHH, (_bufferD, _shapeInfoD, target._bufferD, target._shapeInfoD, targetLen, ews, *stream), LIBND4J_TYPES, LIBND4J_TYPES);
        registerSpecialUse({&target}, {this});
    }

    //////////////////////////////////////////////////////////////////////////
    void NDArray::tile(NDArray& target) const {
        if(rankOf() > target.rankOf())
            throw std::runtime_error("NDArray::tile method - rank of target array must be bigger or equal to the rank of this array !");

        if(!ShapeUtils::areShapesBroadcastable(*this, target))
            throw std::runtime_error("NDArray::tile method - shapeInfo of target array is not suitable for tile operation !");

        prepareSpecialUse({&target}, {this});

        // fill newBuff, loop through all elements of newBuff
        // looping through _buffer goes automatically by means of getSubArrayIndex applying
        const auto ews = target.ews();
        const auto targetLen = target.lengthOf();
        auto stream = _context->getCudaStream();
        BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), tileKernelHH, (_bufferD, _shapeInfoD, target._bufferD, target._shapeInfoD, targetLen, ews, *stream), LIBND4J_TYPES, LIBND4J_TYPES);
        registerSpecialUse({&target}, {this});
    }

    //////////////////////////////////////////////////////////////////////////
    // create new  array by repeating it the number of times given by reps    
    NDArray* NDArray::repeat(int dimension, const std::vector<Nd4jLong>& repeats) const {
        auto outShape = ShapeUtils::evalRepeatShape(dimension, repeats, *this);

        // the size of outShape == rank
        int rank = rankOf();            // = outShape.size()

        std::vector<Nd4jLong> newShape(rank);
        for (int i = 0; i < rank; i++)
            newShape[i] = outShape[i];

        auto ret = new NDArray('c', outShape, _dataType,  _context);

        auto repeatDelta = shape::prodLong(newShape.data(), rank) / this->lengthOf();
        std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(rankOf(), {dimension});
        const Nd4jLong numTads = ShapeUtils::getNumOfSubArrs(_shapeInfo, dimsToExclude); //this->tensorsAlongDimension({dimension});
        //tadOnlyInputShapeInfo, tadInputOffsets, tadOnlyOutputShapeInfo, tadOutputOffsets;
        std::vector<int> copy({dimension});

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), copy);
        auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(ret->getShapeInfo(), copy);

        NDArray::prepareSpecialUse({ret}, {this});

        auto stream = _context->getCudaStream();
        BUILD_SINGLE_SELECTOR(_dataType, repeatKernelH, (_bufferD, ret->_bufferD, numTads, lengthOf(), ret->lengthOf(), packX.specialShapeInfo(), packX.specialOffsets(), packZ.specialShapeInfo(), packZ.specialOffsets(), *stream), LIBND4J_TYPES);

        NDArray::registerSpecialUse({ret}, {this});

        return ret;
    }

    //////////////////////////////////////////////////////////////////////////
    void NDArray::nullify() {
        if (isEmpty())
            return;

        if (isS())
            throw std::runtime_error("Can't nullify string array");

        if (isView()) {
            this->assign(0);
        } else {
            if (_buffer != nullptr)
                memset(_buffer, 0, this->lengthOf() * this->sizeOfT());

            cudaMemsetAsync(_bufferD, 0, this->lengthOf() * this->sizeOfT(), *(this->_context->getCudaStream()));
        }
    }

    //////////////////////////////////////////////////////////////////////////
    // fill array by repeating it the number of times given by reps
    void NDArray::repeat(int dimension, NDArray& target) const {

        if(dimension < 0)
            dimension += rankOf();

        if(rankOf() != target.rankOf())
            throw std::invalid_argument("NDArray::repeat(int dimension, NDArray& target) method: wrong rank of target array it must be equal to this array rank!");

        Nd4jLong repeatDelta = target.sizeAt(dimension) / sizeAt(dimension);

        if(repeatDelta == 0)
            throw std::invalid_argument("NDArray::repeat(int dimension, NDArray& target) method: wrong shape of target array!");


        std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(rankOf(), {dimension});
        const Nd4jLong numTads = ShapeUtils::getNumOfSubArrs(_shapeInfo, dimsToExclude);

        std::vector<int> copy({dimension});
        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), copy);
        auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(target.getShapeInfo(), copy);

        NDArray::prepareSpecialUse({&target}, {this});

        auto stream = _context->getCudaStream();
        BUILD_DOUBLE_SELECTOR(target._dataType, _dataType, repeatKernelHH, (_bufferD, target._bufferD, numTads, lengthOf(), packX.specialShapeInfo(), packX.specialOffsets(), packZ.specialShapeInfo(), packZ.specialOffsets(), *stream), LIBND4J_TYPES, LIBND4J_TYPES);

        NDArray::registerSpecialUse({&target}, {this});
    }

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    void NDArray::printSpecialBuffer(const char* msg, const int precision) const {

        if(_bufferD == nullptr || _length == 0)
            { printf("NDArray::printSpecialBuffer: special buffer is nullptr !\n"); return; }
        if(_length == 0)
            { printf("NDArray::printSpecialBuffer: array length is zero !\n"); return; }

        void* pHost = operator new(sizeof(T) * _length);

        if (ews() != 1) {
            for (uint i = 0; i < _length; i++)
                cudaMemcpyAsync(pHost + i * sizeof(T), _bufferD + getOffset(i) * sizeof(T), sizeof(T), cudaMemcpyDeviceToHost, *_context->getCudaStream());
        }
        else 
            cudaMemcpyAsync(pHost, _bufferD, sizeOfT() * _length, cudaMemcpyDeviceToHost, *_context->getCudaStream());

        cudaError_t cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
        if(cudaResult != 0)
            throw std::runtime_error("NDArray::printSpecialBuffer: cudaStreamSynchronize failed!");                

        for (uint i = 0; i < _length; i++) 
            printf("%.*f, ", precision, (double)reinterpret_cast<T*>(pHost)[i]);
        printf("\n");        

        operator delete(pHost);
    }
    template void NDArray::printSpecialBuffer<int>(const char* msg, const int precision) const;
    template void NDArray::printSpecialBuffer<float>(const char* msg, const int precision) const;
    template void NDArray::printSpecialBuffer<double>(const char* msg, const int precision) const;


} // end namespace nd4j



#endif

