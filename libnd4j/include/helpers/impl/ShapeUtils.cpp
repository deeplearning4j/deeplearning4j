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

//
// @author Yurii Shyrma
//

#include <algorithm>
#include <helpers/ShapeUtils.h>
#include <climits>
#include <numeric>
#include <algorithm>
#include <set>
#include <flatbuffers/util.h>


namespace nd4j {
     
//////////////////////////////////////////////////////////////////////////
// evaluate shape for array resulting from tensorDot operation, also evaluate shapes and dimensions permutations for transposition of two input arrays 
std::vector<Nd4jLong> ShapeUtils::evalShapeForTensorDot(const Nd4jLong* aShapeInfo, const Nd4jLong* bShapeInfo, std::vector<int> axesA, std::vector<int> axesB, std::vector<int>& permutAt, std::vector<int>& permutBt, std::vector<Nd4jLong>& shapeAt, std::vector<Nd4jLong>& shapeBt) {

    int axeAsize = (int) axesA.size();
    int axeBsize = (int) axesB.size();                 
    int aRank = aShapeInfo[0];
    int bRank = bShapeInfo[0];

    if(axeAsize != axeBsize)
        throw std::runtime_error("ShapeUtils::evalShapeForTensorDot method: the numbers of a axes and b axes to make dot product along must have identical values !");
    if(axeAsize > aRank || axeBsize > bRank)
        throw std::runtime_error("ShapeUtils::evalShapeForTensorDot method: the length of vector of a or b axes is larger than array rank !");
    
    // axes validation
    for (int i = 0; i < axeBsize; i++) {        
        if (axesA[i] < 0)
            axesA[i] += aRank;
        if (axesB[i] < 0)
            axesB[i] += bRank;
        if (aShapeInfo[axesA[i] + 1] != bShapeInfo[axesB[i] + 1])
            throw std::runtime_error("ShapeUtils::evalShapeForTensorDot method: the dimensions at given axes for both input arrays must be the same !");
    }
    
    // check whether axesA and axesB contain only unique numbers
    std::set<Nd4jLong> uniqueElems(axesA.begin(), axesA.end());
    if((int)uniqueElems.size() != axeAsize)
        throw std::runtime_error("ShapeUtils::evalShapeForTensorDot method: the vector of a axes contains duplicates !");
    uniqueElems.clear();
    uniqueElems = std::set<Nd4jLong>(axesB.begin(), axesB.end());
    if((int)uniqueElems.size() != axeBsize)
        throw std::runtime_error("ShapeUtils::evalShapeForTensorDot method: the vector of b axes contains duplicates !");

    std::vector<int> list_A, list_B;
    for (int i = 0; i < aRank; i++)
        if (std::find(axesA.begin(), axesA.end(), i) == axesA.end())
            list_A.emplace_back(i);
    for (int i = 0; i < bRank; i++)
        if (std::find(axesB.begin(), axesB.end(), i) == axesB.end())
            list_B.emplace_back(i);
    
    permutAt = list_A;
    permutAt.insert(permutAt.end(), axesA.begin(), axesA.end());
    permutBt = axesB;
    permutBt.insert(permutBt.end(), list_B.begin(), list_B.end());
    
    int n2 = 1;   
    for (int i = 0; i < axeAsize; i++)
        n2 *= aShapeInfo[axesA[i] + 1];
    shapeAt = {-1, n2};

    std::vector<Nd4jLong> oldShapeA;
    if (list_A.empty()) {
        oldShapeA.emplace_back(1);
    } else {
        oldShapeA.resize(list_A.size());
        for (int i = 0; i < (int) oldShapeA.size(); i++)
            oldShapeA[i] = aShapeInfo[list_A[i] + 1];
    }
    
    int n3 = 1;
    for (int i = 0; i < axeBsize; i++)
        n3 *= bShapeInfo[axesB[i] + 1];
    shapeBt = {n3, -1};
    
    std::vector<Nd4jLong> oldShapeB;
    if (list_B.empty()) {
        oldShapeB.emplace_back(1);
    } else {
        oldShapeB.resize(list_B.size()); 
        for (int i = 0; i < (int) oldShapeB.size(); i++)
            oldShapeB[i] = bShapeInfo[list_B[i] + 1];
    }
    
    std::vector<Nd4jLong> aPlusB(oldShapeA);
    aPlusB.insert(aPlusB.end(), oldShapeB.begin(), oldShapeB.end());            
    
    return aPlusB;
}

//////////////////////////////////////////////////////////////////////////
std::vector<Nd4jLong> ShapeUtils::evalShapeForTensorDot(const NDArray* a,   const NDArray* b,  const std::vector<int>& axesA, const std::vector<int>& axesB, std::vector<int>& permutAt, std::vector<int>& permutBt, std::vector<Nd4jLong>& shapeAt, std::vector<Nd4jLong>& shapeBt) {

    return evalShapeForTensorDot(a->getShapeInfo(), b->getShapeInfo(), axesA, axesB, permutAt, permutBt, shapeAt, shapeBt);
}

//////////////////////////////////////////////////////////////////////////
Nd4jLong* ShapeUtils::evalReduceShapeInfo(const char order, std::vector<int>& dimensions, const NDArray& arr, const bool keepDims, const bool supportOldShapes, nd4j::memory::Workspace* workspace) {
    return evalReduceShapeInfo(order, dimensions, arr.getShapeInfo(), keepDims, supportOldShapes, workspace);
}

//////////////////////////////////////////////////////////////////////////
// evaluate shape resulting from reduce operation
Nd4jLong* ShapeUtils::evalReduceShapeInfo(const char order, std::vector<int>& dimensions, const Nd4jLong *shapeInfo, const bool keepDims, const bool supportOldShapes, nd4j::memory::Workspace* workspace) {
    Nd4jLong* newShapeInfo = nullptr;

    int rank = shape::rank(const_cast<Nd4jLong*>(shapeInfo));
    
    if (dimensions.size() == 0) {                                               // return scalar or array with len=1 in this case 
        
        if(keepDims && rank > 1) {
            ALLOCATE(newShapeInfo, workspace, shape::shapeInfoLength(rank), Nd4jLong);
            newShapeInfo[0] = rank;
            for(int i = 0; i < rank; ++i)
                newShapeInfo[i+1] = 1;
            shape::updateStrides(newShapeInfo, order);
            return newShapeInfo;
        }
        else if(supportOldShapes) {
            ALLOCATE(newShapeInfo, workspace, 8, Nd4jLong);
            shape::shapeOldScalar(newShapeInfo, 'c');
        }
        else {
            ALLOCATE(newShapeInfo, workspace, 4, Nd4jLong);
            shape::shapeScalar(newShapeInfo);
        }
        return newShapeInfo;
    }

    shape::checkDimensions(rank, dimensions);
       
    int dimSize = dimensions.size();

    if(keepDims) {
        
        ALLOCATE(newShapeInfo, workspace, shape::shapeInfoLength(rank), Nd4jLong);
        newShapeInfo[0] = rank;
        for(int i = 0; i < rank; ++i)
            if (std::binary_search(dimensions.begin(), dimensions.end(), i))                       // dimensions is already sorted after shape::checkDimensions() has been applied
                newShapeInfo[i+1] = 1;
            else
                newShapeInfo[i+1] = shapeInfo[i+1];

        shape::updateStrides(newShapeInfo, order);

        return newShapeInfo;
    }
    
	int newRank = rank - dimSize;
	if (newRank==0 || (dimSize==1 && dimensions[0]==INT_MAX)) { 			// check whether given dimension is meant for the whole dimension
            
        if(supportOldShapes) {
            ALLOCATE(newShapeInfo, workspace, 8, Nd4jLong);
            shape::shapeOldScalar(newShapeInfo, 'c');
        }
        else {
            ALLOCATE(newShapeInfo, workspace, 4, Nd4jLong);
            shape::shapeScalar(newShapeInfo);
        }
            return newShapeInfo;
	}
       
    ALLOCATE(newShapeInfo, workspace, shape::shapeInfoLength(newRank), Nd4jLong);
    newShapeInfo[0] = newRank;                      // set rank
    int j=1;
    for(int i = 0; i < rank; ++i)
        if (!std::binary_search(dimensions.begin(), dimensions.end(), i))                       // dimensions is already sorted after shape::checkDimensions() has been applied
            newShapeInfo[j++] = shapeInfo[i+1];            
	   	
	//ensure whether vector has proper shape for old shape type
	if (newRank == 1 && supportOldShapes) {
        int oldValue = newShapeInfo[1];
        RELEASE(newShapeInfo, workspace);
        ALLOCATE(newShapeInfo, workspace, shape::shapeInfoLength(2), Nd4jLong);		// set newRank = 2
        newShapeInfo[0] = 2;
        if (dimensions[0] == 0) {
            newShapeInfo[1] = 1; 
            newShapeInfo[2] = oldValue;
        }
        else {
            newShapeInfo[1] = oldValue;
            newShapeInfo[2] = 1; 				
        }
    } 
    
	shape::updateStrides(newShapeInfo, order);
       
    return newShapeInfo;
}

//////////////////////////////////////////////////////////////////////////
// evaluate shape for array which is result of repeat operation applied to arr
    std::vector<Nd4jLong> ShapeUtils::evalRepeatShape(int dimension, const std::vector<Nd4jLong>& repeats, const NDArray& arr) {

    int rank = arr.rankOf();

    if (dimension < 0)
        dimension += rank;

    std::vector<Nd4jLong> reps;

    if ((int) reps.size() < rank) {
        if (dimension > 0) {
            for (int e = 0; e < rank - (int) repeats.size(); e++)
                reps.push_back(1);

            for (auto r: repeats)
                reps.push_back(r);
        } else {
            for (auto r: repeats)
                reps.push_back(r);

            for (int e = 0; e < rank - (int) repeats.size(); e++)
                reps.push_back(1);
        }
    }/* else {
        for (auto r: repeats)
            reps.push_back(r);
    }*/
    
    std::vector<Nd4jLong> outShape(rank);
    for (int i = 0; i < rank; i++)         
        outShape[i] = arr.sizeAt(i) * reps.at(i);        

    return outShape;
}


//////////////////////////////////////////////////////////////////////////
// evaluate shapeInfo of permuted array
    Nd4jLong* ShapeUtils::evalPermShapeInfo(const int* dimensions, const int rank, const NDArray& arr, nd4j::memory::Workspace* workspace) {

        if (!arr.nonNull())
            throw std::runtime_error("ShapeUtils<T>::evalPermShapeInfo static method: wrong arguments in pn/termute method: either array is nullptr!");

        if (rank != arr.rankOf())
            throw std::runtime_error("ShapeUtils<T>::evalPermShapeInfo static method: wrong arguments in pn/termute method: rank is not suitable!");
    
        auto shapeInfoLength = shape::shapeInfoLength(rank);
        // allocate memory for new array - shapeInfo

        Nd4jLong *shapeInfoNew = nullptr;
        ALLOCATE(shapeInfoNew, workspace, shapeInfoLength, Nd4jLong);
        // copy arr _shapeInfo into new array
        memcpy(shapeInfoNew, arr.getShapeInfo(), shape::shapeInfoByteLength(rank));
        // perform buffer permutation
        shape::doPermuteShapeInfo(shapeInfoNew, dimensions);

        return shapeInfoNew;
    }


    //////////////////////////////////////////////////////////////////////////
// evaluate shapeInfo of permuted array
    Nd4jLong* ShapeUtils::evalPermShapeInfo(const Nd4jLong *dimensions, const int rank, const NDArray& arr, nd4j::memory::Workspace* workspace) {

        if (!arr.nonNull())
            throw std::runtime_error("ShapeUtils<T>::evalPermShapeInfo static method: wrong arguments in pn/termute method: either array is nullptr!");

        if (rank != arr.rankOf())
            throw std::runtime_error("ShapeUtils<T>::evalPermShapeInfo static method: wrong arguments in pn/termute method: rank is not suitable!");

        auto shapeInfoLength = shape::shapeInfoLength(rank);
        // allocate memory for new array - shapeInfo

        Nd4jLong *shapeInfoNew = nullptr;
        ALLOCATE(shapeInfoNew, workspace, shapeInfoLength, Nd4jLong);
        // copy arr _shapeInfo into new array
        memcpy(shapeInfoNew, arr.getShapeInfo(), shape::shapeInfoByteLength(rank));
        // perform buffer permutation
        shape::doPermuteShapeInfo(shapeInfoNew, dimensions);

        return shapeInfoNew;
    }

//////////////////////////////////////////////////////////////////////////
// evaluate shapeInfo of transposed array
    Nd4jLong* ShapeUtils::evalTranspShapeInfo(const NDArray& arr, nd4j::memory::Workspace* workspace) {

        int rank = arr.rankOf();
        std::vector<int> dimensions(rank);
        for (int i = 0; i < rank; ++i)
            dimensions[i] = rank - 1 - i;

        auto shapeInfoNew = evalPermShapeInfo(dimensions.data(), dimensions.size(), arr, workspace);

        return shapeInfoNew;
    }

//////////////////////////////////////////////////////////////////////////
    bool ShapeUtils::insertDimension(int rank, Nd4jLong *shape, int axis, Nd4jLong dimension) {
        if (axis >= rank || axis <= -rank)
            return false;

        if (axis < 0)
            axis = rank + axis;

        std::vector<Nd4jLong> tmp;
        for (int e = 0; e < rank; e++) {
            if (shape[e] != 1)
                tmp.emplace_back(shape[e]);
        }

        tmp.insert(tmp.begin() + (Nd4jLong) axis, dimension);
        memcpy(shape, tmp.data(), tmp.size() * sizeof(Nd4jLong));

        return true;
    }

//////////////////////////////////////////////////////////////////////////
    bool ShapeUtils::copyVectorPart(std::vector<int>& target, std::vector<int>& source, int rank, int offset) {
        if (source.size() < offset + rank)
            return false;

        for (int e = offset; e < offset + rank; e++)
            target.push_back(source[e]);

        return true;
    }


//////////////////////////////////////////////////////////////////////////
// return new (shorter) sorted dimensions array without dimensions that are present in input vector
    std::vector<int> ShapeUtils::evalDimsToExclude(const int rank, const std::vector<int>& dimensions) {

    std::vector<int> newDimensions;
    auto size = dimensions.size();
    if(size == 0) {                          // if input vector is empty then return whole shape range
        newDimensions.resize(rank);
        std::iota(newDimensions.begin(), newDimensions.end(), 0);   // fill with 0, 1, ... rank-1
    }
    else {
        bool isAbsent;
        for(int i=0; i<rank; ++i) {
            isAbsent = true;
            for(int j=0; j<size; ++j) {
                int dim = dimensions[j] >= 0 ? dimensions[j] : dimensions[j] + rank;
                if(i == dim) {
                    isAbsent = false;
                    break;
                }
            }
            if(isAbsent)
                newDimensions.emplace_back(i);
        }
    }

    return newDimensions;
}

//////////////////////////////////////////////////////////////////////////
// check whether 2 arrays have mutually broadcastable shapes
// shape comparison starts from the end
bool ShapeUtils::areShapesBroadcastable(const NDArray &arr1, const NDArray &arr2) {
    return areShapesBroadcastable(arr1.getShapeInfo(), arr2.getShapeInfo());
}

bool ShapeUtils::areShapesBroadcastable(Nd4jLong *arr1, Nd4jLong *arr2) {
    int minRank = shape::rank(arr1) < shape::rank(arr2) ? shape::rank(arr1) : shape::rank(arr2);
       
    for (int i = -1; i >= -minRank; --i) 
        if (shape::sizeAt(arr1, i) != shape::sizeAt(arr2, i) && shape::sizeAt(arr1, i) != 1 && shape::sizeAt(arr2, i) != 1) return false;
    
    return true;
}

bool ShapeUtils::areShapesBroadcastable(const std::vector<Nd4jLong>& shape1, const std::vector<Nd4jLong>& shape2) {
    
    const auto rank1 = shape1.size();
    const auto rank2 = shape2.size();
    const int minRank = rank1 < rank2 ? rank1 : rank2;
    
    for (int i = 1; i <= minRank; ++i) 
        if (shape1[rank1-i] != shape2[rank2-i] && shape1[rank1-i] != 1 && shape2[rank2-i] != 1) 
            return false;
    
    return true;
}

//////////////////////////////////////////////////////////////////////////
// check the possibility of broadcast operation, if true then return shapeInfo of resulting array
// if evalMinMax == false the array with larger rank has to be passed as first argument
bool ShapeUtils::evalBroadcastShapeInfo(const NDArray &max, const NDArray &min, const bool evalMinMax, Nd4jLong*& resultShapeInfo, nd4j::memory::Workspace* workspace) {
    return evalBroadcastShapeInfo(max.getShapeInfo(), min.getShapeInfo(), evalMinMax, resultShapeInfo, workspace);
}

bool ShapeUtils::evalBroadcastShapeInfo(Nd4jLong *max, Nd4jLong *min, const bool evalMinMax, Nd4jLong*& resultShapeInfo, nd4j::memory::Workspace* workspace) {

    if (shape::isScalar(max) && shape::isScalar(min)) {
        resultShapeInfo = nullptr;
        if (shape::rank(max) >= shape::rank(min)) {
            COPY_SHAPE_EX(max, resultShapeInfo, workspace);
        } else {
            COPY_SHAPE_EX(min, resultShapeInfo, workspace);
        }
        return true;
    } else if ((shape::rank(max) == 0 && shape::isScalar(min))) {
        // X is the driver here
        resultShapeInfo = ShapeUtils::createScalarShapeInfo(workspace);
        return true;
    }

    // check whether broadcast operation is possible for input arrays
    if(!areShapesBroadcastable(max, min))
        return false;

    auto maxShapeInfo = max; //max.getShapeInfo();
    auto minShapeInfo = min; //min.getShapeInfo();

    if(evalMinMax && (shape::rank(max) < shape::rank(min))) {
        maxShapeInfo = min;
        minShapeInfo = max;
    }
       
    const auto maxRank      = shape::rank(maxShapeInfo);
    const auto minRank      = shape::rank(minShapeInfo);
    
    // evaluate shapeInfo for resulting array
    if(resultShapeInfo != nullptr)
        throw std::runtime_error("std::runtime_error(ShapeUtils::evalBroadcastShapeInfo method: the input pointer on shapeInfo must be empty (=nullptr) !");
    
    ALLOCATE(resultShapeInfo, workspace, shape::shapeInfoLength(maxRank), Nd4jLong);

    // FIXME: get rid of memcpy here
    memcpy(resultShapeInfo, maxShapeInfo, shape::shapeInfoByteLength(maxRank));
    for (int i = 0; i < minRank; ++i)
        if(maxShapeInfo[maxRank-i] < minShapeInfo[minRank-i])
            resultShapeInfo[maxRank - i] = minShapeInfo[minRank-i];

    shape::updateStrides(resultShapeInfo, shape::order(max));

    return true;
}

//////////////////////////////////////////////////////////////////////////
// check the possibility of broadcast operation for set of arrays, if true then return resulting broadcasted shapeInfo
bool ShapeUtils::evalCommonBroadcastShapeInfo(const std::vector<const NDArray*>& arrays, Nd4jLong*& resultShapeInfo, memory::Workspace* workspace) {

    if(resultShapeInfo != nullptr)
        throw std::runtime_error("ShapeUtils::evalCommonBroadcastShapeInfo method: the input pointer on shapeInfo must be empty (=nullptr) !");

    int size = arrays.size();
    int maxRank = arrays[size - 1]->rankOf();

    for(int i = 0; i < size - 1; ++i) {
        if(arrays[i]->rankOf() > maxRank)
            maxRank = arrays[i]->rankOf();
        for(int j = i + 1; j < size; ++j)
            if(!areShapesBroadcastable(*arrays[i], *arrays[j]))
                return false;
    }

    ALLOCATE(resultShapeInfo, workspace, shape::shapeInfoLength(maxRank), Nd4jLong);
    memset(resultShapeInfo, 0, shape::shapeInfoByteLength(maxRank));
    resultShapeInfo[0] = maxRank;

    for(const auto& item : arrays ) {
        for(int i = -1; i >= -item->rankOf(); --i) 
            if(resultShapeInfo[i + 1 + maxRank] < item->sizeAt(i))
                resultShapeInfo[i + 1 + maxRank] = item->sizeAt(i);
    }

    shape::updateStrides(resultShapeInfo, arrays[0]->ordering());

    return true;
}


//////////////////////////////////////////////////////////////////////////
// return sorted vector of dimensions of array with larger dimensions number along which two input arrays have same shape
// the array with larger dimensions number has to be passed as first argument
std::vector<int> ShapeUtils::getDimsWithSameShape(const NDArray& max, const NDArray& min) {

    std::vector<int> result;
    auto maxShapeInfo = max.getShapeInfo(); 
    auto minShapeInfo = min.getShapeInfo();
    int  maxRank      = maxShapeInfo[0];
    int  minRank      = minShapeInfo[0];

    for(int i = 1; i <= minRank; ++i)
        if(minShapeInfo[i] == maxShapeInfo[maxRank - minRank + i])
            result.emplace_back(maxRank - minRank + i - 1);

    return result;
}


//////////////////////////////////////////////////////////////////////////
// return absolute index of array min, min is sub-array of max, index to be returned is min index and it corresponds maxIdx of max array 
Nd4jLong ShapeUtils::getSubArrayIndex(const Nd4jLong* maxShapeInfo, const Nd4jLong* minShapeInfo, const Nd4jLong maxIdx) {
    // check shape consistence 
    if(maxShapeInfo[0] < minShapeInfo[0])
        throw std::runtime_error("ShapeUtils::getSubArrayIndex: rank of max-array must be greater or equal to min-array rank !");
    
    for(int i = 0; i < minShapeInfo[0]; ++i)
        // if((maxShapeInfo[maxShapeInfo[0] - i] < minShapeInfo[minShapeInfo[0] - i]) || (maxShapeInfo[maxShapeInfo[0] - i] % minShapeInfo[minShapeInfo[0] - i] != 0) )        
        if(maxShapeInfo[maxShapeInfo[0] - i] < minShapeInfo[minShapeInfo[0] - i])        
            throw std::runtime_error("ShapeUtils::getSubArrayIndex: some of dimension shape of max-array is smaller than those of min-array or the max shape is not multiple of min shape !");

    return shape::subArrayIndex(maxShapeInfo, minShapeInfo, maxIdx);
}

//////////////////////////////////////////////////////////////////////////
// evaluate shapeInfo for resulting array from tile operation
Nd4jLong* ShapeUtils::evalTileShapeInfo(const NDArray& arr, const std::vector<Nd4jLong>& reps, nd4j::memory::Workspace* workspace) {
    // check whether reps contains at least one zero (then throw exception) or whether all elements in reps are unities (then simply reshape or do nothing)
    int dim = reps.size();  
    int product = 1;
    for(const auto& item : reps)
        product *= item;
    if(product == 0)
        throw std::runtime_error("NDArray::tile method: one of the elements in reps array is zero !");

    int rankOld = arr.rankOf();
    int diff = rankOld - dim;
    
    // evaluate new shapeInfo
    Nd4jLong* newShapeInfo = nullptr;    
    if(diff < 0) {      
        ALLOCATE(newShapeInfo, workspace, shape::shapeInfoLength(dim), Nd4jLong);
        newShapeInfo[0] = dim;                  // set new rank
        for(int i=1; i <= -diff; ++i)
            newShapeInfo[i] = 1;                // set unities to be new dimensions at left-hand side of newShapeInfo shape place
        memcpy(newShapeInfo + 1 - diff, arr.getShapeInfo() + 1, rankOld*sizeof(Nd4jLong));       // copy old dimensions to the right-hand side of newShapeInfo shape place
        for(int i=1; i <= dim; ++i)
            newShapeInfo[i] *= reps[i - 1];     // set new shape by multiplying old dimensions by corresponding numbers from reps 
    }
    else {      
        ALLOCATE(newShapeInfo, workspace, shape::shapeInfoLength(rankOld), Nd4jLong);
        memcpy(newShapeInfo, arr.getShapeInfo(), shape::shapeInfoByteLength(rankOld));      // copy all elements of _shapeInfo to newShapeInfo
        for(int i=1; i <= dim; ++i)
            newShapeInfo[rankOld + 1 - i] *= reps[dim - i];     // set new shape by multiplying old dimensions by corresponding numbers from reps 
    }
    shape::updateStrides(newShapeInfo, arr.ordering());
    
    return newShapeInfo;
}

//////////////////////////////////////////////////////////////////////////
    std::vector<int> ShapeUtils::convertAxisToTadTarget(int rank, std::initializer_list<int> axis) {
        std::vector<int> newAxis(axis);
        return convertAxisToTadTarget(rank, newAxis);
    }

//////////////////////////////////////////////////////////////////////////
    std::vector<int> ShapeUtils::convertAxisToTadTarget(int rank, std::vector<int>& axis) {
        std::vector<int> newAxis;
        for (int e = 0; e < rank; e++) {
            if (std::find(axis.begin(), axis.end(), e) == axis.end())
                newAxis.emplace_back(e);
        }

        return newAxis;
    }

    std::vector<Nd4jLong> ShapeUtils::pullShapeFromShapeInfo(Nd4jLong *shapeInfo) {
        std::vector<Nd4jLong> shape(shape::rank(shapeInfo));

        for (int e = 0; e < shape.size(); e++)
            shape[e] = shape::shapeOf(shapeInfo)[e];

        return shape;
    }

    std::string ShapeUtils::shapeAsString(const NDArray* array) {
        std::string result;

        result.append("[");
        for (int e = 0; e < array->rankOf(); e++) {
            result += flatbuffers::NumToString(array->sizeAt(e));
            if (e < array->rankOf() - 1)
                result.append(", ");
        }
        result.append("]");

        return result;
    }

    std::string ShapeUtils::shapeAsString(const std::vector<Nd4jLong>& shape) {
        std::string result;

        result.append("[");
        for (int e = 0; e < shape.size(); e++) {
            result += flatbuffers::NumToString(shape.at(e));
            if (e < shape.size() - 1)
                result.append(", ");
        }
        result.append("]");

        return result;
    }

    std::string ShapeUtils::shapeAsString(const Nd4jLong* shapeInfo) {
        
        if(!shapeInfo)
            throw std::runtime_error("ShapeUtils<T>::shapeAsString method: input shapeInfo must not be nullptr !");
        
        std::string result;

        result.append("[");
        for (int e = 0; e < shapeInfo[0]; e++) {
            result += flatbuffers::NumToString(shapeInfo[e+1]);
            if (e < shapeInfo[0] - 1)
                result.append(", ");
        }
        result.append("]");

        return result;
    }


    std::string ShapeUtils::shapeAsString(const int rank, const Nd4jLong* shapeInfo) {
        if(!shapeInfo)
            throw std::runtime_error("ShapeUtils<T>::shapeAsString method: input shapeInfo must not be nullptr !");

        std::string result;

        result.append("[");
        for (int e = 0; e < rank; e++) {
            result += flatbuffers::NumToString(shapeInfo[e]);
            if (e < rank - 1)
                result.append(", ");
        }
        result.append("]");

        return result;
    }

//////////////////////////////////////////////////////////////////////////
// evaluate shapeInfo for diagonal array which is made using input arr elements as diagonal
Nd4jLong* ShapeUtils::evalDiagShapeInfo(const Nd4jLong* shapeInfoConst, nd4j::memory::Workspace* workspace){
    auto shapeInfo = const_cast<Nd4jLong*>(shapeInfoConst);
    
    const auto rank = shape::rank(shapeInfo);

    Nd4jLong* outputShapeInfo = nullptr;

    if(shape::isVector(shapeInfo) || shape::isScalar(shapeInfo)) {
        ALLOCATE(outputShapeInfo, workspace, shape::shapeInfoLength(2), Nd4jLong);
        outputShapeInfo[0] = 2;
        outputShapeInfo[1] = outputShapeInfo[2] = shape::length(shapeInfo);
    }
    else {
        ALLOCATE(outputShapeInfo, workspace, shape::shapeInfoLength(2*rank), Nd4jLong);
        outputShapeInfo[0] = 2*rank;
        for(int i = 1; i <= rank; ++i)
            outputShapeInfo[i] = outputShapeInfo[i + rank] = shapeInfo[i];
    }
        
    shape::updateStrides(outputShapeInfo, shape::order(shapeInfo));

    return outputShapeInfo;
}

std::vector<int> ShapeUtils::evalBroadcastBackwardAxis(const Nd4jLong *operandShapeInfo, const Nd4jLong *resultShapeInfo) {
    // rRank >= oRank always  !!
    const auto oRank = shape::rank(operandShapeInfo);
    const auto rRank = shape::rank(resultShapeInfo);
    const auto diff  = rRank - oRank;
    std::vector<int> axis;

    for(int i = 0; i < rRank; ++i)
        if(i < diff || shape::sizeAt(operandShapeInfo, i - diff) != shape::sizeAt(resultShapeInfo, i))
            axis.push_back(i);        

    return axis;
}

////////////////////////////////////////////////////////////////////////////////
Nd4jLong* ShapeUtils::matrixProductShape(Nd4jLong* theFirstShape, Nd4jLong* theSecondShape, bool shouldTranspondFirst, bool shouldTranspondSecond,
    nd4j::memory::Workspace* workspace) {

    auto inA = theFirstShape;
    auto inB = theSecondShape;
    Nd4jLong *shape;
    ALLOCATE(shape, workspace, shape::shapeInfoLength(2), Nd4jLong);

    Nd4jLong *tmpA, *tmpB;
    COPY_SHAPE_EX(inA, tmpA, workspace);
    COPY_SHAPE_EX(inB, tmpB, workspace);



    if (shouldTranspondFirst)
        shape::transposeInplace(tmpA);

    if (shouldTranspondSecond)
        shape::transposeInplace(tmpB);


    if (shape::rank(tmpA) == 1 && shape::isMatrix(tmpB)) {
        // special case here
        Nd4jLong *newShape;
        shape[0] = 1;
        shape[1] = tmpB[2];
        ALLOCATE(newShape, workspace, shape::shapeInfoLength(2), Nd4jLong);
        shape::shapeBufferFortran(2, shape, newShape);

        RELEASE(shape, workspace);
        RELEASE(tmpA, workspace);
        RELEASE(tmpB, workspace);

        return newShape;
    } else if (shape::isScalar(tmpA) && shape::isScalar(tmpB)) {
        // just scalar vs scalar
        shape[0] = 1;
        shape[1] = 1;
    }  else if (shape::isMatrix(tmpA) && shape::isVector(tmpB)) {
        // gemv case
        if (shape::rank(tmpB) == 2) {
            shape[0] = tmpA[1];
            shape[1] = tmpB[2];
        } else {
            // we have new 1D shape here
            Nd4jLong *newShape;
            ALLOCATE(newShape, workspace, shape::shapeInfoLength(2), Nd4jLong);
            shape::shapeVector(tmpA[1], newShape);

            RELEASE(shape, workspace);
            RELEASE(tmpA, workspace);
            RELEASE(tmpB, workspace);

            return newShape;
        }
    } else if ((shape::isMatrix(tmpA) && shape::isMatrix(tmpB)) || 
               (shape::isVector(tmpA) && shape::isMatrix(tmpB)) || 
               (shape::isColumnVector(tmpA) && shape::isVector(tmpB))) {
        // gemm case
        shape[0] = tmpA[1];
        shape[1] = tmpB[2];
    } else if ((shape::isVector(tmpA) && shape::isScalar(tmpB)) || 
        (shape::isScalar(tmpA) && shape::isVector(tmpB))) {
        // element-wise
        shape[0] = 1;
        shape[1] = (int) nd4j::math::nd4j_max<Nd4jLong>(shape::length(tmpA), shape::length(tmpB));
    } else if (shape::isRowVector(tmpA) && shape::isRowVector(tmpB)) {
        // dot case
        shape[0] = 1;
        shape[1] = 1;
    } else if (shape::isRowVector(tmpA) && shape::isColumnVector(tmpB)) {
        // dot case
        shape[0] = 1;
        shape[1] = 1;
    }

    Nd4jLong *newShape;
    ALLOCATE(newShape, workspace, shape::shapeInfoLength(2), Nd4jLong);
    shape::shapeBufferFortran(2, shape, newShape);

    RELEASE(shape, workspace);

    RELEASE(tmpA, workspace);
    RELEASE(tmpB, workspace);
    return newShape;
}

    Nd4jLong* ShapeUtils::createScalarShapeInfo(nd4j::DataType dataType, nd4j::memory::Workspace* workspace) {
        auto res = createScalarShapeInfo(workspace);
        nd4j::ArrayOptions::setDataType(res, dataType);
        return res;
    }

    Nd4jLong* ShapeUtils::createScalarShapeInfo(nd4j::memory::Workspace* workspace) {
        Nd4jLong *newShape;
        ALLOCATE(newShape, workspace, shape::shapeInfoLength(0), Nd4jLong);
        newShape[0] = 0;
        newShape[1] = 0;
        newShape[2] = 1;
        newShape[3] = 99;

        return newShape;
    }

    Nd4jLong* ShapeUtils::createVectorShapeInfo(nd4j::DataType dataType, Nd4jLong length, nd4j::memory::Workspace *workspace) {
        auto res = createVectorShapeInfo(length, workspace);
        nd4j::ArrayOptions::setDataType(res, dataType);
        return res;
    }

    Nd4jLong* ShapeUtils::createVectorShapeInfo(Nd4jLong length, nd4j::memory::Workspace* workspace) {
        Nd4jLong *newShape;
        ALLOCATE(newShape, workspace, shape::shapeInfoLength(1), Nd4jLong);

        newShape[0] = 1;
        newShape[1] = length;
        newShape[2] = 1;
        newShape[3] = 0;
        newShape[4] = 1;
        newShape[5] = 99;

        return newShape;
    }

////////////////////////////////////////////////////////////////////////////////
std::vector<int> ShapeUtils::evalPermutFromTo(const std::vector<Nd4jLong>& shapeFrom, const std::vector<Nd4jLong>& shapeTo) {
    auto rank = shapeFrom.size();
    if(rank != shapeTo.size())
        throw std::runtime_error("ShapeUtils::evalPermutFromTo static method: the input shapes are not suitable for mutual permutation !");

    if (std::equal(begin(shapeFrom), end(shapeFrom), begin(shapeTo)))       // if shapes are identical (permutation is unnecessary) then return empty vector
        return std::vector<int>();

    std::vector<int> permutation(rank, -2);                                 // vector to be returned
    std::vector<Nd4jLong> shapeTo2(shapeTo);                                     // make copy of const vector since we will change the content of shapeTo

    for(int i=0; i<rank; ++i)
        for(int j=0; j<rank; ++j)
            if(shapeFrom[i] == shapeTo2[j]) {
                permutation[j] = i;        
                shapeTo2[j] = -2;                                           // mark coincidence as -2 in order to not account index of shapeTo twice
                break;
            }   

    if(std::find(begin(permutation), end(permutation), -2) != end(permutation))      // if -2 is still present in vector then permutation is impossible
        throw std::runtime_error("ShapeUtils::evalPermutFromTo static method: the input shapes are not suitable for mutual permutation !");

    return permutation;        
}


////////////////////////////////////////////////////////////////////////////////
std::vector<Nd4jLong> ShapeUtils::composeShapeUsingDimsAndIdx(const std::vector<int>& dimsAndIdx) {
    auto size = dimsAndIdx.size();
    if(size % 2 != 0)
        throw std::runtime_error("ShapeUtils::composeShapeUsingDimsAndIdx static method: the size of input vector must be even !");

    size /= 2;

    std::vector<Nd4jLong> shape(size);
    int index;

    for(int i = 0; i < size; ++i) {
        index = dimsAndIdx[i + size];
        if(index > size-1)
            throw std::runtime_error("ShapeUtils::composeShapeUsingDimsAndIdx static method: input index is too large !");
        shape[index] = dimsAndIdx[i];
    }

    return shape;
}


////////////////////////////////////////////////////////////////////////////////
std::vector<Nd4jLong> ShapeUtils::evalShapeForMatmul(const Nd4jLong* xShapeInfo, const Nd4jLong* yShapeInfo, const bool transX, const bool transY) {

    const auto xRank = xShapeInfo[0];
    const auto yRank = yShapeInfo[0];

    const Nd4jLong x0Dim = transX ? xShapeInfo[xRank]   : xShapeInfo[xRank-1];
    const Nd4jLong y0Dim = transY ? yShapeInfo[yRank]   : yShapeInfo[yRank-1];
    const Nd4jLong x1Dim = transX ? xShapeInfo[xRank-1] : xShapeInfo[xRank];
    const Nd4jLong y1Dim = transY ? yShapeInfo[yRank-1] : yShapeInfo[yRank];
    

    if(xRank == 1 && yRank == 1) {   // dot case, output is scalar
        if(xShapeInfo[1] != yShapeInfo[1]) {
            nd4j_printf("ShapeUtils::evalShapeForMatmul method: since input arrays are vectors they must have the same length, but got x length = %i, y length = %i !", xShapeInfo[1], yShapeInfo[1]); 
            throw std::invalid_argument("");
        }
        return std::vector<Nd4jLong>({0});
    }


    if(xRank == 1 && yRank == 2) {  // vector x matrix, i.e. [4] x [4,5] = [5], output is vector
        if(xShapeInfo[1] != y0Dim) {
            nd4j_printf("ShapeUtils::evalShapeForMatmul method: input arrays have inconsistent shapes for vector-matrix product: x %s, y %s !", ShapeUtils::shapeAsString(xShapeInfo).c_str(), ShapeUtils::shapeAsString(yShapeInfo).c_str());
            throw std::invalid_argument("");
        }
        return std::vector<Nd4jLong>({y1Dim});
    }


    if(xRank == 2 && yRank == 1) {  // matrix x vector , i.e. [4,5] x [5] = [4], output is vector
        if(x1Dim != yShapeInfo[1]) {
            nd4j_printf("ShapeUtils::evalShapeForMatmul method: input arrays have inconsistent shapes for vector-matrix product: x %s, y %s !", ShapeUtils::shapeAsString(xShapeInfo).c_str(), ShapeUtils::shapeAsString(yShapeInfo).c_str());
            throw std::invalid_argument("");
        }        
        return std::vector<Nd4jLong>({x0Dim});
    }

    
    // rest cases - usual 2Dx2D or batched mmul    
    if(xRank != yRank) {
        nd4j_printf("ShapeUtils::evalShapeForMatmul static method: the ranks of arrays must be the same, but got xRank = %i and yRank = %i ! \n", xRank, yRank);
        throw std::invalid_argument("");
    }   

    if(x1Dim != y0Dim) {
        nd4j_printf("ShapeUtils::evalShapeForMatmul static method: input shapes are inconsistent: xDim %i != yDim %i \n", x1Dim, y0Dim);
        throw std::invalid_argument("");       
    }

    for(int i = 0; i < xRank - 2; ++i)
        if(xShapeInfo[i+1] != yShapeInfo[i+1]) {
            nd4j_printf("ShapeUtils::evalShapeForMatmul static method: input shapes are inconsistent: xShape = %s, yShape = %s ! \n", ShapeUtils::shapeAsString(xShapeInfo).c_str(), ShapeUtils::shapeAsString(yShapeInfo).c_str());
            throw std::invalid_argument("");       
        }    

    std::vector<Nd4jLong> cShape(xRank);

    // copy batch part of shape (if present)
    for(int i = 0; i < xRank - 2; ++i)
        cShape[i] = xShapeInfo[i+1];
    // copy rest part of shape (two dims: multiplication part)
    cShape[xRank-2] = x0Dim;
    cShape[xRank-1] = y1Dim;

    return cShape;
}

////////////////////////////////////////////////////////////////////////////////
Nd4jLong ShapeUtils::getNumOfSubArrs(const Nd4jLong* shapeInfo, const std::vector<int>& dimsToExclude) {

    Nd4jLong numOfSubArrs = 1;

    for(const auto& dim : dimsToExclude)
        numOfSubArrs *= shapeInfo[dim + 1];

    return numOfSubArrs;
}

////////////////////////////////////////////////////////////////////////////////
void ShapeUtils::evalIdxRangesForSubArr(const Nd4jLong subArrIdx,  const Nd4jLong* shapeInfo, const std::vector<int>& dimsToExclude, Nd4jLong* idxRanges) {

    const auto rank = shape::rank(shapeInfo);
    const auto subArrRank = static_cast<int>(dimsToExclude.size());

    if(subArrRank > rank)
        throw std::invalid_argument("ShapeUtils::evalIdxRangesForSubArr static method: dimsToExclude is empty or has size > rank of array !");

    if(subArrRank == 0) { // means whole array
        memset(idxRanges, 0, 2 * rank * sizeof(Nd4jLong));
        return;
    }

    std::vector<Nd4jLong> shapeOfSubArr(subArrRank), indexes(subArrRank);    
    for(int i = 0; i < subArrRank; ++i)
        shapeOfSubArr[i] = shapeInfo[dimsToExclude[i] + 1];

    shape::ind2subC(subArrRank, shapeOfSubArr.data(), subArrIdx, indexes.data());

    memset(idxRanges, 0, 2 * rank * sizeof(Nd4jLong));

    for(int i = 0; i < subArrRank; ++i) {
        int currIdx = 2 * dimsToExclude[i];
        idxRanges[currIdx]    = indexes[i];
        idxRanges[currIdx +1] = indexes[i] + 1;
    }
}

    Nd4jLong* ShapeUtils::createShapeInfo(nd4j::DataType dataType, const char order, const std::vector<Nd4jLong> shapeOnly, memory::Workspace* workspace) {
        auto res = createShapeInfo(order, shapeOnly, workspace);
        nd4j::ArrayOptions::setDataType(res, dataType);
        return res;
    }

////////////////////////////////////////////////////////////////////////////////
Nd4jLong* ShapeUtils::createShapeInfo(const char order, const std::vector<Nd4jLong> shapeOnly, memory::Workspace* workspace) {
    int rank = shapeOnly.size();

    if(shapeOnly[0] == 0) // scalar case
        rank = 0;
    
    Nd4jLong* shapeInfo = nullptr;
    
    if(rank == 0) {    // scalar case
        shapeInfo = ShapeUtils::createScalarShapeInfo(workspace);
    }
    else {
        ALLOCATE(shapeInfo, workspace, shape::shapeInfoLength(rank), Nd4jLong);
        shapeInfo[0] = rank;
        for(int i = 0; i < rank; ++i)
            shapeInfo[i + 1] = shapeOnly[i];
        shape::updateStrides(shapeInfo, order);
    }

    return shapeInfo;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<Nd4jLong> ShapeUtils::evalDimsWithoutUnities(const Nd4jLong* shapeInfo) {

    std::vector<Nd4jLong> result;
    for(int i = 1; i <= shapeInfo[0]; ++i)
        if(shapeInfo[i] != 1)
            result.push_back(shapeInfo[i]);

    if(result.size() == 0)  // shape consists of unities only 
        return std::vector<Nd4jLong>(1,1);  // return [1]

    return result;
}

}


