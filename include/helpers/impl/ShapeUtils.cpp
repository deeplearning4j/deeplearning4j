//
// @author iuriish@yahoo.com
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
// evaluate shape for array resulting from tensorDot operation, also evaluate shapes and permutation dimensions for transposition of two input arrays 
template<typename T>
std::vector<int> ShapeUtils<T>::evalShapeForTensorDot(const nd4j::NDArray<T>* a, const nd4j::NDArray<T>* b, std::vector<int>& axes_0, std::vector<int>& axes_1, std::vector<int>& permutAt, std::vector<int>& permutBt, std::vector<int>& shapeAt, std::vector<int>& shapeBt) {

    int axe0_size = (int) axes_0.size();
    int axe1_size = (int) axes_1.size();                 
    if(axe0_size != axe1_size)
        throw "ShapeUtils::evalShapeForTensorDot method: the numbers of a axes and b axes to make dot product along must have identical values !";
    if(axe0_size > a->rankOf() || axe1_size > b->rankOf())
        throw "ShapeUtils::evalShapeForTensorDot method: the length of vector of a or b axes is larger than array rank !";
    // validating axes
    for (int i = 0; i < axe1_size; i++) {
        if (a->sizeAt(axes_0[i]) != b->sizeAt(axes_1[i]))
            throw "ShapeUtils::evalShapeForTensorDot method: the dimensions at given axes for both input arrays must be the same !";
        if (axes_0[i] < 0)
            axes_0[i] += a->rankOf();
        if (axes_1[i] < 0)
            axes_1[i] += b->rankOf();
    }
    // check whether axes_0 and axes_1 contain on;y unique numbers
    std::set<T> uniqueElems(axes_0.begin(), axes_0.end());
    if((int)uniqueElems.size() != axe0_size)
        throw "ShapeUtils::evalShapeForTensorDot method: the vector of a axes contains duplicates !";
    uniqueElems.clear();
    uniqueElems = std::set<T>(axes_1.begin(), axes_1.end());
    if((int)uniqueElems.size() != axe1_size)
        throw "ShapeUtils::evalShapeForTensorDot method: the vector of b axes contains duplicates !";

    std::vector<int> list_A, list_B;
    for (int i = 0; i < a->rankOf(); i++)
        if (std::find(axes_0.begin(), axes_0.end(), i) == axes_0.end())
            list_A.emplace_back(i);
    for (int i = 0; i < b->rankOf(); i++)
        if (std::find(axes_1.begin(), axes_1.end(), i) == axes_1.end())
            list_B.emplace_back(i);
    
    permutAt = list_A;
    permutAt.insert(permutAt.end(), axes_0.begin(), axes_0.end());
    permutBt = axes_1;
    permutBt.insert(permutBt.end(), list_B.begin(), list_B.end());
    
    int n2 = 1;   
    for (int i = 0; i < axe0_size; i++)
        n2 *= a->sizeAt(axes_0[i]);
    shapeAt = {-1, n2};

    std::vector<int> oldShapeA;
    if (list_A.empty()) {
        oldShapeA.emplace_back(1);
    } else {
        oldShapeA.resize(list_A.size());
        for (int i = 0; i < (int) oldShapeA.size(); i++)
            oldShapeA[i] = a->sizeAt(list_A[i]);
    }
    
    int n3 = 1;
    for (int i = 0; i < axe1_size; i++)
        n3 *= b->sizeAt(axes_1[i]);
    shapeBt = {n3, -1};
    
    std::vector<int> oldShapeB;
    if (list_B.empty()) {
        oldShapeB.emplace_back(1);
    } else {
        oldShapeB.resize(list_B.size()); 
        for (int i = 0; i < (int) oldShapeB.size(); i++)
            oldShapeB[i] = b->sizeAt(list_B[i]);
    }
    
    std::vector<int> aPlusB(oldShapeA);
    aPlusB.insert(aPlusB.end(), oldShapeB.begin(), oldShapeB.end());            
    
    return aPlusB;
}


    template<typename T>
    int* ShapeUtils<T>::evalReduceShapeInfo(const char order, std::vector<int>& dimensions, const NDArray<T>& arr, const bool keepDims) {
        return evalReduceShapeInfo(order, dimensions, arr.getShapeInfo(), keepDims, arr.getWorkspace());
    }

//////////////////////////////////////////////////////////////////////////
// evaluate resulting shape after reduce operation
template<typename T>
int* ShapeUtils<T>::evalReduceShapeInfo(const char order, std::vector<int>& dimensions, const int *shapeInfo, const bool keepDims, nd4j::memory::Workspace* workspace) {
        
    int rank = shape::rank(const_cast<int*>(shapeInfo));
    shape::checkDimensions(rank, dimensions);
       
	int* newShapeInfo = nullptr;
    int dimSize = dimensions.size();

    if(keepDims) {
        ALLOCATE(newShapeInfo, workspace, shape::shapeInfoLength(rank), int);
        newShapeInfo[0] = rank;
        for(int i = 0; i < rank; ++i)
            if (std::binary_search(dimensions.begin(), dimensions.end(), i))                       // dimensions is already sorted after shape::checkDimensions() has been applied
                newShapeInfo[i+1] = 1;
            else
                newShapeInfo[i+1] = shapeInfo[i+1];
    }
    else {
	   int newRank = rank - dimSize;
	   if (newRank==0 || (dimSize==1 && dimensions[0]==INT_MAX)) { 			// check whether given dimension is meant for the whole dimension
            ALLOCATE(newShapeInfo, workspace, 8, int);						// set newRank = 2
            newShapeInfo[0] = 2;
            newShapeInfo[1] = 1;
            newShapeInfo[2] = 1;			
	   }
        else {
            ALLOCATE(newShapeInfo, workspace, shape::shapeInfoLength(newRank), int);
            newShapeInfo[0] = newRank;                      // set rank
            int j=1;
            for(int i = 0; i < rank; ++i)
                if (!std::binary_search(dimensions.begin(), dimensions.end(), i))                       // dimensions is already sorted after shape::checkDimensions() has been applied
                    newShapeInfo[j++] = shapeInfo[i+1];            
	   }		
	   //ensure vector is proper shape 
	   if (newRank == 1) {
            int oldValue = newShapeInfo[1];
            RELEASE(newShapeInfo, workspace);
            ALLOCATE(newShapeInfo, workspace, shape::shapeInfoLength(2), int);		// set newRank = 2
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
    }

	shape::updateStrides(newShapeInfo, order);
       
    return newShapeInfo;
}

//////////////////////////////////////////////////////////////////////////
// evaluate shape for array which is result of repeat operation applied to arr
	template<typename T>
    std::vector<int> ShapeUtils<T>::evalRepeatShape(int dimension, const std::vector<int>& repeats, const NDArray<T>& arr) {

    int rank = arr.rankOf();

    if (dimension < 0)
        dimension += rank;

    std::vector<int> reps;

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
    
    std::vector<int> outShape(rank);
    for (int i = 0; i < rank; i++)         
        outShape[i] = arr.sizeAt(i) * reps.at(i);        

    return outShape;
}


//////////////////////////////////////////////////////////////////////////
// evaluate shapeInfo of permuted array
    template<typename T>
    int* ShapeUtils<T>::evalPermShapeInfo(const int* dimensions, const int rank, const NDArray<T>& arr) {

    if (!arr.nonNull() || rank != arr.rankOf())
        throw "ShapeUtils<T>::evalPermShapeInfo static method: wrong arguments in permute method: either array is nullptr or rank is not suitable!";
    
    int shapeInfoLength = rank*2 + 4;
    // allocate memory for new array - shapeInfo

    int* shapeInfoNew = nullptr;
    ALLOCATE(shapeInfoNew, arr.getWorkspace(), shapeInfoLength, int);
    // copy arr _shapeInfo into new array       
    memcpy(shapeInfoNew, arr.getShapeInfo(), shapeInfoLength*sizeof(int));  
    // perform buffer permutation   
    shape::doPermuteShapeBuffer(rank, shapeInfoNew, const_cast<int*>(dimensions));      

    return shapeInfoNew;

}

//////////////////////////////////////////////////////////////////////////
// evaluate shapeInfo of transposed array
    template<typename T>
    int* ShapeUtils<T>::evalTranspShapeInfo(const NDArray<T>& arr) {

        int rank = arr.rankOf();
        std::vector<int> dimensions(rank);
        for (int i = 0; i < rank; ++i)
            dimensions[i] = rank - 1 - i;

        int* shapeInfoNew = evalPermShapeInfo(dimensions.data(), dimensions.size(), arr);

        return shapeInfoNew;
    }

//////////////////////////////////////////////////////////////////////////
    template<typename T>
    bool ShapeUtils<T>::insertDimension(int rank, int *shape, int axis, int dimension) {
        if (axis >= rank || axis <= -rank)
            return false;

        if (axis < 0)
            axis = rank + axis;

        std::vector<int> tmp;
        for (int e = 0; e < rank; e++) {
            if (shape[e] != 1)
                tmp.emplace_back(shape[e]);
        }

        tmp.insert(tmp.begin() + axis, dimension);
        memcpy(shape, tmp.data(), tmp.size() * sizeof(int));

        return true;
    }

//////////////////////////////////////////////////////////////////////////
    template<typename T>
    bool ShapeUtils<T>::copyVectorPart(std::vector<int>& target, std::vector<int>& source, int rank, int offset) {
        if (source.size() < offset + rank)
            return false;

        for (int e = offset; e < offset + rank; e++)
            target.push_back(source[e]);

        return true;
    }


//////////////////////////////////////////////////////////////////////////
// return new (shorter) sorted dimensions array without dimensions that are present in input vector
    template<typename T>
    std::vector<int> ShapeUtils<T>::evalDimsToExclude(const int rank, const std::vector<int>& dimensions) {

    std::vector<int> newDimensions;
    int size = dimensions.size();
    if(size == 0) {                          // if input vector is empty then return whole shape range
        newDimensions.resize(rank);
        std::iota(newDimensions.begin(), newDimensions.end(), 0);   // fill with 0, 1, ... rank-1
    }
    else {
        bool isAbsent;
        for(int i=0; i<rank; ++i) {
            isAbsent = true;
            for(int j=0; j<size; ++j) {
                if(i == dimensions[j]) {
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
template <typename T>
bool ShapeUtils<T>::areShapesBroadcastable(const NDArray<T> &arr1, const NDArray<T> &arr2)
{

    int minRank = arr1.rankOf() < arr2.rankOf() ? arr1.rankOf() : arr2.rankOf();    
       
    for (int i = -1; i >= -minRank; --i)        
        if (arr1.sizeAt(i) != arr2.sizeAt(i) && arr1.sizeAt(i) != 1 && arr2.sizeAt(i) != 1)
            return false;
    
    return true;
}


//////////////////////////////////////////////////////////////////////////
// check the possibility of broadcast operation, if true then return shapeInfo of resulting array
// if evalMinMax == false the array with larger rank has to be passed as first argument
template <typename T>
bool ShapeUtils<T>::evalBroadcastShapeInfo(const NDArray<T> &max, const NDArray<T> &min, const bool evalMinMax, int*& resultShapeInfo)
{
    
    // check whether broadcast operation is possible for input arrays
    if(!areShapesBroadcastable(max, min))
        return false;

    int* maxShapeInfo = max.getShapeInfo(); 
    int* minShapeInfo = min.getShapeInfo();

    if(evalMinMax && (max.rankOf() < min.rankOf())) {
        maxShapeInfo = min.getShapeInfo(); 
        minShapeInfo = max.getShapeInfo();
    }
       
    const int  maxRank      = maxShapeInfo[0];
    const int  minRank      = minShapeInfo[0];  
    
    // evaluate shapeInfo for resulting array
    if(resultShapeInfo != nullptr)
        throw "ShapeUtils::evalBroadcastShapeInfo method: the input pointer on shapeInfo must be empty (=nullptr) !" ;
    
    ALLOCATE(resultShapeInfo, max.getWorkspace(), shape::shapeInfoLength(maxRank), int);
    memcpy(resultShapeInfo, maxShapeInfo, shape::shapeInfoLength(maxRank) * sizeof(int));
    for (int i = 0; i < minRank; ++i)
        if(maxShapeInfo[maxRank-i] < minShapeInfo[minRank-i])
            resultShapeInfo[maxRank - i] = minShapeInfo[minRank-i];

    shape::updateStrides(resultShapeInfo, max.ordering());

    return true;
}

//////////////////////////////////////////////////////////////////////////
// check the possibility of broadcast operation for set of arrays, if true then return resulting broadcasted shapeInfo
template <typename T>
bool ShapeUtils<T>::evalCommonBroadcastShapeInfo(const std::vector<const NDArray<T>*>& arrays, int*& resultShapeInfo, memory::Workspace* workspace) {

    if(resultShapeInfo != nullptr)
        throw "ShapeUtils::evalCommonBroadcastShapeInfo method: the input pointer on shapeInfo must be empty (=nullptr) !" ;

    int size = arrays.size();
    int maxRank = arrays[size - 1]->rankOf();

    for(int i = 0; i < size - 1; ++i) {
        if(arrays[i]->rankOf() > maxRank)
            maxRank = arrays[i]->rankOf();
        for(int j = i + 1; j < size; ++j)
            if(!areShapesBroadcastable(*arrays[i], *arrays[j]))
                return false;
    }

    ALLOCATE(resultShapeInfo, workspace, shape::shapeInfoLength(maxRank), int);
    memset(resultShapeInfo, 0, shape::shapeInfoLength(maxRank) * sizeof(int));
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
template <typename T>
std::vector<int> ShapeUtils<T>::getDimsWithSameShape(const NDArray<T>& max, const NDArray<T>& min) {

    std::vector<int> result;
    int* maxShapeInfo = max.getShapeInfo(); 
    int* minShapeInfo = min.getShapeInfo();
    int  maxRank      = maxShapeInfo[0];
    int  minRank      = minShapeInfo[0];

    for(int i = 1; i <= minRank; ++i)
        if(minShapeInfo[i] == maxShapeInfo[maxRank - minRank + i])
            result.emplace_back(maxRank - minRank + i - 1);

    return result;
}


//////////////////////////////////////////////////////////////////////////
// return absolute index of array min, min is sub-array of max, index to be returned is min index and it corresponds maxIdx of max array 
template <typename T>
int ShapeUtils<T>::getSubArrayIndex(const int* maxShapeInfo, const int* minShapeInfo, const int maxIdx) {
    // check shape consistence 
    if(maxShapeInfo[0] < minShapeInfo[0])
        throw "ShapeUtils::getSubArrayIndex: rank of max-array must be greater or equal to min-array rank !";
    
    for(int i = 0; i < minShapeInfo[0]; ++i)
        // if((maxShapeInfo[maxShapeInfo[0] - i] < minShapeInfo[minShapeInfo[0] - i]) || (maxShapeInfo[maxShapeInfo[0] - i] % minShapeInfo[minShapeInfo[0] - i] != 0) )        
        if(maxShapeInfo[maxShapeInfo[0] - i] < minShapeInfo[minShapeInfo[0] - i])        
            throw "ShapeUtils::getSubArrayIndex: some of dimension shape of max-array is smaller than those of min-array or the max shape is not multiple of min shape !";

    return shape::subArrayIndex(maxShapeInfo, minShapeInfo, maxIdx);
}

//////////////////////////////////////////////////////////////////////////
// evaluate shapeInfo for resulting array from tile operation
template <typename T>
int* ShapeUtils<T>::evalTileShapeInfo(const NDArray<T>& arr, const std::vector<int>& reps) {

    // check whether reps contains at least one zero (then throw exception) or whether all elements in reps are unities (then simply reshape or do nothing)
    int dim = reps.size();  
    int product = 1;
    for(const auto& item : reps)
        product *= item;
    if(product == 0)
        throw "NDArray::tile method: one of the elements in reps array is zero !";

    int rankOld = arr.rankOf();
    int diff = rankOld - dim;
    
    // evaluate new shapeInfo
    int* newShapeInfo = nullptr;    
    if(diff < 0) {      
        ALLOCATE(newShapeInfo, arr.getWorkspace(), dim*2 + 4, int);
        newShapeInfo[0] = dim;                  // set new rank
        for(int i=1; i <= -diff; ++i)
            newShapeInfo[i] = 1;                // set unities to be new dimensions at left-hand side of newShapeInfo shape place
        memcpy(newShapeInfo + 1 - diff, arr.getShapeInfo() + 1, rankOld*sizeof(int));       // copy old dimensions to the right-hand side of newShapeInfo shape place
        for(int i=1; i <= dim; ++i)
            newShapeInfo[i] *= reps[i - 1];     // set new shape by multiplying old dimensions by corresponding numbers from reps 
    }
    else {      
        ALLOCATE(newShapeInfo, arr.getWorkspace(), rankOld*2 + 4, int);
        memcpy(newShapeInfo, arr.getShapeInfo(), (rankOld*2 + 4)*sizeof(int));      // copy all elements of _shapeInfo to newShapeInfo
        for(int i=1; i <= dim; ++i)
            newShapeInfo[rankOld + 1 - i] *= reps[dim - i];     // set new shape by multiplying old dimensions by corresponding numbers from reps 
    }
    shape::updateStrides(newShapeInfo, arr.ordering());
    
    return newShapeInfo;
}

//////////////////////////////////////////////////////////////////////////
    template<typename T>
    std::vector<int> ShapeUtils<T>::convertAxisToTadTarget(int rank, std::initializer_list<int> axis) {
        std::vector<int> newAxis(axis);
        return convertAxisToTadTarget(rank, newAxis);
    }

//////////////////////////////////////////////////////////////////////////
    template<typename T>
    std::vector<int> ShapeUtils<T>::convertAxisToTadTarget(int rank, std::vector<int>& axis) {
        std::vector<int> newAxis;
        for (int e = 0; e < rank; e++) {
            if (std::find(axis.begin(), axis.end(), e) == axis.end())
                newAxis.emplace_back(e);
        }

        return newAxis;
    }

    template<typename T>
    std::vector<int> ShapeUtils<T>::pullShapeFromShapeInfo(int *shapeInfo) {
        std::vector<int> shape((int) shape::rank(shapeInfo));

        for (int e = 0; e < shape.size(); e++)
            shape[e] = shape::shapeOf(shapeInfo)[e];

        return shape;
    }

    template<typename T>
    std::string ShapeUtils<T>::shapeAsString(NDArray<T> &array) {
        auto vec = array.getShapeAsVector();
        return shapeAsString(vec);
    }
    template<typename T>
    std::string ShapeUtils<T>::shapeAsString(std::vector<int>& shape) {
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


//////////////////////////////////////////////////////////////////////////
// evaluate shapeInfo for diagonal array which is made using input arr elements as diagonal
template<typename T>
int* ShapeUtils<T>::evalDiagShapeInfo(const NDArray<T>& arr){    

    const int rank = arr.rankOf();

    int* outputShapeInfo = nullptr;

    if(arr.isVector() || arr.isScalar()) {
        ALLOCATE(outputShapeInfo, arr.getWorkspace(), shape::shapeInfoLength(rank), int);
        outputShapeInfo[0] = rank;
        outputShapeInfo[1] = outputShapeInfo[2] = arr.lengthOf();
    }
    else {
        ALLOCATE(outputShapeInfo, arr.getWorkspace(), shape::shapeInfoLength(2*rank), int);
        outputShapeInfo[0] = 2*rank;
        for(int i = 0; i < rank; ++i)
            outputShapeInfo[i + 1] = outputShapeInfo[i + 1 + rank] = arr.sizeAt(i);
    }
        
    shape::updateStrides(outputShapeInfo, arr.ordering());

    return outputShapeInfo;
}

template class ND4J_EXPORT ShapeUtils<float>;
template class ND4J_EXPORT ShapeUtils<float16>;
template class ND4J_EXPORT ShapeUtils<double>;


}

