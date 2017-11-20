//
// @author iuriish@yahoo.com
//

#include <algorithm>
#include <helpers/ShapeUtils.h>
#include <climits>
#include <numeric>
#include <set>


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
    int* ShapeUtils<T>::evalReduceShapeInfo(const char order, std::vector<int>& dimensions, const NDArray<T>& arr) {
        return evalReduceShapeInfo(order, dimensions, arr.getShapeInfo(), arr.getWorkspace());
    }

//////////////////////////////////////////////////////////////////////////
// evaluate resulting shape after reduce operation
template<typename T>
int* ShapeUtils<T>::evalReduceShapeInfo(const char order, std::vector<int>& dimensions, const int *shape , nd4j::memory::Workspace* workspace) {
        
    int rank = shape::rank(const_cast<int*>(shape));
    shape::checkDimensions(rank, dimensions);
       
	int* newShape = nullptr;
    int dimSize = dimensions.size();
	int newRank = rank - dimSize;
	if (newRank==0 || (dimSize==1 && dimensions[0]==INT_MAX)) { 			// check whether given dimension is meant for the whole dimension
		ALLOCATE(newShape, workspace, 8, int);						// set newRank = 2
		newShape[0] = 2;
		newShape[1] = 1;
		newShape[2] = 1;			
	}
       else {
		ALLOCATE(newShape, workspace, shape::shapeInfoLength(2), int);
		int* tempShape = shape::removeIndex(shape::shapeOf(const_cast<int*>(shape)), const_cast<int*>(dimensions.data()), rank, dimSize);
           newShape[0] = newRank;                      // set rank
		for(int i=0; i<newRank; ++i)
			newShape[i+1] = tempShape[i]; 			// ignore zero index (rank)
		delete []tempShape;
	}		
	//ensure vector is proper shape 
	if (newRank == 1) {
		int oldValue = newShape[1];
		RELEASE(newShape, workspace);
		ALLOCATE(newShape, workspace, shape::shapeInfoLength(2), int);		// set newRank = 2
		newShape[0] = 2;
        if (dimensions[0] == 0) {
               newShape[1] = 1; 
			newShape[2] = oldValue;
		}
           else {
               newShape[1] = oldValue;
			newShape[2] = 1; 				
		}
    } 
	shape::updateStrides(newShape, order);
       
    return newShape;
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
// check the possibility of broadcast operation, if true then return shapeInfo of resulting array
// the array with larger dimensions number has to be passed as first argument
template <typename T>
int* ShapeUtils<T>::evalBroadcastShapeInfo(const NDArray<T> &max, const NDArray<T> &min)
{

    int* maxShapeInfo = max.getShapeInfo(); 
    int* minShapeInfo = min.getShapeInfo();
    int  maxRank      = maxShapeInfo[0];
    int  minRank      = minShapeInfo[0];

    // check whether broadcast operation is possible for input arrays
    for (int i = 0; i < minRank; ++i)
        if (maxShapeInfo[maxRank - i] != minShapeInfo[minRank - i] && maxShapeInfo[maxRank - i] != 1 && minShapeInfo[minRank - i] != 1)
            throw "ShapeUtils::evalBroadcastShapeInfo method: the shapes of input arrays are not compatible for broadcast operation !" ;
    
    // evaluate shapeInfo for resulting array
    int *shapeInfoNew = nullptr;
    ALLOCATE(shapeInfoNew, max.getWorkspace(), shape::shapeInfoLength(maxRank), int);
    memcpy(shapeInfoNew, maxShapeInfo, shape::shapeInfoLength(maxRank) * sizeof(int));
    for (int i = 0; i < minRank; ++i)
        shapeInfoNew[maxRank - i] = maxShapeInfo[maxRank-i] > minShapeInfo[minRank-i] ? maxShapeInfo[maxRank-i] : minShapeInfo[minRank-i];

    shape::updateStrides(shapeInfoNew, max.ordering());

    return shapeInfoNew;
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
        throw "ShapeUtils::getSubArrayIndex: rank of max-array must greater or equal to min-array rank !";
    bool isConsistent = true;
    for(int i = 0; i < minShapeInfo[0]; ++i)
        if(maxShapeInfo[maxShapeInfo[0] - i] < minShapeInfo[minShapeInfo[0] - i])
            isConsistent = false;
    if(!isConsistent)
        throw "ShapeUtils::getSubArrayIndex: some of dimension shape of max-array is smaller than those of min-array or the max shape is not multiple of min shape !";

    return shape::subArrayIndex(maxShapeInfo, minShapeInfo, maxIdx);
}

//////////////////////////////////////////////////////////////////////////
// evaluate shapeInfo for resulting array of tile operation
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




template class ND4J_EXPORT ShapeUtils<float>;
template class ND4J_EXPORT ShapeUtils<float16>;
template class ND4J_EXPORT ShapeUtils<double>;
}

