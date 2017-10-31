//
// @author iuriish@yahoo.com
//

#include <helpers/ShapeUtils.h>
#include <climits>


namespace nd4j {
     
//////////////////////////////////////////////////////////////////////////
// evaluate shape for array resulting from tensorDot operation, also evaluate shapes and permutation dimensions for transposition of two input arrays 
template<typename T>
std::vector<int> ShapeUtils<T>::evalShapeForTensorDot(const nd4j::NDArray<T>* a, const nd4j::NDArray<T>* b, std::vector<int>& axes_0, std::vector<int>& axes_1, std::vector<int>& permutAt, std::vector<int>& permutBt, std::vector<int>& shapeAt, std::vector<int>& shapeBt) {

    int axe0_size = (int) axes_0.size();
    int axe1_size = (int) axes_1.size();                 
    // validating axes
    int validationLength = nd4j::math::nd4j_min<int>(axe0_size, axe1_size);
    for (int i = 0; i < validationLength; i++) {
        if (a->sizeAt(axes_0[i]) != b->sizeAt(axes_1[i]))
            throw "Size of the given axes at each dimension must be the same size.";
        if (axes_0[i] < 0)
            axes_0[i] += a->rankOf();
        if (axes_1[i] < 0)
            axes_1[i] += b->rankOf();
    }
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
    int aLength = nd4j::math::nd4j_min<int>(a->rankOf(), axes_0.size());
    for (int i = 0; i < aLength; i++)
        n2 *= a->sizeAt(axes_0[i]);
    shapeAt = {-1, n2};
    std::vector<int> oldShapeA;
    if (list_A.size() == 0) {
        oldShapeA.emplace_back(1);
    } else {
        oldShapeA.insert(oldShapeA.end(), list_A.begin(), list_A.end());            
        for (int i = 0; i < (int) oldShapeA.size(); i++)
            oldShapeA[i] = a->sizeAt(oldShapeA[i]);
    }
    int n3 = 1;
    int bNax = nd4j::math::nd4j_min<int>(b->rankOf(), axes_1.size());
    for (int i = 0; i < bNax; i++)
        n3 *= b->sizeAt(axes_1[i]);
    shapeBt = {n3, -1};
    std::vector<int> oldShapeB;
    if (list_B.size() == 0) {
        oldShapeB.emplace_back(1);
    } else {
        oldShapeB.insert(oldShapeB.end(), list_B.begin(), list_B.end());            
        for (int i = 0; i < (int) oldShapeB.size(); i++)
            oldShapeB[i] = b->sizeAt(oldShapeB[i]);
    }
    std::vector<int> aPlusB(oldShapeA);
    aPlusB.insert(aPlusB.end(), oldShapeB.begin(), oldShapeB.end());            
    return aPlusB;
}
    
 

//////////////////////////////////////////////////////////////////////////
// evaluate resulting shape after reduce operation
template<typename T>
int* ShapeUtils<T>::evalReduceShapeInfo(const char order, std::vector<int>& dimensions, const NDArray<T>& arr) {
        
    int rank = arr.rankOf();
    shape::checkDimensions(rank, dimensions);
       
	int* newShape = nullptr;
    int dimSize = dimensions.size();
	int newRank = rank - dimSize;
	if (newRank==0 || (dimSize==1 && dimensions[0]==INT_MAX)) { 			// check whether given dimension is meant for the whole dimension
		ALLOCATE(newShape, arr.getWorkspace(), 8, int);						// set newRank = 2
		newShape[0] = 2;
		newShape[1] = 1;
		newShape[2] = 1;			
	}
       else {
		ALLOCATE(newShape, arr.getWorkspace(), shape::shapeInfoLength(2), int);
		int* tempShape = shape::removeIndex(arr.shapeOf(), const_cast<int*>(dimensions.data()), rank, dimSize);
           newShape[0] = newRank;                      // set rank
		for(int i=0; i<newRank; ++i)
			newShape[i+1] = tempShape[i]; 			// ignore zero index (rank)
		delete []tempShape;
	}		
	//ensure vector is proper shape 
	if (newRank == 1) {
		int oldValue = newShape[1];
		RELEASE(newShape, arr.getWorkspace());
		ALLOCATE(newShape, arr.getWorkspace(), shape::shapeInfoLength(2), int);		// set newRank = 2
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

template class ND4J_EXPORT ShapeUtils<float>;
template class ND4J_EXPORT ShapeUtils<float16>;
template class ND4J_EXPORT ShapeUtils<double>;




}
