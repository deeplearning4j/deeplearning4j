//
// @author Yurii Shyrma, created on 16.04.2018
//

#ifndef LIBND4J_REVERSE_H
#define LIBND4J_REVERSE_H

#include <ops/declarable/helpers/helpers.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

	template <typename T>
	void reverseArray(T* inArr, Nd4jLong *inShapeBuffer, T *result, Nd4jLong *zShapeBuffer, int numOfElemsToReverse = 0);

	
	template <typename T>
	void reverseSequence(const NDArray<T>* input, const NDArray<T>* seqLengths, NDArray<T>* output, int seqDim, const int batchDim);


	template<typename T>
	void reverse(const NDArray<T>* input, NDArray<T>* output, const std::vector<int>* intArgs);

    

}
}
}


#endif //LIBND4J_REVERSESEQUENCE_H
