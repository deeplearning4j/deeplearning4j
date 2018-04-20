//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 19.04.2018
//

#ifndef LIBND4J_ACTIVATIONS_H
#define LIBND4J_ACTIVATIONS_H

#include <ops/declarable/helpers/helpers.h>

namespace nd4j {
namespace ops {
namespace helpers {

	template <typename T>
	void softMaxForVector(const NDArray<T>& input, NDArray<T>& output);

	template <typename T>
	void logSoftMaxForVector(const NDArray<T>& input, NDArray<T>& output);

	template <typename T>
	void softmax(const NDArray<T>& input, NDArray<T>& output, const int dimension);
	    

}
}
}


#endif //LIBND4J_ACTIVATIONS_H
