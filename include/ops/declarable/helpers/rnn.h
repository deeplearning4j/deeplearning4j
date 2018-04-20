//
// @author Yurii Shyrma, created on 16.04.2017
//

#ifndef LIBND4J_RNN_H
#define LIBND4J_RNN_H

#include <ops/declarable/helpers/helpers.h>

namespace nd4j    {
namespace ops     {
namespace helpers {


	template <typename T>
	void rnnCell(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* ht);

	template <typename T>
	void rnnTimeLoop(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* h, NDArray<T>* hFinal);	    

}
}
}


#endif //LIBND4J_RNN_H
