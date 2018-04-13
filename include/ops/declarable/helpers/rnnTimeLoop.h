//
// @author Yurii Shyrma, created on 04.03.2017
//

#ifndef LIBND4J_RNNTIMELOOP_H
#define LIBND4J_RNNTIMELOOP_H

#include <ops/declarable/helpers/helpers.h>

namespace nd4j    {
namespace ops     {
namespace helpers {


	template <typename T>
	void rnnTimeLoop(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* h, NDArray<T>* hFinal);
	
    

}
}
}


#endif //LIBND4J_RNNTIMELOOP_H
