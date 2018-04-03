//
// @author Yurii Shyrma, created on 27.03.2017
//

#ifndef LIBND4J_RNNCELL_H
#define LIBND4J_RNNCELL_H

#include <ops/declarable/helpers/helpers.h>

namespace nd4j    {
namespace ops     {
namespace helpers {


	template <typename T>
	void rnnCell(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* ht);
	
    

}
}
}


#endif //LIBND4J_RNNCELL_H
