//
// @author Yurii Shyrma (iuriish@yahoo.com), created by on 18.04.2018
//

#ifndef LIBND4J_SRU_H
#define LIBND4J_SRU_H

#include <ops/declarable/helpers/helpers.h>

namespace nd4j    {
namespace ops     {
namespace helpers {


	template <typename T>
	void meshgrid(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs, const bool swapFirst2Dims);

	
    

}
}
}


#endif //LIBND4J_SRU_H
