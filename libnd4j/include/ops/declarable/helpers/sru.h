//
// @author Yurii Shyrma (iuriish@yahoo.com), created by on 06.04.2018
//

#ifndef LIBND4J_SRU_H
#define LIBND4J_SRU_H

#include <ops/declarable/helpers/helpers.h>

namespace nd4j    {
namespace ops     {
namespace helpers {


	template <typename T>
	void sruCell(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs);

	template <typename T>
	void sruTimeLoop(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs);
	
    

}
}
}


#endif //LIBND4J_SRU_H
