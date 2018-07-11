//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 15.02.2018
//

#ifndef LIBND4J_GRU_H
#define LIBND4J_GRU_H

#include <ops/declarable/helpers/helpers.h>

namespace nd4j {
namespace ops {
namespace helpers {


	template <typename T>
	void gruCell(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* h);

	template <typename T>
	void gruTimeLoop(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* h);
	
	template <typename T>
	void gruCellBP(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs);    

}
}
}


#endif //LIBND4J_GRU_H
