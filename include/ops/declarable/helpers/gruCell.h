//
// Created by Yurii Shyrma on 15.02.2018
//

#ifndef LIBND4J_LSTMCELL_H
#define LIBND4J_LSTMCELL_H

#include <ops/declarable/helpers/helpers.h>

namespace nd4j {
namespace ops {
namespace helpers {


	template <typename T>
	void gruCell(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* outArrs);
	
    

}
}
}


#endif //LIBND4J_LSTMCELL_H
