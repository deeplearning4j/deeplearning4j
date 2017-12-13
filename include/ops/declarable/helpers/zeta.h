//
// Created by Yurii Shyrma on 12.12.2017.
//

#ifndef LIBND4J_ZETA_H
#define LIBND4J_ZETA_H

#include <ops/declarable/helpers/helpers.h>
#include "NDArray.h"

namespace nd4j {
namespace ops {
namespace helpers {


	// calculate the Hurwitz zeta function
    template <typename T>
    NDArray<T> zeta(const NDArray<T>& x, const NDArray<T>& q);
    

}
}
}


#endif //LIBND4J_ZETA_H
