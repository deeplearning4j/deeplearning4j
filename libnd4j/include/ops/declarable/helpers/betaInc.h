//
// Created by Yurii Shyrma on 11.12.2017
//

#ifndef LIBND4J_BETAINC_H
#define LIBND4J_BETAINC_H

#include <ops/declarable/helpers/helpers.h>
#include "NDArray.h"

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    NDArray<T> betaInc(const NDArray<T>& a, const NDArray<T>& b, const NDArray<T>& x);
    

}
}
}


#endif //LIBND4J_BETAINC_H