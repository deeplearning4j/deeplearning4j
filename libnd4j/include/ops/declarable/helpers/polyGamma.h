//
// Created by Yurii Shyrma on 13.12.2017.
//

#ifndef LIBND4J_POLYGAMMA_H
#define LIBND4J_POLYGAMMA_H

#include <ops/declarable/helpers/helpers.h>
#include "NDArray.h"

namespace nd4j {
namespace ops {
namespace helpers {


	// calculate the polygamma function
    template <typename T>
    NDArray<T> polyGamma(const NDArray<T>& n, const NDArray<T>& x);
    

}
}
}


#endif //LIBND4J_POLYGAMMA_H
