//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 17.05.2018
//

#ifndef LIBND4J_PERCENTILE_H
#define LIBND4J_PERCENTILE_H

#include <ops/declarable/helpers/helpers.h>
#include "NDArray.h"

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void percentile(const NDArray<T>& input, NDArray<T>& output, std::vector<int>& axises, const T q, const int interpolation);
    

}
}
}


#endif //LIBND4J_PERCENTILE_H