//
//  @author raver119@gmail.com
//

#ifndef LIBND4J_HEADERS_TPARTY_H
#define LIBND4J_HEADERS_TPARTY_H

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        #if NOT_EXCLUDED(OP_firas_sparse)
        DECLARE_CUSTOM_OP(firas_sparse, 1, 1, false, 0, -1);
        #endif
    }
}

#endif