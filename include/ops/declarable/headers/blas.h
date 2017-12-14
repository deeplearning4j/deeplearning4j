//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        
        /**
         * This op is general matmum implementation. Depending on inputs dimensionality output result might be different.
         * matrix x matrix = BLAS gemm
         * vector x matrix = BLAS gemm
         * vector x vector = BLAS dot
         * vector x scalar = element-wise mul
         * scalar x vector = element-wise mul
         */
        DECLARE_CUSTOM_OP(matmul, 2, 1, false, -2, 0);

        /**
         * tensorMmul/tensorDot operation
         * takes 2 ndarrays, and 2 sets of axes
         *
         * Integer argumens map:
         * IArgs[0] - number of axes along for first array
         * IArgs[1]... axes values for first array
         * IArgs[] - number of axes along for second array
         * IArgs[1]... axes values for second array
         */
        DECLARE_CUSTOM_OP(tensormmul, 2, 1, false, 0, -1);   

        /**
         * This op is simple implementation of BLAS AXPY method.
         * Math is: y += a * x;
         * 
         */
        DECLARE_CONFIGURABLE_OP(axpy, 2, 1, false, -2, 0);
    }
}