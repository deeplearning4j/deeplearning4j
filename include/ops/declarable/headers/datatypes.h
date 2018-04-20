//
//  @author raver119@gmail.com
//
#ifndef LIBND4J_HEADERS_DTYPE_H
#define LIBND4J_HEADERS_DTYPE_H

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        /**
         * This operation casts elements of input array to double data type
         * 
         * PLEASE NOTE: This op is disabled atm, and reserved for future releases.
         */
        #if NOT_EXCLUDED(OP_to_double)
        DECLARE_OP(to_double, 1, 1, true);
        #endif

        /**
         * This operation casts elements of input array to float16 data type
         * 
         * PLEASE NOTE: This op is disabled atm, and reserved for future releases.
         */
        #if NOT_EXCLUDED(OP_to_float16)
        DECLARE_OP(to_float16, 1, 1, true);
        #endif

        /**
         * This operation casts elements of input array to float data type
         * 
         * PLEASE NOTE: This op is disabled atm, and reserved for future releases.
         */
        #if NOT_EXCLUDED(OP_to_float32)
        DECLARE_OP(to_float32, 1, 1, true);
        #endif

        /**
         * This operation casts elements of input array to int32 data type
         * 
         * PLEASE NOTE: This op is disabled atm, and reserved for future releases.
         */
        #if NOT_EXCLUDED(OP_to_int32)
        DECLARE_OP(to_int32, 1, 1, true);
        #endif

        /**
         * This operation casts elements of input array to int64 (aka long long) data type
         * 
         * PLEASE NOTE: This op is disabled atm, and reserved for future releases.
         */
        #if NOT_EXCLUDED(OP_to_int64)
        DECLARE_OP(to_int64, 1, 1, true);
        #endif

        /**
         * This operation casts elements of input array to unsinged int32 data type
         * 
         * PLEASE NOTE: This op is disabled atm, and reserved for future releases.
         */
        #if NOT_EXCLUDED(OP_to_uint32)
        DECLARE_OP(to_uint32, 1, 1, true);
        #endif

        /**
         * This operation casts elements of input array to unsigned int64 (aka unsigned long long) data type
         * 
         * PLEASE NOTE: This op is disabled atm, and reserved for future releases.
         */
        #if NOT_EXCLUDED(OP_to_uint64)
        DECLARE_OP(to_uint64, 1, 1, true);
        #endif

        /**
         * This operation casts elements of input array to specified data type
         * 
         * PLEASE NOTE: This op is disabled atm, and reserved for future releases.
         * 
         * 
         * Int args:
         * 0: target DataType
         */
        #if NOT_EXCLUDED(OP_cast)
        DECLARE_CUSTOM_OP(cast, 1, 1, false, 0, 1);
        #endif
    }
}

#endif