//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        DECLARE_CONFIGURABLE_OP(clipbyvalue, 1, 1, true, 2, 0);
        DECLARE_CONFIGURABLE_OP(clipbynorm, 1, 1, true, 1, 0);
        DECLARE_CONFIGURABLE_OP(clipbyavgnorm, 1, 1, true, 1, 0);
        DECLARE_CONFIGURABLE_OP(cumsum, 1, 1, true, 0, -2);
        DECLARE_CONFIGURABLE_OP(cumprod, 1, 1, true, 0, -2);
        DECLARE_CUSTOM_OP(tile, 1, 1, false, 0, -2);
        DECLARE_CUSTOM_OP(repeat, 1, 1, true, 0, -1); 
        DECLARE_CONFIGURABLE_OP(invert_permutation, 1, 1, false, 0, 0);  

        DECLARE_CUSTOM_OP(concat, -1, 1, false, 0, -2);
        DECLARE_CUSTOM_OP(concat_bp, -1, -1, false, 0, 1);

        DECLARE_OP(mergemax, -1, 1, false);
        DECLARE_OP(mergemaxindex, -1, 1, false);
        DECLARE_OP(mergeadd, -1, 1, false);
        DECLARE_OP(mergeavg, -1, 1, false);   

        DECLARE_CONFIGURABLE_OP(scatter_update, 2, 1, true, 0, -1); 

        DECLARE_OP(Floor, 1, 1, true);

        DECLARE_OP(Log1p, 2, 1, true);

        DECLARE_CONFIGURABLE_OP(reverse, 1, 1, true, 0, -2);

        DECLARE_CUSTOM_OP(gather, 1, 1, false, 0, 1);

        DECLARE_CUSTOM_OP(pad, 2, 1, false, 0, 1);

        /**
         * creates identity 2D matrix or batch of identical 2D identity matrices
         * 
         * Input array:
         * provide some array - in any case operation simply neglects it
         * 
         * Input integer arguments:
         * IArgs[0]       - order of output identity matrix, 99 -> 'c'-order, 102 -> 'f'-order
         * IArgs[1]       - the number of rows in output inner-most 2D identity matrix
         * IArgs[2]       - optional, the number of columns in output inner-most 2D identity matrix, if this argument is not provided then it is taken to be equal to number of rows
         * IArgs[3,4,...] - optional, shape of batch, output matrix will have leading batch dimensions of this shape         
         */
        DECLARE_CUSTOM_OP(eye, 1, 1, false, 0, 2);

        DECLARE_CUSTOM_OP(gather_nd, 2, 1, false, 0, 0);

        DECLARE_CUSTOM_OP(reverse_sequense, 2, 1, false, 0, 2);

        DECLARE_CUSTOM_OP(trace, 1, 1, false, 0, 0);

        DECLARE_OP(random_shuffle, 1, 1, true);

        /**
         * clip a list of given tensors with given average norm when needed
         * 
         * Input:
         *    a list of tensors (at least one)
         * 
         * Input floating point argument:
         *    clip_norm - a value that used as threshold value and norm to be used
         *
         * return a list of clipped tensors
         *  and global_norm as scalar tensor at the end
         */
        DECLARE_CUSTOM_OP(clip_by_global_norm, 1, 2, true, 1, 0);

    }
}