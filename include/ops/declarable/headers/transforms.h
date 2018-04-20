//
//  @author raver119@gmail.com
//

#ifndef LIBND4J_HEADERS_TRANSFORMS_H
#define LIBND4J_HEADERS_TRANSFORMS_H

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        #if NOT_EXCLUDED(OP_clipbyvalue)
        DECLARE_CONFIGURABLE_OP(clipbyvalue, 1, 1, true, 2, 0);
        #endif

        #if NOT_EXCLUDED(OP_clipbynorm)
        DECLARE_CONFIGURABLE_OP(clipbynorm, 1, 1, true, 1, 0);
        #endif

        #if NOT_EXCLUDED(OP_clipbyavgnorm)
        DECLARE_CONFIGURABLE_OP(clipbyavgnorm, 1, 1, true, 1, 0);
        #endif

        #if NOT_EXCLUDED(OP_cumsum)
        DECLARE_CONFIGURABLE_OP(cumsum, 1, 1, true, 0, -2);
        #endif

        #if NOT_EXCLUDED(OP_cumprod)
        DECLARE_CONFIGURABLE_OP(cumprod, 1, 1, true, 0, -2);
        #endif

        #if NOT_EXCLUDED(OP_tile)
        DECLARE_CUSTOM_OP(tile, 1, 1, false, 0, -2);
        #endif

        #if NOT_EXCLUDED(OP_repeat)
        DECLARE_CUSTOM_OP(repeat, 1, 1, true, 0, -1);
        #endif

        #if NOT_EXCLUDED(OP_invert_permutation)
        DECLARE_CONFIGURABLE_OP(invert_permutation, 1, 1, false, 0, 0);
        #endif

        DECLARE_CUSTOM_OP(concat, -1, 1, false, 0, -2);
        DECLARE_CUSTOM_OP(concat_bp, -1, -1, false, 0, 1);

        #if NOT_EXCLUDED(OP_mergemax)
        DECLARE_OP(mergemax, -1, 1, false);
        #endif

        #if NOT_EXCLUDED(OP_mergemaxindex)
        DECLARE_OP(mergemaxindex, -1, 1, false);
        #endif

        #if NOT_EXCLUDED(OP_mergeadd)
        DECLARE_OP(mergeadd, -1, 1, false);
        #endif

        #if NOT_EXCLUDED(OP_mergeavg)
        DECLARE_OP(mergeavg, -1, 1, false);
        #endif

        #if NOT_EXCLUDED(OP_scatter_update)
        DECLARE_CONFIGURABLE_OP(scatter_update, 2, 1, true, 0, -1);
        #endif

        #if NOT_EXCLUDED(OP_Floor)
        DECLARE_OP(Floor, 1, 1, true);
        #endif

        #if NOT_EXCLUDED(OP_Log1p)
        DECLARE_OP(Log1p, 2, 1, true);
        #endif

        #if NOT_EXCLUDED(OP_reverse)
        DECLARE_CONFIGURABLE_OP(reverse, 1, 1, true, 0, -2);
        #endif

        #if NOT_EXCLUDED(OP_gather)
        DECLARE_CUSTOM_OP(gather, 1, 1, false, 0, -2);
        #endif

        #if NOT_EXCLUDED(OP_pad)
        DECLARE_CUSTOM_OP(pad, 2, 1, false, 0, 1);
        #endif

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
        #if NOT_EXCLUDED(OP_eye)
        DECLARE_CUSTOM_OP(eye, 1, 1, false, 0, 2);
        #endif

        #if NOT_EXCLUDED(OP_gather_nd)
        DECLARE_CUSTOM_OP(gather_nd, 2, 1, false, 0, 0);
        #endif

        #if NOT_EXCLUDED(OP_reverse_sequence)
        DECLARE_CUSTOM_OP(reverse_sequence, 2, 1, false, 0, 2);
        #endif

        #if NOT_EXCLUDED(OP_trace)
        DECLARE_CUSTOM_OP(trace, 1, 1, false, 0, 0);
        #endif

        #if NOT_EXCLUDED(OP_random_shuffle)
        DECLARE_OP(random_shuffle, 1, 1, true);
        #endif

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
        #if NOT_EXCLUDED(OP_clip_by_global_norm)
        DECLARE_CUSTOM_OP(clip_by_global_norm, 1, 2, true, 1, 0);
        #endif

        DECLARE_CUSTOM_OP(tri, -2, 1, false, 0, 1);

        DECLARE_CUSTOM_OP(triu, 1, 1, false, 0, 0);

        DECLARE_CUSTOM_OP(triu_bp, 2, 1, false, 0, 0);

    }
}

#endif