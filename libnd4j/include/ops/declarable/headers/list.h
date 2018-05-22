//
//  @author raver119@gmail.com
//

#ifndef LIBND4J_HEADERS_LIST_H
#define LIBND4J_HEADERS_LIST_H

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        // list operations, basically all around NDArrayList

        /**
         * This operations puts given NDArray into (optionally) given NDArrayList. 
         * If no NDArrayList was provided - new one will be created
         */
        #if NOT_EXCLUDED(OP_write_list)
        DECLARE_LIST_OP(write_list, 2, 1, 0, -2);
        #endif

        /**
         * This operation concatenates given NDArrayList, and returns NDArray as result
         */
        #if NOT_EXCLUDED(OP_stack_list)
        DECLARE_LIST_OP(stack_list, 1, 1, 0, 0);
        #endif

        /**
         * This operations selects specified index fron NDArrayList and returns it as NDArray
         * Expected arguments:
         * x: non-empty list
         * indices: optional, scalar with index
         * 
         * Int args:
         * optional, index
         */
        #if NOT_EXCLUDED(OP_read_list)
        DECLARE_LIST_OP(read_list, 1, 1, 0, 0);
        #endif

        /**
         * This operations selects specified indices fron NDArrayList and returns them as NDArray
         * Expected arguments:
         * x: non-empty list
         * indices: optional, vector with indices
         * 
         * Int args:
         * optional, indices
         */
        #if NOT_EXCLUDED(OP_pick_list)
        DECLARE_LIST_OP(pick_list, 1, 1, -2, -2);
        #endif

        /**
         * This operations returns scalar, with number of existing arrays within given NDArrayList
         * Expected arguments:
         * x: list
         */
        #if NOT_EXCLUDED(OP_size_list)
        DECLARE_LIST_OP(size_list, 1, 1, 0, 0);
        #endif

        /**
         * This operation creates new empty NDArrayList
         */
        #if NOT_EXCLUDED(OP_create_list)
        DECLARE_LIST_OP(create_list, 1, 2, 0, -2);
        #endif

        /**
         * This operation unpacks given NDArray into specified NDArrayList wrt specified indices
         */
        #if NOT_EXCLUDED(OP_scatter_list)
        DECLARE_LIST_OP(scatter_list, 1, 1, 0, -2);
        #endif

        /**
         * This operation splits given NDArray into chunks, and stores them into given NDArrayList wert sizes
         * Expected arguments:
         * list: optional, NDArrayList. if not available - new NDArrayList will be created
         * array: array to be split
         * sizes: vector with sizes for each chunk
         */
        #if NOT_EXCLUDED(OP_split_list)
        DECLARE_LIST_OP(split_list, 2, 1, 0, -2);
        #endif

        /**
         * This operation builds NDArray from NDArrayList using indices
         * Expected arguments:
         * x: non-empty list
         * indices: vector with indices for gather operation
         */
        #if NOT_EXCLUDED(OP_gather_list)
        DECLARE_LIST_OP(gather_list, 2, 1, 0, -2);
        #endif

        /**
         * This operation clones given NDArrayList
         */
        #if NOT_EXCLUDED(OP_clone_list)
        DECLARE_LIST_OP(clone_list, 1, 1, 0, 0);
        #endif

        /**
         * This operation unstacks given NDArray into NDArrayList
         */
        #if NOT_EXCLUDED(OP_unstack_list)
        DECLARE_LIST_OP(unstack_list, 1, 1, 0, 0);
        #endif
    }
}

#endif