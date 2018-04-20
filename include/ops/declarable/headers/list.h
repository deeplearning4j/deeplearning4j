//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        // list operations, basically all around NDArrayList

        /**
         * This operations puts given NDArray into (optionally) given NDArrayList. 
         * If no NDArrayList was provided - new one will be created
         */
        DECLARE_LIST_OP(write_list, 2, 1, 0, -2);

        /**
         * This operation concatenates given NDArrayList, and returns NDArray as result
         */
        DECLARE_LIST_OP(stack_list, 1, 1, 0, 0);

        /**
         * This operations selects specified index fron NDArrayList and returns it as NDArray
         * Expected arguments:
         * x: non-empty list
         * indices: optional, scalar with index
         * 
         * Int args:
         * optional, index
         */
        DECLARE_LIST_OP(read_list, 1, 1, 0, 0);

        /**
         * This operations selects specified indices fron NDArrayList and returns them as NDArray
         * Expected arguments:
         * x: non-empty list
         * indices: optional, vector with indices
         * 
         * Int args:
         * optional, indices
         */
        DECLARE_LIST_OP(pick_list, 1, 1, -2, -2);

        /**
         * This operations returns scalar, with number of existing arrays within given NDArrayList
         * Expected arguments:
         * x: list
         */
        DECLARE_LIST_OP(size_list, 1, 1, 0, 0);

        /**
         * This operation creates new empty NDArrayList
         */
        DECLARE_LIST_OP(create_list, 1, 2, 0, -2);

        /**
         * This operation unpacks given NDArray into specified NDArrayList wrt specified indices
         */
        DECLARE_LIST_OP(scatter_list, 1, 1, 0, -2);

        /**
         * This operation splits given NDArray into chunks, and stores them into given NDArrayList wert sizes
         * Expected arguments:
         * list: optional, NDArrayList. if not available - new NDArrayList will be created
         * array: array to be split
         * sizes: vector with sizes for each chunk
         */
        DECLARE_LIST_OP(split_list, 2, 1, 0, -2);

        /**
         * This operation builds NDArray from NDArrayList using indices
         * Expected arguments:
         * x: non-empty list
         * indices: vector with indices for gather operation
         */
        DECLARE_LIST_OP(gather_list, 2, 1, 0, -2);

        /**
         * This operation clones given NDArrayList
         */
        DECLARE_LIST_OP(clone_list, 1, 1, 0, 0);

        /**
         * This operation unstacks given NDArray into NDArrayList
         */
        DECLARE_LIST_OP(unstack_list, 1, 1, 0, 0);
    }
}