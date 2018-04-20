//
// Created by raver119 on 06.11.2017.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_split_list)

#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {
        LIST_OP_IMPL(split_list, 2, 1, 0, -2) {
            NDArrayList<T> *list = nullptr;
            NDArray<T>* array = nullptr;
            NDArray<T>* sizes = nullptr;

            bool hasList = false;

            if (block.width() == 3){
                list = INPUT_LIST(0);
                array = INPUT_VARIABLE(1);
                sizes = INPUT_VARIABLE(2);
                hasList = true;
            } else {
                array = INPUT_VARIABLE(0);
                sizes = INPUT_VARIABLE(1);
                list = new NDArrayList<T>(sizes->lengthOf(), false);
                block.trackList(list);
            }

            // now let's build subarrays
            //nd4j_debug("Sizes length: %i\n", sizes->lengthOf());
            int cnt = 0;
            for (int e = 0; e < sizes->lengthOf(); e++) {
                int c_size = (int) sizes->getIndexedScalar(e);
                IndicesList indices;

                //nd4j_debug("Slice start: [%i]; Slice size: [%i]\n", cnt, c_size);

                REQUIRE_TRUE(c_size > 0, 0, "Slice size should have postive value, but got %i instead", c_size);
                REQUIRE_TRUE(cnt < array->sizeAt(0) && cnt + c_size <= array->sizeAt(0), 0, "Slices size should NOT be higher then number of TADs of source array. Source size: [%i]; Slice start: [%i]; Slice size: [%i]", array->sizeAt(0), cnt, c_size);

                // we're adding our interval along zeroth dimension
                indices.push_back(NDIndex::interval(cnt, cnt+c_size));

                // and then we set all other dimensions to All
                for (int e = 1; e < array->rankOf(); e++)
                    indices.push_back(NDIndex::all());


                auto subarray = array->subarray(indices);

                auto status = list->write(e, subarray->dup(array->ordering()));
                if (status != ND4J_STATUS_OK)
                    return status;

                delete subarray;

                cnt += c_size;
            }

            if (!hasList) {
                OVERWRITE_RESULT(list);
            }

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(TensorArraySplitV3, split_list);
        DECLARE_SYN(tensorarraysplitv3, split_list);
    }
}

#endif