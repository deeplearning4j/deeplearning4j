//
// Created by george@skymind.io on 2/21/2018.
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/segment.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(segment_sum, 2, 1, false, 0, 0) {
            NDArray<T>* input = INPUT_VARIABLE(0);
            NDArray<T>* idxSegments = INPUT_VARIABLE(1);
            NDArray<T>* segmentedOutput = OUTPUT_VARIABLE(0);
            REQUIRE_TRUE(idxSegments->isVector(), 0, "segment_sum: segment indexes array should be a vector, but it rank is %i.", idxSegments->rankOf());
            REQUIRE_TRUE(idxSegments->lengthOf() == input->sizeAt(0), 0, "segment_sum: segment indexes array length should be equal to the input first dimension, but %i != %i.", idxSegments->lengthOf(), input->sizeAt(0));

            T expected, wrong;

            REQUIRE_TRUE(helpers::segmentIndicesValidate(idxSegments, expected, wrong), 0, "segment_sum: segment indices should be arranged, but %2.1f > %2.1f",
                    expected, wrong);

            helpers::segmentSumFunctor(input, idxSegments, segmentedOutput);

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(segment_sum) {

            NDArray<T>* idxVector = INPUT_VARIABLE(1);

            auto in = inputShape->at(0);
            int outRank = shape::rank(in);
            Nd4jLong* outputShape = nullptr;
            T val = (*idxVector)(idxVector->lengthOf() - 1);

            int numOfClasses = static_cast<int>(val) + 1;

            ALLOCATE(outputShape, block.getWorkspace(), shape::shapeInfoLength(outRank), Nd4jLong);

            outputShape[0] = outRank;
            outputShape[1] = numOfClasses;
            for(int i = 1; i < outRank; ++i)
                outputShape[i + 1] = shape::sizeAt(in, i);

            shape::updateStrides(outputShape, shape::order(in));

            return SHAPELIST(outputShape);
        }
    }

}
