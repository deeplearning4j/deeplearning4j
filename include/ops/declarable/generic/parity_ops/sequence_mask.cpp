//
// Created to use with batched tensor by GS <sgazeos@gmail.com> 3/27/2018
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/sequence_mask.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(sequence_mask, 1, 1, false, 0, 0) {
            NDArray<T>* input  = INPUT_VARIABLE(0);
            NDArray<T>* output = OUTPUT_VARIABLE(0);
            const int inRank = input->rankOf();

            //REQUIRE_TRUE(inRank >= 1, 0, "sequence_mask: input array must have rank >= 1, but %i given!", inRank);
            Nd4jLong maxInd = input->argMax();
            T max = input->getScalar(maxInd);
            if (block.getIArguments()->size() > 0) {
                maxInd = INT_ARG(0);
                if (T(maxInd) < max)
                    maxInd = static_cast<Nd4jLong>(max);
            }
            else if (block.width() > 1) {
                NDArray<T>* maxlen = INPUT_VARIABLE(1);
                //REQUIRE_TRUE(maxlen->lengthOf() == 1, "sequence_mask: 2nd input (max length) should be a scalar array.");
                T tmaxlen = maxlen->getScalar(0);
                if (tmaxlen > max)
                    maxInd = static_cast<Nd4jLong>(tmaxlen);
            }
            else
                maxInd = static_cast<Nd4jLong>(max);

            helpers::sequenceMask(input, output, maxInd);

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(sequence_mask) {

            Nd4jLong* outShapeInfo = nullptr;
            Nd4jLong* in = inputShape->at(0);
            int outRank = shape::rank(in) + 1;
            NDArray<T>* input = INPUT_VARIABLE(0);
            Nd4jLong maxInd = input->argMax();
            T max = input->getScalar(maxInd);
            if (block.getIArguments()->size() > 0) {
                maxInd = INT_ARG(0);
                if (T(maxInd) < max)
                    maxInd = static_cast<Nd4jLong>(max);
            }
            else if (block.width() > 1) {
                NDArray<T>* maxlen = INPUT_VARIABLE(1);
                T tmaxlen = maxlen->getScalar(0);
                if (tmaxlen > max)
                    maxInd = static_cast<Nd4jLong>(tmaxlen);
            }
            else
                maxInd = static_cast<Nd4jLong>(max);

            int lastDimension = maxInd;
            ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(outRank), Nd4jLong);
            outShapeInfo[0] = outRank;
            for(int i = 0; i < outRank - 1; ++i)
                outShapeInfo[i + 1] = shape::sizeAt(in, i);
            outShapeInfo[outRank] = lastDimension;

            shape::updateStrides(outShapeInfo, shape::order(in));

            return SHAPELIST(outShapeInfo);
    }
}
}

