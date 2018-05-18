//
//  @author GS <sgazeos@gmail.com>
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_dynamic_partition)

#include <ops/declarable/CustomOperations.h>
#include <array>
#include <ops/declarable/helpers/dynamic.h>

namespace nd4j {
namespace ops {
    CUSTOM_OP_IMPL(dynamic_partition, 2, 1, false, 0, 1) {
        auto input = INPUT_VARIABLE(0);
        auto indices = INPUT_VARIABLE(1);

        input->printShapeInfo("input");
        indices->printShapeInfo("indices");

        REQUIRE_TRUE(input->rankOf() >= indices->rankOf(), 0, "dynamic_partition: data tensor rank should be non-lesser than indices\' tensor, but %i < %i given,",
            input->rankOf(), indices->rankOf());
        for (int dim = 0; dim < indices->rankOf(); dim++) {
            REQUIRE_TRUE(input->sizeAt(dim) == indices->sizeAt(dim), 0, "dynamic_partition: dimensions should be equals for data and indices tensors, but at axis[%i] %i != %i given",
                dim, input->sizeAt(dim), indices->sizeAt(dim));
        }

        auto numPartition = INT_ARG(0);
        std::vector<NDArray<T>*> outputList(numPartition);
        for(int o = 0; o < numPartition; ++o) {
            outputList[o] = OUTPUT_VARIABLE(o);
        }
        helpers::dynamicPartitionFunctor(input, indices, outputList);

        return ND4J_STATUS_OK;
    }

    DECLARE_SHAPE_FN(dynamic_partition) {
        auto numPartition = INT_ARG(0);
        NDArray<T>* indices = INPUT_VARIABLE(1);
        std::vector<int> partitionSizes(numPartition, 0);
        auto in = inputShape->at(0);
        auto idx = inputShape->at(1); 
        for (int i = 0; i < numPartition; i++) {
            for (int e = 0; e < indices->lengthOf(); ++e)
                if ((*indices)(e) == T(i))
                    partitionSizes[i]++;
        }

        auto shapes = SHAPELIST();
        int outRank = shape::rank(in) - shape::rank(idx) + 1;
        for (int e = 0; e < numPartition; e++) {
            Nd4jLong *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(outRank), Nd4jLong);
            //shape::shapeVector(partitionSizes[e], newShape);
            newShape[0] = outRank;
            newShape[1] = partitionSizes[e];
            for(int i = 1; i < outRank; ++i)
                newShape[i + 1] = shape::sizeAt(in, outRank + i - 1);

            shape::updateStrides(newShape, shape::order(in));

            shapes->push_back(newShape);
        }

        return shapes;
    }
}
}

#endif