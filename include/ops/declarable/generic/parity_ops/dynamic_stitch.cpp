//
//  @author GS <sgazeos@gmail.com>
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_dynamic_stitch)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/dynamic.h>

namespace nd4j {
namespace ops {
    CUSTOM_OP_IMPL(dynamic_stitch, 2, 1, false, 0, 0) {
        int numOfData = block.width();
//        int k = 0;
        REQUIRE_TRUE(numOfData % 2 == 0, 0, 
            "dynamic_stitch: The input params should contains"
            " both indeces and data lists with same length.");
        numOfData /= 2;

        NDArray<T>* output = OUTPUT_VARIABLE(0);
        std::vector<NDArray<T>*> inputs(numOfData);
        std::vector<NDArray<T>*> indices(numOfData);
        for (int e = 0; e < numOfData; e++) {
            NDArray<T> *data = INPUT_VARIABLE(numOfData + e);
            NDArray<T> *index = INPUT_VARIABLE(e);
            inputs[e] = data;
            indices[e] = index;
        }

        return helpers::dynamicStitchFunctor(inputs, indices, output);
    }

    DECLARE_SHAPE_FN(dynamic_stitch) {

        int maxValue = 0;
        int numOfData = block.width();
        numOfData /= 2; // only index part it's needed to review
        int* restShape = inputShape->at(numOfData);
        int* firstShape = inputShape->at(0);
        for(int i = 0; i < numOfData; i++) {
            NDArray<T>* input = INPUT_VARIABLE(i);
            
            for (int e = 0; e < input->lengthOf(); ++e) {
                if (T(maxValue) < (*input)(e))
                    maxValue = static_cast<int>((*input)(e));
            }
        }

        int *outShapeInfo;
        int outRank = shape::rank(restShape) - shape::rank(firstShape) + 1;
        ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(outRank), int);

        outShapeInfo[0] = outRank;
        outShapeInfo[1] = maxValue + 1;
        for(int i = 1; i < outRank; ++i)
            outShapeInfo[i + 1] = shape::sizeAt(restShape, i);

        shape::updateStrides(outShapeInfo, shape::order(firstShape));

        //shape::shapeVector(maxValue + 1, newShape);

        return SHAPELIST(outShapeInfo);
    }
}
}

#endif