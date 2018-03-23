//
//  @author @shugeo
//

#include <ops/declarable/CustomOperations.h>
//#include <array>

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
        for (int e = 0; e < numOfData; e++) {
            NDArray<T>* data = INPUT_VARIABLE(numOfData + e);
            NDArray<T>* index = INPUT_VARIABLE(e);
            REQUIRE_TRUE(data->lengthOf() == index->lengthOf(), 0,
                "dynamic_stitch: The length of proper index and data arrays should be equal. But %i and %i were given.", 
                index->lengthOf(), data->lengthOf());

            for (int i = 0; i < index->lengthOf(); i++) {
                T val = (*data)(i); 
                int pos = (*index)(i);
                REQUIRE_TRUE(pos >= 0, 0, "dynamic_stitch: Index value should be non-negative."
                    " But %i was given", pos);
                REQUIRE_TRUE(pos < output->lengthOf(), 0, 
                    "dynamic_stitch: Index should be less than %i. But %i was given", 
                    output->lengthOf(), pos);
                output->putScalar(pos, val);
            }
        }
        
        return ND4J_STATUS_OK;
    }

    DECLARE_SHAPE_FN(dynamic_stitch) {

        int maxValue = 0;
        int numOfData = block.width();
        numOfData /= 2; // only index part it's needed to review
        for(int i = 0; i < numOfData; i++) {
            NDArray<T>* input = INPUT_VARIABLE(i);
            
            for (int e = 0; e < input->lengthOf(); ++e) {
                if (T(maxValue) < (*input)(e))
                    maxValue = static_cast<int>((*input)(e));
            }
        }

        auto shapes = SHAPELIST();
        int *newShape;
        ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(1), int);

        shape::shapeVector(maxValue + 1, newShape);

        shapes->push_back(newShape);
        return shapes;
    }
}
}