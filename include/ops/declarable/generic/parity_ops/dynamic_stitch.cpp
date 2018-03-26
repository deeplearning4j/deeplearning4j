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
        if (output->isVector()) {
            for (int e = 0; e < numOfData; e++) {
                NDArray<T>* data = INPUT_VARIABLE(numOfData + e);
                NDArray<T>* index = INPUT_VARIABLE(e);
                for (int i = 0; i < index->lengthOf(); i++) {
                    int pos = (*index)(i);
                    REQUIRE_TRUE(pos >= 0, 0, "dynamic_stitch: Index value should be non-negative."
                        " But %i was given", pos);
                    REQUIRE_TRUE(pos < output->lengthOf(), 0,
                        "dynamic_stitch: Index should be less than %i. But %i was given",
                        output->lengthOf(), pos);
                    (*output)(pos) = (*data)(i);
                }
            }
        }
        else {
        std::vector<int> restDims(output->rankOf() - 1);
        for (int i = restDims.size(); i > 0;  i--)
            restDims[restDims.size() - i] = output->rankOf() - i;

        ResultSet<T>* listOfOutTensors = NDArrayFactory<T>::allTensorsAlongDimension(output, restDims);

        for (int e = 0; e < numOfData; e++) {
            NDArray<T>* data = INPUT_VARIABLE(numOfData + e);
            NDArray<T>* index = INPUT_VARIABLE(e);
            std::vector<int> sourceDims(data->rankOf() - index->rankOf());
            for (int i = sourceDims.size(); i > 0;  i--)
                sourceDims[sourceDims.size() - i] = data->rankOf() - i;

            ResultSet<T>* listOfTensors = NDArrayFactory<T>::allTensorsAlongDimension(data, sourceDims);

            for (int i = 0; i < index->lengthOf(); i++) {
                int pos = (*index)(i);
                REQUIRE_TRUE(pos >= 0, 0, "dynamic_stitch: Index value should be non-negative."
                    " But %i was given", pos);
                REQUIRE_TRUE(pos < output->lengthOf(), 0, 
                    "dynamic_stitch: Index should be less than %i. But %i was given", 
                    output->lengthOf(), pos);

                listOfOutTensors->at(pos)->assign(listOfTensors->at(i));
            }
            delete listOfTensors;
        }
        delete listOfOutTensors;
        }
        return ND4J_STATUS_OK;
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