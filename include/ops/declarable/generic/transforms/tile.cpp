//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(tile, 1, 1, false, 0, -2) {
            auto input = INPUT_VARIABLE(0);
            NDArray<T>* output;
            std::vector<int> reps(input->rankOf());
            bool overwrite = false;

            if (block.getIArguments()->size() >= input->rankOf()) {
                reps = *(block.getIArguments());

                output = OUTPUT_VARIABLE(0);
            } else if (block.width() > 1) {
                auto reps_vector = INPUT_VARIABLE(1);
                REQUIRE_TRUE(reps_vector->lengthOf() == input->rankOf(), 0, "Tile: repeats vector length should be equal to input rank");

                for (int e = 0; e < input->rankOf(); e++)
                    reps[e] = (int) reps_vector->getScalar(e);


                std::vector<int> shape(input->rankOf());
                for (int e = 0; e < input->rankOf(); e++)
                    shape[e] = input->sizeAt(e) * reps[e];

                output = new NDArray<T>(input->ordering(), shape, block.getWorkspace());

                overwrite = true;
            } else {
                REQUIRE_TRUE(false, 0, "Tile: this op requires input array and repeats vector, either as IArgs or second array");
            }
            
            input->tile(reps, *output);

            if (overwrite) {
                OVERWRITE_RESULT(output);
            }

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(tile) {
            auto shapeList = new ShapeList();
            auto inShape = inputShape->at(0);

            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inShape), int);
            std::vector<int> shape(shape::rank(inShape));

            for (int e = 0; e < shape::rank(inShape); e++)
                shape[e] = shape::sizeAt(inShape, e);

            if (block.getIArguments()->size() >= shape.size()) {
                auto reps = block.getIArguments();

                for (int e = 0; e < shape.size(); e++)
                    shape[e] *= reps->at(e);

                if (shape::order(inShape) == 'c')
                    shape::shapeBuffer(shape.size(), shape.data(), newShape);
                else
                    shape::shapeBufferFortran(shape.size(), shape.data(), newShape);
            } else {
                // runtime evaluation, sorry
                REPLICATE_SHAPE(inShape, newShape);
            }

            shapeList->push_back(newShape);

            return shapeList;
        }
    }
}