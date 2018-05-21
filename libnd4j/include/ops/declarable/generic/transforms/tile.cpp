//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_tile)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(tile, 1, 1, false, 0, -2) {
            auto input = INPUT_VARIABLE(0);
            NDArray<T>* output;
            std::vector<Nd4jLong> reps(input->rankOf());
            bool overwrite = false;

            if (block.getIArguments()->size() >= input->rankOf()) {
                reps = ArrayUtils::toLongVector(*(block.getIArguments()));

                output = OUTPUT_VARIABLE(0);
            } else if (block.width() > 1) {
                auto reps_vector = INPUT_VARIABLE(1);
                REQUIRE_TRUE(reps_vector->lengthOf() == input->rankOf(), 0, "TILE op: repeats vector length should be equal to input rank, but got %i and %i correspondingly !", reps_vector->lengthOf(), input->rankOf());

                for (int e = 0; e < input->rankOf(); e++)
                    reps[e] = (int) (*reps_vector)(e);


                std::vector<int> shape(input->rankOf());
                for (int e = 0; e < input->rankOf(); e++)
                    shape[e] = input->sizeAt(e) * reps[e];

                output = OUTPUT_VARIABLE(0);

            } else {
                REQUIRE_TRUE(false, 0, "Tile: this op requires input array and repeats vector, either as IArgs or second array !");
            }
            
            input->tile(reps, *output);

            return Status::OK();
        }

        DECLARE_SHAPE_FN(tile) {
            auto shapeList = SHAPELIST();
            auto inShape = inputShape->at(0);

            if (block.width() > 1) {
                auto repsVectorShape = inputShape->at(1);
                REQUIRE_TRUE(shape::length(repsVectorShape) == shape::rank(inShape), 0, "TILE op: repeats vector length should be equal to input rank, but got %i and %i correspondingly !", shape::length(repsVectorShape), shape::rank(inShape));
            }

            Nd4jLong *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inShape), Nd4jLong);
            std::vector<Nd4jLong> shape(shape::rank(inShape));

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
                auto r = INPUT_VARIABLE(1);
                auto reps = r->template asVectorT<Nd4jLong>();

                // runtime evaluation, sorry
                for (int e = 0; e < shape.size(); e++)
                    shape[e] *= reps[e];

                if (shape::order(inShape) == 'c')
                    shape::shapeBuffer(shape.size(), shape.data(), newShape);
                else
                    shape::shapeBufferFortran(shape.size(), shape.data(), newShape);
            }

            shapeList->push_back(newShape);

            return shapeList;
        }
    }
}

#endif