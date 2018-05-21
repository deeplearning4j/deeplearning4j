//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_random_exponential)

#include <ops/declarable/headers/random.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(random_exponential, 1, 1, true, 1, 0) {
            // uniform distribution
            auto rng = block.getRNG();

            if (rng == nullptr)
                return Status::THROW("RNG is null, aborting...");

            auto x = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);

            if (block.width() == 1)
                functions::random::RandomFunction<T>::template execTransform<randomOps::ExponentialDistribution<T>>(block.getRNG(), z->getBuffer(), z->getShapeInfo(), block.getTArguments()->data());
            else {
                auto y = INPUT_VARIABLE(1);
                REQUIRE_TRUE(y->isSameShape(z), 0, "ExponentialDistribution: Y shape should be equal to Z shape");

                functions::random::RandomFunction<T>::template execTransform<randomOps::ExponentialDistribution<T>>(block.getRNG(), y->getBuffer(), y->getShapeInfo(), z->getBuffer(), z->getShapeInfo(), block.getTArguments()->data());
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }


        DECLARE_SHAPE_FN(random_exponential) {
            auto in = INPUT_VARIABLE(0);
            auto shape = in->template asVectorT<Nd4jLong>();

            Nd4jLong *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(shape.size()), Nd4jLong);
            shape::shapeBuffer(shape.size(), shape.data(), newShape);

            return SHAPELIST(newShape);
        }
    }
}

#endif