//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_random_normal)

#include <ops/declarable/headers/random.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(random_normal, 1, 1, true, 2, 0) {
            // normal distribution
            auto rng = block.getRNG();

            REQUIRE_TRUE(rng != nullptr, 0, "RNG isn't defined for this Graph instance");

            auto x = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);

            functions::random::RandomFunction<T>::template execTransform<randomOps::GaussianDistribution<T>>(block.getRNG(), z->getBuffer(), z->getShapeInfo(), z->getBuffer(), z->getShapeInfo(), z->getBuffer(), z->getShapeInfo(), block.getTArguments()->data());

            return Status::OK();
        }

        DECLARE_SHAPE_FN(random_normal) {
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