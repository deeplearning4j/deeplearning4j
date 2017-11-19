//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(crelu, 1, 1, false, 0, 0) {
            auto x = INPUT_VARIABLE(0);

            auto tmp = x->dup();
            tmp->template applyTransform<simdOps::Neg<T>>();

            auto z = OUTPUT_VARIABLE(0);

            NDArrayFactory<T>::concat({x, tmp}, -1, z);

            // TODO: make this configurable?
            T threshold = (T) 0.0f;
            z->template applyTransform<simdOps::RELU<T>>(&threshold);

            STORE_RESULT(z);

            delete tmp;

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(crelu) {
            auto inShape = inputShape->at(0);
            std::vector<int> shape;
            for (int e = 0; e < shape::rank(inShape); e++)
                shape.emplace_back(shape::shapeOf(inShape)[e]);
            
            shape[shape.size()-1] *= 2;
            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inShape), int);
            if (shape::order(inShape) == 'c')
                shape::shapeBuffer(shape.size(), shape.data(), newShape);
            else
                shape::shapeBufferFortran(shape.size(), shape.data(), newShape);

            return new ShapeList(newShape);
        }
    }
}