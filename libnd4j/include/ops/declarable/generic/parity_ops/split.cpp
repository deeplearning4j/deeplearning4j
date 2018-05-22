//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_split)

#include <ops/declarable/headers/parity_ops.h>
#include <array>

namespace nd4j {
namespace ops {
    CUSTOM_OP_IMPL(split, 1, -1, false, 0, 1) {
        NDArray<T> *input = nullptr;
        int num_splits = INT_ARG(0);

        // axis is 0 by default
        int axis = 0;

        if (block.width() == 1) {
            input = INPUT_VARIABLE(0);
        } else {
            auto a = INPUT_VARIABLE(0);
            auto b = INPUT_VARIABLE(1);

            if (a->isScalar()) {
                // axis goes first
                axis = a->getScalar(0);
                input = b;
            } else if (b->isScalar()) {
                axis = b->getScalar(0);
                input = a;
            }
        }

        if (block.numI() == 2)
            axis = INT_ARG(1);

        REQUIRE_TRUE(input->sizeAt(axis) % num_splits == 0, 0, "Split: num_splits has wrong value, remainder of division should be 0, but it's %i", input->sizeAt(axis) % num_splits);

        int pos = 0;
        int split = input->sizeAt(axis) / num_splits;
        for (int e = 0; e < num_splits; e++) {
            auto out = OUTPUT_VARIABLE(e);

            IndicesList indices;
            for (int d = 0; d < input->rankOf(); d++) {
                if (d == axis)
                    indices.push_back(NDIndex::interval(pos, pos + split));
                else 
                    indices.push_back(NDIndex::all());
            }

            auto sub = input->subarray(indices);
            
            out->assign(sub);

            delete sub;

            pos += split;
        }



        return ND4J_STATUS_OK;
    }

    DECLARE_SHAPE_FN(split) {
        int num_splits = INT_ARG(0);
        Nd4jLong *input = nullptr;

        // axis is 0 by default
        int axis = 0;

        if (inputShape->size() == 1)
            input = inputShape->at(0);
        else {
            auto shape0 = inputShape->at(0);
            auto shape1 = inputShape->at(1);

            if (shape::isScalar(shape0)) {
                input = shape1;
                auto _a = INPUT_VARIABLE(0);
                axis = _a->getScalar(0);
            } else if (shape::isScalar(shape1)) {
                input = shape0;
                auto _a = INPUT_VARIABLE(1);
                axis = _a->getScalar(0);
            }
        }

        if (block.numI() == 2)
            axis = INT_ARG(1);

        if (axis < 0)
            axis += shape::rank(input);

        std::vector<Nd4jLong> shape(shape::rank(input));

        for (int e = 0; e < shape::rank(input); e++)
            if (e == axis)
                shape[e] = shape::sizeAt(input, e) / num_splits;
            else 
                shape[e] = shape::sizeAt(input, e);

        auto shapes = SHAPELIST();

        for (int e = 0; e < num_splits; e++) {
            Nd4jLong *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(input), Nd4jLong);

            if (shape::order(input) == 'c')
                shape::shapeBuffer(shape.size(), shape.data(), newShape);
            else
                shape::shapeBufferFortran(shape.size(), shape.data(), newShape);

            shapes->push_back(newShape);
        }

        return shapes;
    }
}
}

#endif