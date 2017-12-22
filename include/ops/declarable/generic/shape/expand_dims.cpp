//
// Created by raver119 on 02.11.2017.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(expand_dims, 1, 1, false, 0, 1) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            auto axis = INT_ARG(0);

            if (axis < 0)
                axis += input->rankOf();

            std::vector<int> shape;
            for(int e = 0; e < input->rankOf(); e++)
                shape.emplace_back(input->sizeAt(e));

            shape.insert(shape.begin() + axis, 1);

            auto tmp = input->reshape(input->ordering(), shape);
            output->assign(tmp);

            delete tmp;

            STORE_RESULT(output);

            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(expand_dims) {
            auto inShape = inputShape->at(0);

            auto x_rank = shape::rank(inShape);
            char order = shape::order(inShape);

            auto axis = INT_ARG(0);

            if (axis < 0)
                axis += x_rank;

            int* newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(x_rank+1), int);

            std::vector<int> shape;
            for(int e = 0; e < x_rank; e++)
                shape.emplace_back(shape::shapeOf(inShape)[e]);

            shape.insert(shape.begin() + axis, 1);

            if (order == 'c')
                shape::shapeBuffer(x_rank+1, shape.data(), newShape);
            else
                shape::shapeBufferFortran(x_rank+1, shape.data(), newShape);


            return new ShapeList(newShape);
        }
    }
}

