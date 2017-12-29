//
// Created by raver119 on 23.11.17.
//

//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(squeeze, -1, -1, true, 0, 0) {

            for (int e = 0; e < block.width(); e++) {
                auto input = INPUT_VARIABLE(e);
                auto output = OUTPUT_VARIABLE(e);

                if (input->rankOf() == 0 || (input->rankOf() == 1 && input->lengthOf() == 1)) {
                    output->assign(input->buffer()[0]);
                    continue;
                }

                std::vector<int> shape;
                for (int d = 0; d < input->rankOf(); d++)
                    if (input->sizeAt(d) > 1)
                        shape.emplace_back(input->sizeAt(d));

                if (block.isInplace()) {
                    output->reshapei(input->ordering(), shape);
                } else {
                    auto tmp = input->reshape(input->ordering(), shape);
                    output->assign(tmp);
                    delete tmp;
                }
            }

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(squeeze) {
            auto shapeList = new ShapeList();

            for (int e = 0; e < inputShape->size(); e++) {
                int* newShape;
                auto in = inputShape->at(e);
                auto rank = shape::rank(in);
                auto length = shape::length(in);

                if (rank == 0 || (rank == 1 && length == 1)) {
                    ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(0), int);
                    newShape[0] = 0;
                    newShape[1] = 0;
                    newShape[2] = 1;
                    newShape[3] = 99;

                    shapeList->push_back(newShape);
                    continue;
                }

                auto order = shape::order(in);
                auto oldShape = shape::shapeOf(in);

            

                std::vector<int> shape;
                for (int i = 0; i < rank; i++) {
                    if (oldShape[i] > 1)
                        shape.emplace_back(oldShape[i]);
                }

                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(shape.size()), int);
                if (order == 'c')
                    shape::shapeBuffer(shape.size(), shape.data(), newShape);
                else
                    shape::shapeBufferFortran(shape.size(), shape.data(), newShape);

                shapeList->push_back(newShape);
            }

            return shapeList;
        }
    }
}