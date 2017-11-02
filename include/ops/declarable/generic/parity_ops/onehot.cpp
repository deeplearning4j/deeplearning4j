//
// Created by raver119 on 01/11/17.
//

#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(onehot, 1, 1, false, 2, 2) {
            auto input = INPUT_VARIABLE(0);

            T on = T_ARG(0);
            T off = T_ARG(1);

            auto depth = INT_ARG(0);
            auto axis = INT_ARG(1);

            //REQUIRE_TRUE(input->isVector(), 0, "One-hot input should be Vector, but got %iD instead", input->rankOf());

            auto output = OUTPUT_VARIABLE(0);

            if (axis < 0)
                axis = output->rankOf() + axis;

            auto tads = NDArrayFactory<T>::allTensorsAlongDimension(output, {axis});
            for (int e = 0; e < tads->size(); e++) {
                auto tad = tads->at(e);
                tad->assign(off);

                int idx = (int) input->getScalar(e);
                if (idx < 0 || idx >= tad->lengthOf())
                    continue;

                tad->putIndexedScalar(idx, on);
            }

            delete tads;

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(onehot) {
            int *inShape = inputShape->at(0);

            auto depth = INT_ARG(0);
            auto axis = INT_ARG(1);

            int *newShape;
            int rank = shape::rank(inShape);
            if (shape::isVector(inShape)) {
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(rank), int);

                int* shape;
                ALLOCATE(shape, block.getWorkspace(), rank, int);
                memcpy(shape, shape::shapeOf(inShape), rank * sizeof(int));

                ShapeUtils<T>::insertDimension(rank, shape, axis, depth);
                shape::shapeBuffer(rank, shape, newShape);

                RELEASE(shape, block.getWorkspace());
            } else {
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(rank+1), int);

                std::vector<int> shape;
                for (int e = 0; e < rank; e++)
                    shape.push_back(shape::shapeOf(inShape)[e]);

                if (axis < 0)
                    axis = rank + 1 + axis;

                shape.insert(shape.begin() + axis, depth);
                shape::shapeBuffer(rank+1, shape.data(), newShape);
            }

            return new ShapeList(newShape);
        }
    }
}
