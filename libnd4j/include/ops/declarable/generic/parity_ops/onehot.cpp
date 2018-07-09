//
// Created by raver119 on 01/11/17.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_onehot)

#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(onehot, 1, 1, false, -2, -2) {
            auto input = INPUT_VARIABLE(0);

            T on(1.0f); // T_ARG(0);
            T off(0.0f); //T_ARG(1);

            auto depth = -1; //INT_ARG(0);
            auto axis = -1; //INT_ARG(1);

            if (block.numI() > 0)
                axis = INT_ARG(0);

            if (block.numI() > 1) {
                depth = INT_ARG(1);
            } else if (block.width() > 1) {
                depth = static_cast<int>(INPUT_VARIABLE(1)->getScalar(0));
            }

            REQUIRE_TRUE(depth > 0, 0, "OneHot: depth must be positive value");


            if (block.width() > 2) {
                on = INPUT_VARIABLE(2)->getScalar(0);

                if (block.width() > 3)
                    off = INPUT_VARIABLE(3)->getScalar(0);
            } else if (block.numT() > 0) {
                on = T_ARG(0);

                if (block.numT() > 1)
                    off = T_ARG(1);
            }

            auto output = OUTPUT_VARIABLE(0);

            if (axis < 0)
                axis = output->rankOf() + axis;

            auto vec = ShapeUtils<T>::convertAxisToTadTarget(input->rankOf(), {axis});
            auto tads = output->allTensorsAlongDimension({axis});
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
            auto inShape = inputShape->at(0);

            int depth = -1;
            int axis = -1;

            if (block.numI() > 0)
                axis = INT_ARG(0);

             if (block.numI() > 1) {
                depth = INT_ARG(1);
            } else if (block.width() > 1) {
                depth = static_cast<int>(INPUT_VARIABLE(1)->getScalar(0));
            }

            REQUIRE_TRUE(depth > 0, 0, "OneHot: depth must be positive value");

            Nd4jLong *newShape;
            int rank = shape::rank(inShape);

            if (inShape[0] == 2 && inShape[1] == 1) {
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);

                Nd4jLong* shape;
                ALLOCATE(shape, block.getWorkspace(), rank, Nd4jLong);
                memcpy(shape, shape::shapeOf(inShape), rank * sizeof(Nd4jLong));

                ShapeUtils<T>::insertDimension(rank, shape, axis, depth);
                shape::shapeBuffer(rank, shape, newShape);

                RELEASE(shape, block.getWorkspace());
            } else {
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(rank+1), Nd4jLong);

                if (axis < 0)
                    axis = rank + 1 + axis;

                std::vector<Nd4jLong> shape;
                for (int e = 0; e < rank; e++)
                    shape.push_back(shape::shapeOf(inShape)[e]);

                shape.insert(shape.begin() + axis, depth);
                shape::shapeBuffer(rank+1, shape.data(), newShape);
            }

            return SHAPELIST(newShape);
        }
    }
}

#endif