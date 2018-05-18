//
// Created by raver119 on 12.10.2017.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_tear)

#include <ops/declarable/CustomOperations.h>
#include <TAD.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(tear, 1, -1, false, 0, -1) {
            auto input = INPUT_VARIABLE(0);

            REQUIRE_TRUE(!block.getIArguments()->empty(), 0, "At least 1 dimension should be specified for Tear");

            std::vector<int> dims(*block.getIArguments());

            for (auto &v: dims)
                REQUIRE_TRUE(v >= 0 && v < input->rankOf(), 0, "Tear dimensions should be non-negative values, and lower then input rank. Got %i instead", v);

            auto tads = NDArrayFactory<T>::allTensorsAlongDimension(input, dims);
            for (int e = 0; e < tads->size(); e++) {
                auto outE = OUTPUT_VARIABLE(e);
                outE->assign(tads->at(e));

                this->storeResult(block, e, *outE);
            }

            delete tads;

            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(tear) {
            auto inShape = inputShape->at(0);

            std::vector<int> dims(*block.getIArguments());
            std::sort(dims.begin(), dims.end());

            shape::TAD tad(inShape, dims.data(), (int) dims.size());
            tad.createTadOnlyShapeInfo();
            Nd4jLong numTads = shape::length(inShape) / shape::tadLength(inShape, dims.data(), (int) dims.size());

            auto result = SHAPELIST();
            for (int e = 0; e < numTads; e++) {
                Nd4jLong *newShape;
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(tad.tadOnlyShapeInfo), Nd4jLong);
                if (shape::order(inShape) == 'c')
                    shape::shapeBuffer(shape::rank(tad.tadOnlyShapeInfo), shape::shapeOf(tad.tadOnlyShapeInfo), newShape);
                else
                    shape::shapeBufferFortran(shape::rank(tad.tadOnlyShapeInfo), shape::shapeOf(tad.tadOnlyShapeInfo), newShape);
                result->push_back(newShape);
            }

            return result;
        }
    }
}

#endif