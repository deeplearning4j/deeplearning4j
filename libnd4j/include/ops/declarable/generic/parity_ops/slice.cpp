//
// Created by raver119 on 02.11.2017.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_slice)

#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(slice, 1, 1, false, 0, -1) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            std::vector<int> begin;
            std::vector<int> end;

            int x_rank = input->rankOf();

            ShapeUtils<T>::copyVectorPart(begin, *(block.getIArguments()), x_rank, 0);
            ShapeUtils<T>::copyVectorPart(end, *(block.getIArguments()), x_rank, x_rank);

            IndicesList indices;
            for (int e = 0; e < x_rank; e++) {
                int stop = end[e];
                int start = begin[e];


                REQUIRE_TRUE(stop > 0, 0, "Slice: interval for dimension %i is less then 1", e);

                indices.push_back(NDIndex::interval(start, start+stop, 1));
            }
            auto sub = input->subarray(indices);
            output->assign(sub);

            delete sub;

            STORE_RESULT(output);

            return Status::OK();
        }

        DECLARE_SHAPE_FN(slice) {
            auto inShape = inputShape->at(0);

            std::vector<int> begin;
            std::vector<int> end;

            int x_rank = shape::rank(inShape);

            ShapeUtils<T>::copyVectorPart(begin, *(block.getIArguments()), x_rank, 0);
            ShapeUtils<T>::copyVectorPart(end, *(block.getIArguments()), x_rank, x_rank);

            Nd4jLong *newShape;
            std::vector<Nd4jLong> shape;
            for (int e = 0; e < x_rank; e++) {
                int stop = end[e];
                int start = begin[e];

                shape.push_back(stop);
            }

            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(x_rank), Nd4jLong);
            shape::shapeBuffer(x_rank, shape.data(), newShape);

            return SHAPELIST(newShape);
        }



        CUSTOM_OP_IMPL(slice_bp, 2, 1, false, 0, -1) {
            auto input = INPUT_VARIABLE(0);
            auto epsNext = INPUT_VARIABLE(1);
            auto output = OUTPUT_VARIABLE(0);

            std::vector<int> begin;
            std::vector<int> end;

            int x_rank = input->rankOf();

            ShapeUtils<T>::copyVectorPart(begin, *(block.getIArguments()), x_rank, 0);
            ShapeUtils<T>::copyVectorPart(end, *(block.getIArguments()), x_rank, x_rank);

            IndicesList indices;
            for (int e = 0; e < x_rank; e++) {
                int stop = end[e];
                int start = begin[e];


                REQUIRE_TRUE(stop > 0, 0, "Slice: interval for dimension %i is less then 1", e);

                indices.push_back(NDIndex::interval(start, start+stop, 1));
            }
            auto sub = output->subarray(indices);
            sub->assign(epsNext);
            delete sub;

            return Status::OK();
        }

        DECLARE_SHAPE_FN(slice_bp) {
            auto inShape = inputShape->at(0);
            Nd4jLong *newShape;
            COPY_SHAPE(inShape, newShape);

            return SHAPELIST(newShape);
        }
    }
}

#endif