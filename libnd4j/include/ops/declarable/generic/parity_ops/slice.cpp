//
// Created by raver119 on 02.11.2017.
//

#include <op_boilerplate.h>
//#if NOT_EXCLUDED(OP_slice)

#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(slice, 1, 1, false, 0, -2) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            int x_rank = input->rankOf();

            std::vector<int> begin;
            std::vector<int> end;

            if (block.width() == 3) {
                auto b = INPUT_VARIABLE(1);
                auto e = INPUT_VARIABLE(2);

                begin = b->template asVectorT<int>();
                end = e->template asVectorT<int>();
            } else {
                REQUIRE_TRUE(block.numI() >= x_rank * 2, 0, "Number of IArgs should be equal to [%i] but got [%i] instead", x_rank * 2, block.numI());

                ShapeUtils<T>::copyVectorPart(begin, *(block.getIArguments()), x_rank, 0);
                ShapeUtils<T>::copyVectorPart(end, *(block.getIArguments()), x_rank, x_rank);
            }

            REQUIRE_TRUE(begin.size() == x_rank, 0, "begin array should have length of [%i] but got [%i] instead", x_rank, begin.size());
            REQUIRE_TRUE(end.size() == x_rank, 0, "end array should have length of [%i] but got [%i] instead", x_rank, end.size());

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
            auto x_rank = shape::rank(inShape);

            std::vector<int> begin;
            std::vector<int> end;

            if (block.width() == 3) {
                auto b = INPUT_VARIABLE(1);
                auto e = INPUT_VARIABLE(2);

                begin = b->template asVectorT<int>();
                end = e->template asVectorT<int>();
            } else {
                REQUIRE_TRUE(block.numI() >= x_rank * 2, 0, "Number of IArgs should be equal to [%i] but got [%i] instead", x_rank * 2, block.numI());

                ShapeUtils<T>::copyVectorPart(begin, *(block.getIArguments()), x_rank, 0);
                ShapeUtils<T>::copyVectorPart(end, *(block.getIArguments()), x_rank, x_rank);
            }

            REQUIRE_TRUE(begin.size() == x_rank, 0, "begin array should have length of [%i] but got [%i] instead", x_rank, begin.size());
            REQUIRE_TRUE(end.size() == x_rank, 0, "end array should have length of [%i] but got [%i] instead", x_rank, end.size());

            Nd4jLong *newShape;
            std::vector<Nd4jLong> shape;
            for (int e = 0; e < x_rank; e++) {
                auto stop = end[e];
                auto start = begin[e];

                shape.emplace_back(stop);
            }

            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(x_rank), Nd4jLong);
            shape::shapeBuffer(x_rank, shape.data(), newShape);

            return SHAPELIST(newShape);
        }



        CUSTOM_OP_IMPL(slice_bp, 2, 1, false, 0, -1) {
            auto input = INPUT_VARIABLE(0);
            auto epsNext = block.width() == 4 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(1);

            auto output = OUTPUT_VARIABLE(0);

            int x_rank = input->rankOf();

            std::vector<int> begin;
            std::vector<int> end;

            if (block.width() == 4) {
                auto b = INPUT_VARIABLE(1);
                auto e = INPUT_VARIABLE(2);

                begin = b->template asVectorT<int>();
                end = e->template asVectorT<int>();
            } else {
                REQUIRE_TRUE(block.numI() >= x_rank * 2, 0, "Number of IArgs should be equal to [%i] but got [%i] instead", x_rank * 2, block.numI());

                ShapeUtils<T>::copyVectorPart(begin, *(block.getIArguments()), x_rank, 0);
                ShapeUtils<T>::copyVectorPart(end, *(block.getIArguments()), x_rank, x_rank);
            }

            REQUIRE_TRUE(begin.size() == x_rank, 0, "begin array should have length of [%i] but got [%i] instead", x_rank, begin.size());
            REQUIRE_TRUE(end.size() == x_rank, 0, "end array should have length of [%i] but got [%i] instead", x_rank, end.size());

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

//#endif