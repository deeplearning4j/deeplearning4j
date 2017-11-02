//
// Created by raver119 on 12.10.2017.
//
#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {


        CUSTOM_OP_IMPL(strided_slice, 1, 1, false, 0, -1) {
            auto x = INPUT_VARIABLE(0);

            REQUIRE_TRUE(block.getIArguments()->size() == x->rankOf() * 3, 0, "Number of Integer arguments should be equal to input rank x 3 = %i, but got %i instead", (x->rankOf() * 3), block.getIArguments()->size());

            std::vector<int> begin;
            std::vector<int> end;
            std::vector<int> strides;

            ShapeUtils<T>::copyVectorPart(begin, *(block.getIArguments()), x->rankOf(), 0);
            ShapeUtils<T>::copyVectorPart(end, *(block.getIArguments()), x->rankOf(), x->rankOf());
            ShapeUtils<T>::copyVectorPart(strides, *(block.getIArguments()), x->rankOf(), x->rankOf() * 2);

            auto z = OUTPUT_VARIABLE(0);

            Nd4jIndex offset = 0;
            Nd4jIndex length = 1;
            std::vector<int> newShape;
            IndicesList indices;
            for (int e = 0; e < x->rankOf(); e++) {
                auto start = begin[e];
                auto stop = end[e];
                auto stride = strides[e];
                auto elements = (stop - start) / stride;

                indices.push_back(NDIndex::interval(start, stop, stride));
            }

            auto sub = x->subarray(indices);
            z->assign(sub);

            STORE_RESULT(*z);
            delete sub;

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(stridedslice, strided_slice);

        DECLARE_SHAPE_FN(strided_slice) {
            auto inShape = inputShape->at(0);

            std::vector<int> begin;
            std::vector<int> end;
            std::vector<int> strides;

            int x_rank = shape::rank(inShape);

            ShapeUtils<T>::copyVectorPart(begin, *(block.getIArguments()), x_rank, 0);
            ShapeUtils<T>::copyVectorPart(end, *(block.getIArguments()), x_rank, x_rank);
            ShapeUtils<T>::copyVectorPart(strides, *(block.getIArguments()), x_rank, x_rank * 2);

            int *newShape;

            Nd4jIndex length = 1;
            std::vector<int> shape;
            for (int e = 0; e < shape::rank(inShape); e++) {
                auto start = begin[e];
                auto stop = end[e];
                auto stride = strides[e];

                int elements = 0;
                for (int i = start; i < stop; i += stride)
                    elements++;

                shape.push_back(elements);

                length *= elements;
            }

            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inShape), int);
            shape::shapeBuffer(x_rank, shape.data(), newShape);

            return new ShapeList(newShape);
        }
    }
}