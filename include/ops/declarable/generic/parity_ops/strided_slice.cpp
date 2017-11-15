//
// Created by raver119 on 12.10.2017.
//
#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>
#include <helpers/BitwiseUtils.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(strided_slice, 1, 1, false, 0, 5) {
            auto x = INPUT_VARIABLE(0);

            int begin_mask = INT_ARG(0);
            int ellipsis_mask = INT_ARG(1);
            int end_mask = INT_ARG(2);
            int new_axis_mask = INT_ARG(3);
            int shrink_axis_mask = INT_ARG(4);

            int dim_values = 0; //block.getIArguments()->size() - 5;
            int delta = 0; //dim_values % 3;
            int elements = 0; //dim_values / 3;

            std::vector<int> begin;
            std::vector<int> end;
            std::vector<int> strides;

            std::vector<int> args;

            // statically evaluated 
            if (block.getIArguments()->size() > 5) {
                dim_values = block.getIArguments()->size() - 5;
                delta = dim_values % 3;
                elements = dim_values / 3;

                for (int e = 5; e < block.getIArguments()->size(); e++)
                    args.emplace_back(INT_ARG(e));

            } else if (block.width() >= 3) {
                auto v_begin = INPUT_VARIABLE(1);
                auto v_end = INPUT_VARIABLE(2);

                elements = v_begin->lengthOf();

                REQUIRE_TRUE(v_begin->lengthOf() == v_end->lengthOf(), 0, "Length of begin/end should match, but got %i vs %i instead", (int) v_begin->lengthOf(), (int) v_end->lengthOf());

                for (int e = 0; e < v_begin->lengthOf(); e++)
                    args.emplace_back((int) v_begin->getIndexedScalar(e));

                for (int e = 0; e < v_end->lengthOf(); e++)
                    args.emplace_back((int) v_end->getIndexedScalar(e));

                if (block.width() >= 4) {
                    auto v_stride = INPUT_VARIABLE(3);

                    REQUIRE_TRUE(v_stride->lengthOf() == v_begin->lengthOf(), 0, "Length of begin/end/stride should match, but got %i vs %i vs %i instead", (int) v_begin->lengthOf(), (int) v_end->lengthOf(), (int) v_stride->lengthOf());

                    for (int e = 0; e < v_stride->lengthOf(); e++)
                        args.emplace_back((int) v_stride->getIndexedScalar(e));
                } else {
                    for (int e = 0; e < v_begin->lengthOf(); e++)
                        args.emplace_back(1);
                }
            } else {
                REQUIRE_TRUE(false, 0, "Can't find begin/end/stride information neither in IArguments or in input arrays");
            }

            REQUIRE_TRUE(delta == 0, 0, "Number of Integer arguments should be equal to input rank x 3 = %i, but got %i instead", (x->rankOf() * 3), dim_values);

            int ellipsis = -1;
            if (ellipsis_mask != 0)
                ellipsis = BitwiseUtils::valueBit(ellipsis_mask);

            ShapeUtils<T>::copyVectorPart(begin, args, elements, 0);
            ShapeUtils<T>::copyVectorPart(end, args, elements, elements);
            ShapeUtils<T>::copyVectorPart(strides, args, elements, elements * 2);

            auto z = OUTPUT_VARIABLE(0);

            Nd4jIndex offset = 0;
            Nd4jIndex length = 1;
            IndicesList indices;
            std::vector<int> shrinks;
            if (shrink_axis_mask != 0)
                shrinks = BitwiseUtils::valueBits(shrink_axis_mask);

            for (int e = 0; e < x->rankOf(); e++) {
                if (e < begin.size()) {
                    auto start = begin[e];
                    auto stop = end[e];
                    auto stride = strides[e];
                    auto elements = (stop - start) / stride;

                    if (shrink_axis_mask != 0 && shrinks[e] != 0)
                        indices.push_back(NDIndex::point(start));
                    else
                        indices.push_back(NDIndex::interval(start, stop, stride));
                } else {
                    indices.push_back(NDIndex::all());
                }
            }


            auto sub = x->subarray(indices);

            std::vector<int> new_axis_positions;
            if (new_axis_mask != 0) {
                new_axis_positions = BitwiseUtils::valueBits(new_axis_mask);

                std::vector<int> newShape = sub->getShapeAsVector();

                for (int e = 0; e < sub->rankOf(); e++) {
                    if (new_axis_positions[e] == 1)
                        newShape.insert(newShape.begin() + e, 1);
                }

                sub->reshapei(sub->ordering(), newShape);
            }

            z->assign(sub);

            STORE_RESULT(*z);
            delete sub;

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(stridedslice, strided_slice);

        DECLARE_SHAPE_FN(strided_slice) {
            auto inShape = inputShape->at(0);

            int begin_mask = INT_ARG(0);
            int ellipsis_mask = INT_ARG(1);
            int end_mask = INT_ARG(2);
            int new_axis_mask = INT_ARG(3);
            int shrink_axis_mask = INT_ARG(4);

            int x_rank = shape::rank(inShape);

            int dim_values = block.getIArguments()->size() - 5;
            int delta = dim_values % 3;
            int elements = dim_values / 3;

            int delta2 = dim_values / x_rank;

            std::vector<int> begin;
            std::vector<int> end;
            std::vector<int> strides;

            std::vector<int> args;
            for (int e = 5; e < block.getIArguments()->size(); e++)
                args.emplace_back(INT_ARG(e));

            ShapeUtils<T>::copyVectorPart(begin, args, elements, 0);
            ShapeUtils<T>::copyVectorPart(end, args, elements, elements);
            ShapeUtils<T>::copyVectorPart(strides, args, elements, elements * 2);

            int *newShape;

            Nd4jIndex length = 1;
            std::vector<int> shape;
            // depending on existance of specific shapes - we should calculate result array shape
            for (int e = 0; e < shape::rank(inShape); e++) {
                if (e < begin.size()) {
                    auto start = begin[e];
                    auto stop = end[e];
                    auto stride = strides[e];

                    int els = 0;
                    for (int i = start; i < stop; i += stride)
                        els++;

                    shape.push_back(els);

                    length *= els;
                } else {
                    int els = shape::shapeOf(inShape)[e];
                    shape.push_back(els);

                    length *= els;
                }
            }

            // shape reduction part, applies only to arrays with rank > 2
            if (shrink_axis_mask != 0) {
                std::vector<int> shrinks = BitwiseUtils::valueBits(shrink_axis_mask);
                std::vector<int> shrinked;
                for (int e = 0; e < shape.size(); e++) {
                    if (shrinks[e] == 1) {
                        // noop
                    } else {
                        shrinked.push_back(shape[e]);
                    }
                }

                shape = shrinked;
            }

            // we don't want shape ranks below 2
            if (shape.size() < 2)
                shape.insert(shape.begin(), 1);

            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(shape.size()), int);
            shape::shapeBuffer(shape.size(), shape.data(), newShape);

            //shape::printShapeInfoLinear(newShape);

            return new ShapeList(newShape);
        }
    }
}