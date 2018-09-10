/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_strided_slice)

#include <array>
#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>
#include <helpers/BitwiseUtils.h>

namespace nd4j {
    namespace ops {

        constexpr int kShrinkAxis = -1, kNewAxis = -2;

        struct StridedSliceSparseSpec {
            int dims;
            int num_add_axis_after_ellipsis;
            std::vector<int>* begin_tensor;
            const std::vector<int>* end_tensor;
            const std::vector<int>* strides_tensor;
            const int begin_mask, end_mask;
            int ellipsis_mask;
            const int new_axis_mask, shrink_axis_mask;
        };

        struct StridedSliceDenseSpec {
            const int dims;
            int begin_mask;
            int end_mask;
            bool begin_valid;
            bool end_valid;
            std::vector<int>& begin;
            std::vector<int>& end;
            std::vector<int>& strides;
            std::vector<int> final_shape_gather_indices;
            int shrink_axis_mask;

            public:
                bool buildDenseSpec(StridedSliceSparseSpec& sparse_spec) {
                    if (this->begin.size() < dims)
                        this->begin.resize(dims);

                    if (this->end.size() < dims)
                        this->end.resize(dims);

                    if (this->strides.size() < dims)
                        this->strides.resize(dims);
                    this->begin_mask = 0;
                    this->end_mask = 0;
                    this->shrink_axis_mask = 0;
                    {
                        int full_index = 0;

                        this->begin_valid = sparse_spec.begin_tensor != nullptr;
                        this->end_valid = sparse_spec.end_tensor != nullptr;

                        for (int e = 0; e < sparse_spec.dims; e++) {
                            if ((1 << e) & sparse_spec.ellipsis_mask) {
                                int next_index = nd4j::math::nd4j_min<int>(this->dims - (sparse_spec.dims - e) + 1 + sparse_spec.num_add_axis_after_ellipsis, this->dims);
                            
                                for (; full_index < next_index; full_index++) {
                                    // new_axis' aren't real axis so you have to skip
                                    this->begin[full_index] = this->end[full_index] = 0;
                                    this->strides[full_index] = 1;
                                    this->begin_mask |= (1 << full_index);
                                    this->end_mask |= (1 << full_index);
                                    this->final_shape_gather_indices.push_back(full_index);
                                }
                            } else if ((1 << e) & sparse_spec.new_axis_mask) {
                                this->final_shape_gather_indices.emplace_back(kNewAxis);
                            } else {
                                if (full_index == this->begin.size()) {
                                    nd4j_printf("Index out of range: %i out of %i\n", full_index, this->dims);
                                    return false;
                                }

                                // Gather slicing spec into appropriate index
                                if (sparse_spec.begin_tensor != nullptr)
                                    this->begin[full_index] = sparse_spec.begin_tensor->at(e);
                                
                                
                                if (sparse_spec.end_tensor != nullptr)
                                    this->end[full_index] = sparse_spec.end_tensor->at(e);
                                
                                this->strides[full_index] = sparse_spec.strides_tensor->at(e);
                
                                if (sparse_spec.begin_mask & (1 << e))
                                    this->begin_mask |= (1 << full_index);
                                
                        
                                if (sparse_spec.end_mask & (1 << e))
                                    this->end_mask |= (1 << full_index);
                                

                                // If shrink, record where to get the dimensionality from (i.e.
                                // new_axis creates a fake 1 size dimension. Also remember shrink
                                // axis (now in dense form) so we can ignore dense->end below.
                                if (sparse_spec.shrink_axis_mask & (1 << e)) {
                                    this->final_shape_gather_indices.push_back(kShrinkAxis);
                                    this->shrink_axis_mask |= (1 << full_index);
                                } else {
                                    this->final_shape_gather_indices.push_back(full_index);
                                }
                                full_index++;
                            }
                        }
                    }
                    return true;
                }
        };

        void vectorize(std::vector<Nd4jLong>& input_shape) {
            if (input_shape.size() == 2 && input_shape[0] == 1) {
                int v = input_shape[1];
                input_shape.clear();
                input_shape.emplace_back(v);
            }
        }

        bool _preprocess_strided_slice(IndicesList* indicesList, std::vector<Nd4jLong>* final_shape, std::vector<Nd4jLong>& input_shape, std::vector<int>& begin, std::vector<int>& end, std::vector<int>& strides, int begin_mask, int ellipsis_mask, int end_mask, int new_axis_mask, int shrink_axis_mask, bool* is_identity, bool* is_simple_slice, bool* slice_dim0) {
            std::vector<int> preshape;

            bool ellipsis_seen = false;

            StridedSliceSparseSpec sparse_spec = {(int) strides.size(),
                                        0,
                                        &begin,
                                        &end,
                                        &strides,
                                        begin_mask,
                                        end_mask,
                                        ellipsis_mask,
                                        new_axis_mask,
                                        shrink_axis_mask};

            for (int i = 0; i < sparse_spec.dims; i++) {
                if (ellipsis_seen && ((1 << i) & new_axis_mask) != 0) {
                    sparse_spec.num_add_axis_after_ellipsis++;
                }
                if ((1 << i) & ellipsis_mask) {
                    ellipsis_seen = true;
                }
            }
            // If no ellipsis insert one at the end
            if (!ellipsis_seen) {
                sparse_spec.ellipsis_mask |= (1 << sparse_spec.dims);
                sparse_spec.dims++;  // this effects loop iteration below
            }

            StridedSliceDenseSpec dense_spec = {(int) input_shape.size(), 0, 0, false, false, begin, end, strides};
            if (!dense_spec.buildDenseSpec(sparse_spec))
                return false;

            //nd4j_printv("Input shape: ", input_shape);

            for (int e = 0; e < (int) input_shape.size(); e++) {
                int begin_idx = begin[e];
                int end_idx = end[e];
                int stride_idx = strides[e];
                int size_idx = input_shape[e];

                bool shrink_i = (dense_spec.shrink_axis_mask & (1 << e));

                if (stride_idx == 0) {
                    nd4j_printf("Stride is 0 at index %i\n", e);
                    return false;
                }
                if (size_idx == -1) {
                    preshape.emplace_back(shrink_i ? 1 : -1);
                    continue;
                }

                const std::array<int, 2> masks = {{dense_spec.begin_mask & (1 << e), dense_spec.end_mask & (1 << e)}};
                const std::array<int, 2> valid_range = {{stride_idx > 0 ? 0 : -1, stride_idx > 0 ? size_idx : size_idx - 1}};

                auto canonical = [stride_idx, e, size_idx, masks, valid_range](int x, int c) {
                    if (masks[c]) {
                        return stride_idx > 0 ? valid_range[c] : valid_range[(c + 1) & 1];
                    } else {
                        int x_fwd = x < 0 ? size_idx + x : x;  // make negative indices positive
                        return x_fwd < valid_range[0] ? valid_range[0] : x_fwd > valid_range[1] ? valid_range[1] : x_fwd;
                    }
                };

                if (shrink_i && stride_idx <= 0) {
                    nd4j_printf("StridedSlice: only stride 1 allowed on non-range indexing\n", e);
                    return false;
                }

                (*is_simple_slice) &= stride_idx == 1;

                const bool begin_and_end_masked = (begin_mask & (1 << e)) && (end_mask & (1 << e));

                if (dense_spec.begin_valid && dense_spec.end_valid) {
                    if (shrink_i) {
                        int x_fwd = begin_idx < 0 ? size_idx + begin_idx : begin_idx;
                        begin_idx = x_fwd;
                        end_idx = begin_idx + 1;
                        if (x_fwd < 0 || x_fwd >= size_idx) {
                            nd4j_printf("slice index %i of dimension %i out of bounds.\n", begin_idx, e);
                            return false;
                        }
                    } else {
                        begin_idx = canonical(begin_idx, 0);
                        end_idx = canonical(end_idx, 1);
                    }
                } else {
                    (*is_identity) &= stride_idx == 1 && begin_and_end_masked;
                    (*slice_dim0) &= (e == 0 && stride_idx == 1) || begin_and_end_masked;
                }

                int interval_length;
                bool known_interval = false;
                if (dense_spec.begin_valid && dense_spec.end_valid) {
                    interval_length = end_idx - begin_idx;
                    known_interval = true;
                } else if (shrink_i) {
                    interval_length = 1;
                    known_interval = true;
                } else if (begin_and_end_masked) {
                    if (size_idx > 0) {
                        if (stride_idx < 0) {
                            interval_length = -size_idx;
                        } else {
                            interval_length = size_idx;
                        }

                        known_interval = true;
                    }
                }

                if (known_interval) {
                    int size_i;
                    if (interval_length == 0 || ((interval_length < 0) != (stride_idx < 0))) {
                        size_i = 0;
                    } else {
                        size_i = interval_length / stride_idx + (interval_length % stride_idx != 0 ? 1 : 0);
                    }
                
                    if (indicesList != nullptr) {
                        if (interval_length > 1)
                            indicesList->push_back(NDIndex::interval(begin_idx, end_idx, stride_idx));
                        else if (interval_length == 1)
                            indicesList->push_back(NDIndex::point(begin_idx));
                    }

                    preshape.emplace_back(size_i);
                } else {
                    preshape.emplace_back(-1);
                }
            }


            std::vector<int> postshape;
            //nd4j_printv("Preshape: ", preshape);

            final_shape->clear();
            for (auto gather_index : dense_spec.final_shape_gather_indices) {
                if (gather_index >= 0) {
                    final_shape->emplace_back(preshape.at(gather_index));
                } else if (gather_index == kNewAxis) {
                    final_shape->emplace_back(1);
                }
            }

            //nd4j_printv("Preshape: ", preshape);
            //nd4j_printv("Postshape: ", *final_shape);

            return true;
        }


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

            bool isLive = false;

            std::vector<int> args;

            // statically evaluated 
            if (block.getIArguments()->size() > 5) {
                dim_values = block.getIArguments()->size() - 5;
                delta = dim_values % 3;
                elements = dim_values / 3;

                for (int e = 5; e < block.getIArguments()->size(); e++)
                    args.emplace_back(INT_ARG(e));

                REQUIRE_TRUE(delta == 0, 0, "StridedSlice: Number of Integer arguments should be equal to input rank x 3 = %i, but got %i instead", (x->rankOf() * 3), dim_values);

                ShapeUtils<T>::copyVectorPart(begin, args, elements, 0);
                ShapeUtils<T>::copyVectorPart(end, args, elements, elements);
                ShapeUtils<T>::copyVectorPart(strides, args, elements, elements * 2);

            } else if (block.width() >= 3) {
                isLive = true;

                auto v_begin = INPUT_VARIABLE(1);
                auto v_end = INPUT_VARIABLE(2);

                elements = v_begin->lengthOf();

                REQUIRE_TRUE(v_begin->lengthOf() == v_end->lengthOf(), 0, "StridedSlice: Length of begin/end should match, but got %i vs %i instead", (int) v_begin->lengthOf(), (int) v_end->lengthOf());

                for (int e = 0; e < v_begin->lengthOf(); e++)
                    begin.emplace_back(v_begin->getIndexedScalar<int>(e));

                for (int e = 0; e < v_end->lengthOf(); e++)
                    end.emplace_back(v_end->getIndexedScalar<int>(e));

                if (block.width() >= 4) {
                    auto v_stride = INPUT_VARIABLE(3);

                    REQUIRE_TRUE(v_stride->lengthOf() == v_begin->lengthOf(), 0, "StridedSlice: Length of begin/end/stride should match, but got %i vs %i vs %i instead", (int) v_begin->lengthOf(), (int) v_end->lengthOf(), (int) v_stride->lengthOf());

                    for (int e = 0; e < v_stride->lengthOf(); e++)
                        strides.emplace_back(v_stride->getIndexedScalar<int>(e));
                } else {
                    for (int e = 0; e < v_begin->lengthOf(); e++)
                        strides.emplace_back(1);
                }
            } else {
                REQUIRE_TRUE(false, 0, "StridedSlice: Can't find begin/end/stride information neither in IArguments or in input arrays");
            }

            IndicesList indices;
            auto input_shape = x->getShapeAsVector();
            std::vector<Nd4jLong> final_shape;
            bool is_identity;
            bool is_simple_slice;
            bool is_dim0;

            // FIXME: remove this method once we get 1D vectors supported
            vectorize(input_shape);
            REQUIRE_TRUE(_preprocess_strided_slice(&indices, &final_shape, input_shape, begin, end, strides, begin_mask, ellipsis_mask, end_mask, new_axis_mask, shrink_axis_mask, &is_identity, &is_simple_slice, &is_dim0), 0, "StridedSlice: shape calculation failed");

            auto sub = x->subarray(indices);

            auto z = OUTPUT_VARIABLE(0);
            z->assign(sub);

            delete sub;

            return Status::OK();
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


            std::vector<int> begin;
            std::vector<int> end;
            std::vector<int> strides;

            // if that's live - shape will be resolved in runtime
            if (dim_values == 0 ) {
                begin = INPUT_VARIABLE(1)->template asVectorT<int>();
                end = INPUT_VARIABLE(2)->template asVectorT<int>();
                strides = INPUT_VARIABLE(3)->template asVectorT<int>();
            } else if (dim_values > 0) {
                int delta2 = dim_values / x_rank;

                std::vector<int> args;
                for (int e = 5; e < block.getIArguments()->size(); e++)
                    args.emplace_back(INT_ARG(e));

                // FIXME: propably template required here
                ShapeUtils::copyVectorPart(begin, args, elements, 0);
                ShapeUtils::copyVectorPart(end, args, elements, elements);
                ShapeUtils::copyVectorPart(strides, args, elements, elements * 2);
            }

            REQUIRE_TRUE(begin.size() > 0 && end.size() > 0 && strides.size() > 0, 0, "Strided_Slice: empty arguments");

            Nd4jLong *newShape;
            std::vector<Nd4jLong> input_shape(shape::rank(inShape));
            std::vector<Nd4jLong> shape;
            
            for (int e = 0; e < shape::rank(inShape); e++)
                input_shape[e] = shape::shapeOf(inShape)[e];

            bool is_identity;
            bool is_simple_slice;
            bool is_dim0;

            // FIXME: remove this, once we bring in 1D NDArrays 
            vectorize(input_shape);
            bool result = _preprocess_strided_slice(nullptr, &shape, input_shape, begin, end, strides, begin_mask, ellipsis_mask, end_mask, new_axis_mask, shrink_axis_mask, &is_identity, &is_simple_slice, &is_dim0);

            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(shape.size()), Nd4jLong);
            shape::shapeBuffer(shape.size(), shape.data(), newShape);

            return SHAPELIST(newShape);
        }


        CUSTOM_OP_IMPL(strided_slice_bp, 2, 1, false, 0, 5) {
            auto x = INPUT_VARIABLE(0);
            auto epsNext = INPUT_VARIABLE(1);
            auto output = OUTPUT_VARIABLE(0);

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

            bool isLive = false;

            std::vector<int> args;

            // statically evaluated
            if (block.getIArguments()->size() > 5) {
                dim_values = block.getIArguments()->size() - 5;
                delta = dim_values % 3;
                elements = dim_values / 3;

                for (int e = 5; e < block.getIArguments()->size(); e++)
                    args.emplace_back(INT_ARG(e));

                REQUIRE_TRUE(delta == 0, 0, "StridedSliceBP: Number of Integer arguments should be equal to input rank x 3 = %i, but got %i instead", (x->rankOf() * 3), dim_values);

                ShapeUtils<T>::copyVectorPart(begin, args, elements, 0);
                ShapeUtils<T>::copyVectorPart(end, args, elements, elements);
                ShapeUtils<T>::copyVectorPart(strides, args, elements, elements * 2);

            } else if (block.width() >= 3) {
                isLive = true;

                auto v_begin = INPUT_VARIABLE(1);
                auto v_end = INPUT_VARIABLE(2);

                elements = v_begin->lengthOf();

                REQUIRE_TRUE(v_begin->lengthOf() == v_end->lengthOf(), 0, "StridedSliceBP: Length of begin/end should match, but got %i vs %i instead", (int) v_begin->lengthOf(), (int) v_end->lengthOf());

                for (int e = 0; e < v_begin->lengthOf(); e++)
                    begin.emplace_back(v_begin->getIndexedScalar<int>(e));

                for (int e = 0; e < v_end->lengthOf(); e++)
                    end.emplace_back(v_end->getIndexedScalar<int>(e));

                if (block.width() >= 4) {
                    auto v_stride = INPUT_VARIABLE(3);

                    REQUIRE_TRUE(v_stride->lengthOf() == v_begin->lengthOf(), 0, "StridedSliceBP: Length of begin/end/stride should match, but got %i vs %i vs %i instead", (int) v_begin->lengthOf(), (int) v_end->lengthOf(), (int) v_stride->lengthOf());

                    for (int e = 0; e < v_stride->lengthOf(); e++)
                        strides.emplace_back(v_stride->getIndexedScalar<int>(e));
                } else {
                    for (int e = 0; e < v_begin->lengthOf(); e++)
                        strides.emplace_back(1);
                }
            } else {
                REQUIRE_TRUE(false, 0, "StridedSliceBP: Can't find begin/end/stride information neither in IArguments or in input arrays");
            }

            IndicesList indices;
            auto input_shape = x->getShapeAsVector();
            std::vector<Nd4jLong> final_shape;
            bool is_identity;
            bool is_simple_slice;
            bool is_dim0;

            // FIXME: remove this method once we get 1D vectors supported
            vectorize(input_shape);
            REQUIRE_TRUE(_preprocess_strided_slice(&indices, &final_shape, input_shape, begin, end, strides, begin_mask, ellipsis_mask, end_mask, new_axis_mask, shrink_axis_mask, &is_identity, &is_simple_slice, &is_dim0), 0, "StridedSliceBP: shape calculation failed");

            auto sub = output->subarray(indices);
            sub->assign(epsNext);
            delete sub;

            return Status::OK();
        }

        DECLARE_SHAPE_FN(strided_slice_bp) {
            auto inShape = inputShape->at(0);
            Nd4jLong *newShape;
            COPY_SHAPE(inShape, newShape);

            return SHAPELIST(newShape);
        }
    }
}

#endif