/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_strided_slice)

#include <helpers/BitwiseUtils.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>
#include <legacy/NativeOpExecutioner.h>
#include <array>

namespace sd {
namespace ops {

constexpr size_t kShrinkAxis = -1, kNewAxis = -2;

struct StridedSliceSparseSpec {
  int dims;
  int num_add_axis_after_ellipsis;
  std::vector<LongType>* begin_tensor;
  const std::vector<LongType>* end_tensor;
  const std::vector<LongType>* strides_tensor;
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
  std::vector<LongType>& begin;
  std::vector<LongType>& end;
  std::vector<LongType>& strides;
  std::vector<LongType> final_shape_gather_indices;
  int shrink_axis_mask;

 public:
  bool buildDenseSpec(StridedSliceSparseSpec& sparse_spec) {
    if (this->begin.size() < static_cast<size_t>(dims)) this->begin.resize(dims);

    if (this->end.size() < static_cast<size_t>(dims)) this->end.resize(dims);

    if (this->strides.size() < static_cast<size_t>(dims)) this->strides.resize(dims);
    this->begin_mask = 0;
    this->end_mask = 0;
    this->shrink_axis_mask = 0;
    {
      int full_index = 0;

      this->begin_valid = sparse_spec.begin_tensor != nullptr;
      this->end_valid = sparse_spec.end_tensor != nullptr;

      for (int e = 0; e < sparse_spec.dims; e++) {
        if ((1 << e) & sparse_spec.ellipsis_mask) {
          int next_index = sd::math::sd_min<int>(
              this->dims - (sparse_spec.dims - e) + 1 + sparse_spec.num_add_axis_after_ellipsis, this->dims);

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
          if (static_cast<size_t>(full_index) == this->begin.size()) {
            return false;
          }

          // Gather slicing spec into appropriate index
          if (sparse_spec.begin_tensor != nullptr) this->begin[full_index] = sparse_spec.begin_tensor->at(e);

          if (sparse_spec.end_tensor != nullptr) this->end[full_index] = sparse_spec.end_tensor->at(e);

          this->strides[full_index] = sparse_spec.strides_tensor->at(e);

          if (sparse_spec.begin_mask & (1 << e)) this->begin_mask |= (1 << full_index);

          if (sparse_spec.end_mask & (1 << e)) this->end_mask |= (1 << full_index);

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

void vectorize(std::vector<LongType>& input_shape) {
  if (input_shape.size() == 2 && input_shape[0] == 1) {
    int v = input_shape[1];
    input_shape.clear();
    input_shape.emplace_back(v);
  }
}

bool _preprocess_strided_slice(std::vector<sd::LongType>* indicesList, std::vector<sd::LongType>* final_shape,
                               std::vector<sd::LongType>& input_shape, std::vector<sd::LongType>& begin,
                               std::vector<sd::LongType>& end, std::vector<sd::LongType>& strides, int begin_mask, int ellipsis_mask, int end_mask,
                               int new_axis_mask, int shrink_axis_mask, bool* is_identity, bool* is_simple_slice,
                               bool* slice_dim0) {

  // FIX: Check for zero strides and fix them
  bool hasZeroStride = false;
  for (size_t i = 0; i < strides.size(); i++) {
    if (strides[i] == 0) {
      THROW_EXCEPTION("WARNING: Zero stride detected at index %zu, setting to 1\n");
    }
  }

  // FIX: Check if end values are 0 when they shouldn't be
  // For ONNX slice [0:1] on axis 0, end should be 1, not 0
  if (end.size() == 1 && end[0] == 0 && begin.size() == 1 && begin[0] == 0) {
    THROW_EXCEPTION("Invalid bounds for strided slice. Result is empty.");
  }

  std::vector<int> preshape;
  bool ellipsis_seen = false;

  // Special handling for ONNX-style slicing
  bool is_onnx_style_slice = false;
  if (input_shape.size() == 2 && begin.size() == 1 && end.size() == 1 && strides.size() == 1) {
    // This looks like ONNX slice on first dimension only
    is_onnx_style_slice = true;

    // Extend begin/end/strides to cover all dimensions
    // For other dimensions, use full range
    if (begin.size() < input_shape.size()) {
      begin.push_back(0);
      end.push_back(input_shape[1]);
      strides.push_back(1);
      // Update masks to indicate we want full range on second dimension
      begin_mask |= (1 << 1);
      end_mask |= (1 << 1);
    }
  }

  StridedSliceSparseSpec sparse_spec = {(int)strides.size(), 0,        &begin,        &end,          &strides,
                                        begin_mask,          end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask};

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

  StridedSliceDenseSpec dense_spec = {
      (int)input_shape.size(),  // dims
      0,                        // begin_mask
      0,                        // end_mask
      false,                    // begin_valid
      false,                    // end_valid
      begin,                    // begin (reference)
      end,                      // end (reference)
      strides,                  // strides (reference)
      {},                       // final_shape_gather_indices (empty vector)
      0                         // shrink_axis_mask
  };

  // Build the dense spec from sparse spec
  if (!dense_spec.buildDenseSpec(sparse_spec)) {
    return false;
  }

  for (int e = 0; e < (int)input_shape.size(); e++) {
    sd::LongType begin_idx = begin[e];
    sd::LongType end_idx = end[e];
    int stride_idx = strides[e];
    int size_idx = input_shape[e];

    bool shrink_i = (dense_spec.shrink_axis_mask & (1 << e));

    if (size_idx == -1) {
      preshape.emplace_back(shrink_i ? 1 : -1);
      continue;
    }

    const std::array<int, 2> masks = {{dense_spec.begin_mask & (1 << e), dense_spec.end_mask & (1 << e)}};
    const std::array<int, 2> valid_range = {{stride_idx > 0 ? 0 : -1, stride_idx > 0 ? size_idx : size_idx - 1}};

    // Improved canonical function with better bounds checking
    auto canonical = [stride_idx, size_idx, masks, valid_range](sd::LongType x, int c) -> sd::LongType {
      if (masks[c]) {
        return stride_idx > 0 ? valid_range[c] : valid_range[(c + 1) & 1];
      } else {
        sd::LongType x_fwd = x < 0 ? size_idx + x : x;  // make negative indices positive
        // Add bounds checking to prevent invalid indices
        if (stride_idx > 0) {
          x_fwd = sd::math::sd_max<sd::LongType, sd::LongType, sd::LongType>(
              static_cast<sd::LongType>(valid_range[0]),
              sd::math::sd_min<sd::LongType, sd::LongType, sd::LongType>(
                  static_cast<sd::LongType>(valid_range[1]), x_fwd));
        } else {
          x_fwd = sd::math::sd_max<sd::LongType, sd::LongType, sd::LongType>(
              static_cast<sd::LongType>(valid_range[1]),
              sd::math::sd_min<sd::LongType, sd::LongType, sd::LongType>(
                  static_cast<sd::LongType>(valid_range[0]), x_fwd));
        }
        return x_fwd;
      }
    };



    (*is_simple_slice) &= stride_idx == 1;

    const bool begin_and_end_masked = (begin_mask & (1 << e)) && (end_mask & (1 << e));

    if (dense_spec.begin_valid && dense_spec.end_valid) {
      if (shrink_i) {
        int x_fwd = begin_idx < 0 ? size_idx + begin_idx : begin_idx;
        begin_idx = x_fwd;
        end_idx = begin_idx + 1;
        if (x_fwd < 0 || x_fwd >= size_idx) {
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

    // Improved interval calculation and validation
    int interval_length = 1;
    bool known_interval = false;

    if (dense_spec.begin_valid && dense_spec.end_valid) {
      // Ensure begin and end are properly canonicalized
      begin_idx = canonical(begin_idx, 0);
      end_idx = canonical(end_idx, 1);

      interval_length = end_idx - begin_idx;
      known_interval = true;



      // Validate interval based on stride direction
      if (stride_idx > 0) {
        if (interval_length < 0) {
          // For positive stride, if end < begin, treat as empty slice
          interval_length = 0;
        }
      } else if (stride_idx < 0) {
        if (interval_length > 0) {
          // For negative stride, if end > begin, treat as empty slice
          interval_length = 0;
        } else {
          // Make interval positive for calculation
          interval_length = -interval_length;
        }
      }
    } else if (shrink_i) {
      interval_length = 1;
      known_interval = true;
    } else if (begin_and_end_masked) {
      if (size_idx > 0) {
        interval_length = size_idx;
        known_interval = true;
      }
    }

    // Improved size calculation
    if (known_interval) {
      int size_i;

      // Handle empty slices
      if (interval_length == 0) {
        size_i = 0;
      }
        // Handle shrink axis
      else if (shrink_i) {
        size_i = 1;  // Will be removed from final shape later
      }
        // Normal slice calculation
      else if (stride_idx != 0) {
        // Calculate absolute values for size computation
        int abs_interval = interval_length < 0 ? -interval_length : interval_length;
        int abs_stride = stride_idx < 0 ? -stride_idx : stride_idx;

        // Calculate the number of elements in the slice
        size_i = (abs_interval + abs_stride - 1) / abs_stride;  // Ceiling division

        // Ensure non-negative result
        size_i = size_i < 0 ? 0 : size_i;
      } else {
        // This should never happen as we check for zero stride earlier
        THROW_EXCEPTION("ERROR: Zero stride encountered in size calculation for dimension %d\n");
        return false;
      }


      // Update indices list for actual slicing operation
      if (indicesList != nullptr) {
        if (size_i > 0 || shrink_i) {
          indicesList->push_back(begin_idx);
          indicesList->push_back(end_idx);
          indicesList->push_back(stride_idx);
        }
      }

      preshape.emplace_back(size_i);
    } else {
      preshape.emplace_back(-1);
    }
  }

  std::vector<int> * postshape = new std::vector<int>();
  final_shape->clear();
  for (LongType gather_index : dense_spec.final_shape_gather_indices) {
    if (gather_index == kShrinkAxis) {
      // Skip shrink axis dimensions - they are removed from output shape
      continue;
    } else if (gather_index >= 0 && static_cast<size_t>(gather_index) < preshape.size()) {
      final_shape->emplace_back(preshape.at(gather_index));
    } else {
      final_shape->emplace_back(1);
    }
  }

  // Validate generated indices before returning
  if (indicesList && !indicesList->empty()) {
    // Analyze indices in groups of 3
    for (size_t i = 0; i < indicesList->size(); i += 3) {
      if (i + 2 < indicesList->size()) {
        sd::LongType dim_begin = (*indicesList)[i];
        sd::LongType dim_end = (*indicesList)[i + 1];
        sd::LongType dim_stride = (*indicesList)[i + 2];
        size_t dim_idx = i / 3;
      }
    }
  }


  return true;
}

CUSTOM_OP_IMPL(strided_slice, 1, 1, false, 0, 5) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);
  if (z->isEmpty() || z->lengthOf() == 0) {
    return Status::OK;
  }

  int begin_mask = INT_ARG(0);
  int ellipsis_mask = INT_ARG(1);
  int end_mask = INT_ARG(2);
  int new_axis_mask = INT_ARG(3);
  int shrink_axis_mask = INT_ARG(4);

  int dim_values = 0;  // block.getIArguments()->size() - 5;
  int delta = 0;       // dim_values % 3;
  int elements = 0;    // dim_values / 3;

  std::vector<LongType> *begin = new std::vector<LongType>();
  std::vector<LongType> *end = new std::vector<LongType>();
  std::vector<LongType> *strides = new  std::vector<LongType>();

  std::vector<LongType> *args = new std::vector<LongType>();

  // statically evaluated
  if (block.getIArguments()->size() > 5) {
    dim_values = block.getIArguments()->size() - 5;
    delta = dim_values % 3;
    elements = dim_values / 3;

    for (size_t e = 5; e < block.getIArguments()->size(); e++) args->emplace_back(INT_ARG(e));

    REQUIRE_TRUE(delta == 0, 0,
                 "StridedSlice: Number of Integer arguments should be equal to input rank x 3 = %i, but got %i instead",
                 (x->rankOf() * 3), dim_values);

    ShapeUtils::copyVectorPart(*begin, *args, elements, 0);
    ShapeUtils::copyVectorPart(*end, *args, elements, elements);
    ShapeUtils::copyVectorPart(*strides, *args, elements, elements * 2);

  } else if (block.width() > 1) {
    auto v_begin = INPUT_VARIABLE(1);
    auto v_end = INPUT_VARIABLE(2);

    elements = v_begin->lengthOf();

    REQUIRE_TRUE(v_begin->lengthOf() == v_end->lengthOf(), 0,
                 "StridedSlice: Length of begin/end should match, but got %i vs %i instead", v_begin->lengthOf(),
                 v_end->lengthOf());

    for (int e = 0; e < v_begin->lengthOf(); e++) begin->emplace_back(v_begin->e<LongType>(e));

    for (int e = 0; e < v_end->lengthOf(); e++) {
      if(v_end->e<int>(e) < 0) {
        // Special case: -1 means "to the end"
        if(v_end->e<int>(e) == -1) {
          end->emplace_back(x->sizeAt(e));
        } else {
          // Other negative indices: convert to positive
          end->emplace_back(v_end->e<LongType>(e) + x->sizeAt(e));
        }
      } else {
        end->emplace_back(v_end->e<LongType>(e));
      }
    }

    if (block.width() > 3) {
      auto v_stride = INPUT_VARIABLE(3);

      REQUIRE_TRUE(v_stride->lengthOf() == v_begin->lengthOf(), 0,
                   "StridedSlice: Length of begin/end/stride should match, but got %i vs %i vs %i instead",
                   v_begin->lengthOf(), v_end->lengthOf(), v_stride->lengthOf());


      for (int e = 0; e < v_stride->lengthOf(); e++) strides->emplace_back(v_stride->e<LongType>(e));
    } else {
      for (int e = 0; e < v_begin->lengthOf(); e++) strides->emplace_back(1);
    }
  } else {
    REQUIRE_TRUE(false, 0,
                 "StridedSlice: Can't find begin/end/stride information neither in IArguments or in input arrays");
  }

  // validation of begin and start
  std::vector<LongType> ignoreBegin = BitwiseUtils::valueBits(begin_mask);
  std::vector<LongType> ignoreEnd = BitwiseUtils::valueBits(end_mask);
  std::vector<LongType> addAxes = BitwiseUtils::valueBits(new_axis_mask);
  std::vector<LongType> moveAxes = BitwiseUtils::valueBits(shrink_axis_mask);
  if (shrink_axis_mask == 0)
    for (size_t dim = 0, b = 0, e = 0; dim < static_cast<size_t>(x->rankOf()); ++dim) {
      if (moveAxes[dim]) continue;

      if (b < begin->size() && !ignoreBegin[b] && !addAxes[dim]) {
        int first = strides->at(b) > 0 ? begin->at(b) : math::sd_abs<int,int>(begin->at(b)) - 1;
        REQUIRE_TRUE(first <= x->sizeAt(dim), 0,
                     "StridedSlice: begin index should be <= corresponding dimension of input array, but got end_index "
                     "= %i for dimension %i!",
                     begin->at(b), dim);
      }
      if (e < end->size() && !ignoreEnd[e] && !addAxes[dim]) {
        int last = strides->at(e) > 0 ? end->at(e) : math::sd_abs<int,int>(end->at(e)) - 1;
        REQUIRE_TRUE(last <= x->sizeAt(dim), 0,
                     "StridedSlice: end index should be <= corresponding dimension of input array, but got end_index = "
                     "%i for dimension %i!",
                     end->at(e), dim);
      }
      ++b;
      ++e;
    }

  std::vector<LongType> *indices = new std::vector<sd::LongType>();
  auto input_shape = x->getShapeAsVector();
  std::vector<LongType> *final_shape = new std::vector<sd::LongType>();
  bool is_identity;
  bool is_simple_slice;
  bool is_dim0;

  REQUIRE_TRUE(
      _preprocess_strided_slice(indices, final_shape, input_shape, *begin, *end, *strides, begin_mask, ellipsis_mask,
                                end_mask, new_axis_mask, shrink_axis_mask, &is_identity, &is_simple_slice, &is_dim0),
      0, "StridedSlice: shape calculation failed");
  if (indices->size()) {
    LongType* subArrShapeInfo = nullptr;
    ALLOCATE(subArrShapeInfo, block.getWorkspace(), shape::shapeInfoLength(x->rankOf()) * 8, sd::LongType);
    LongType offset;

    shape::calcSubArrShapeInfoAndOffset(indices->data(), x->shapeInfo(), subArrShapeInfo, offset, true, true);
    auto subArrShapeInfoPack = ConstantShapeHelper::getInstance().bufferForShapeInfo(subArrShapeInfo);

    NDArray::prepareSpecialUse({z}, {x});

    NativeOpExecutioner::execTransformAny(block.launchContext(), transform::Assign, x->bufferWithOffset(offset),
                                          subArrShapeInfoPack->primary(), x->specialBufferWithOffset(offset),
                                          subArrShapeInfoPack->special(), z->buffer(), z->shapeInfo(),
                                          z->specialBuffer(), z->specialShapeInfo(), nullptr, true);

    NDArray::registerSpecialUse({z}, {x});

  } else if (!z->isEmpty()) {
    NDArray get = x->e(0);
    z->assign(&get);
  }

  delete indices;
  delete final_shape;
  delete begin;
  delete end;
  delete strides;
  delete args;

  return Status::OK;
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

  //print all masks
  std::vector<LongType> begin;
  std::vector<LongType> end;
  std::vector<LongType> strides;

  // if that's live - shape will be resolved in runtime
  if (block.width() > 1) {
    begin = INPUT_VARIABLE(1)->template asVectorT<LongType>();
    end = INPUT_VARIABLE(2)->template asVectorT<LongType>();
    for(size_t  e = 0; e < end.size(); e++) {
      if(end[e] < 0) {
        // Special case: -1 means "to the end"
        if(end[e] == -1) {
          end[e] = shape::shapeOf(inShape)[e];
        } else {
          end[e] += shape::shapeOf(inShape)[e];
        }
      }
    }

    strides = INPUT_VARIABLE(3)->template asVectorT<LongType>();
  } else if (dim_values > 0) {

    std::vector<LongType> *args = new std::vector<LongType>();
    for (size_t e = 5; e < block.getIArguments()->size(); e++) args->emplace_back(INT_ARG(e));

    // FIXME: probably template required here
    ShapeUtils::copyVectorPart(begin, *args, elements, 0);
    ShapeUtils::copyVectorPart(end, *args, elements, elements);
    ShapeUtils::copyVectorPart(strides, *args, elements, elements * 2);
  }

  REQUIRE_TRUE(begin.size() > 0 && end.size() > 0 && strides.size() > 0, 0, "Strided_Slice: empty arguments");


  std::vector<LongType> *input_shape = new std::vector<LongType>();
  std::vector<LongType> *shape = new std::vector<LongType>();

  auto rank = shape::rank(inShape);
  auto shortShape = shape::shapeOf(inShape);
  for (auto e = 0; e < rank; e++) input_shape->emplace_back(shortShape[e]);

  bool is_identity;
  bool is_simple_slice;
  bool is_dim0;

  std::vector<LongType> *indices = new std::vector<sd::LongType>();
  bool result =
      _preprocess_strided_slice(indices, shape, *input_shape, begin, end, strides, begin_mask, ellipsis_mask, end_mask,
                                new_axis_mask, shrink_axis_mask, &is_identity, &is_simple_slice, &is_dim0);


  if (indices->size()) {
    auto retDtype = block.numD() > 0 ? block.getDArguments()->at(0) : ArrayOptions::dataType(inShape);
    auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(retDtype, 'c', *shape);
    return SHAPELIST(newShape);
  }

  std::vector<LongType> *retShape = new std::vector<sd::LongType>{0};
  return SHAPELIST(ConstantShapeHelper::getInstance().emptyShapeInfoWithShape(ArrayOptions::dataType(inShape),*retShape));
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

  int dim_values = 0;  // block.getIArguments()->size() - 5;
  int delta = 0;       // dim_values % 3;
  int elements = 0;    // dim_values / 3;

  std::vector<LongType> begin;
  std::vector<LongType> end;
  std::vector<LongType> strides;


  std::vector<LongType> args;

  // statically evaluated
  if (block.getIArguments()->size() > 5) {
    dim_values = block.getIArguments()->size() - 5;
    delta = dim_values % 3;
    elements = dim_values / 3;

    for (size_t e = 5; e < block.getIArguments()->size(); e++) args.emplace_back(INT_ARG(e));

    REQUIRE_TRUE(
        delta == 0, 0,
        "StridedSliceBP: Number of Integer arguments should be equal to input rank x 3 = %i, but got %i instead",
        (x->rankOf() * 3), dim_values);

    ShapeUtils::copyVectorPart(begin, args, elements, 0);
    ShapeUtils::copyVectorPart(end, args, elements, elements);
    ShapeUtils::copyVectorPart(strides, args, elements, elements * 2);

  } else if (block.width() >= 3) {

    auto v_begin = INPUT_VARIABLE(2);
    auto v_end = INPUT_VARIABLE(3);

    elements = v_begin->lengthOf();

    REQUIRE_TRUE(v_begin->lengthOf() == v_end->lengthOf(), 0,
                 "StridedSliceBP: Length of begin/end should match, but got %i vs %i instead", (int)v_begin->lengthOf(),
                 (int)v_end->lengthOf());

    for (int e = 0; e < v_begin->lengthOf(); e++) begin.emplace_back(v_begin->e<int>(e));

    for (int e = 0; e < v_end->lengthOf(); e++) {
      if(v_end->e<int>(e) < 0) {
        end.emplace_back(v_end->e<int>(e) + x->sizeAt(e));
      } else {
        end.emplace_back(v_end->e<int>(e));
      }



    }

    if (block.width() >= 4) {
      auto v_stride = INPUT_VARIABLE(4);

      REQUIRE_TRUE(v_stride->lengthOf() == v_begin->lengthOf(), 0,
                   "StridedSliceBP: Length of begin/end/stride should match, but got %i vs %i vs %i instead",
                   (int)v_begin->lengthOf(), (int)v_end->lengthOf(), (int)v_stride->lengthOf());

      for (int e = 0; e < v_stride->lengthOf(); e++) strides.emplace_back(v_stride->e<int>(e));
    } else {
      for (int e = 0; e < v_begin->lengthOf(); e++) strides.emplace_back(1);
    }
  } else {
    REQUIRE_TRUE(false, 0,
                 "StridedSliceBP: Can't find begin/end/stride information neither in IArguments or in input arrays");
  }

  // validation of begin and start
  std::vector<LongType> ignoreBegin = BitwiseUtils::valueBits(begin_mask);
  std::vector<LongType> ignoreEnd = BitwiseUtils::valueBits(end_mask);
  std::vector<LongType> addAxes = BitwiseUtils::valueBits(new_axis_mask);
  std::vector<LongType> moveAxes = BitwiseUtils::valueBits(shrink_axis_mask);

  for (size_t dim = 0, b = 0, e = 0; dim < static_cast<size_t>(x->rankOf()); ++dim) {
    if (moveAxes[dim]) continue;

    if (b < begin.size() && !ignoreBegin[b] && !addAxes[dim]) {
      int first = strides[b] > 0 ? begin[b] : math::sd_abs<int,int>(begin[b]) - 1;
      REQUIRE_TRUE(first <= x->sizeAt(dim), 0,
                   "StridedSlice: begin index should be <= corresponding dimension of input array, but got end_index = "
                   "%i for dimension %i!",
                   begin[b], dim);
    }
    if (e < end.size() && !ignoreEnd[e] && !addAxes[dim]) {
      int last = strides[e] > 0 ? end[e] : math::sd_abs<int,int>(end[e]) - 1;
      REQUIRE_TRUE(last <= x->sizeAt(dim), 0,
                   "StridedSlice: end index should be <= corresponding dimension of input array, but got end_index = "
                   "%i for dimension %i!",
                   end[e], dim);
    }
    ++b;
    ++e;
  }

  auto input_shape = x->getShapeAsVector();
  std::vector<LongType> indices;
  std::vector<LongType> final_shape;
  bool is_identity;
  bool is_simple_slice;
  bool is_dim0;

  // FIXME: remove this method once we get 1D vectors supported
  vectorize(input_shape);
  REQUIRE_TRUE(
      _preprocess_strided_slice(&indices, &final_shape, input_shape, begin, end, strides, begin_mask, ellipsis_mask,
                                end_mask, new_axis_mask, shrink_axis_mask, &is_identity, &is_simple_slice, &is_dim0),
      0, "StridedSliceBP: shape calculation failed");

  output->nullify();
  //
  // the first case: only for scalar gradient step
  if (epsNext->lengthOf() == 1 &&
      ((indices.size() == 3 && (indices[1] - indices[0]) == 1) || (indices[2] - indices[0] == 1))) {
    output->p(indices[0], epsNext);
  } else {  // else for other cases
    auto sub = (*output)(indices, true, true);
    sub.assign(epsNext);
  }

  return Status::OK;
}

DECLARE_SHAPE_FN(strided_slice_bp) {
  auto inShape = inputShape->at(0);
  return SHAPELIST(CONSTANT(inShape));
}

DECLARE_TYPES(strided_slice) { getOpDescriptor()->setAllowedInputTypes(ANY); }

DECLARE_TYPES(strided_slice_bp) {
  getOpDescriptor()->setAllowedInputTypes(ANY);
}
}  // namespace ops
}  // namespace sd

#endif