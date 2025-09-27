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

//
// @author Yurii Shyrma, created on 26.02.2018
//
//
// @author AbdelRauf
//
#include <execution/ThreadPool.h>
#include <execution/Threads.h>
#include <helpers/LoopsCoordsHelper.h>
#include <ops/declarable/helpers/addBias.h>
#include <exceptions/datatype_exception.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <system/selective_rendering.h>
#include <climits>
#include <sstream>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static SD_INLINE void _add(const T* __restrict xx, const T* __restrict yy, T* __restrict zz, const size_t& N) {
  PRAGMA_OMP_SIMD
  for (size_t c = 0; c < N; c++) zz[c] = xx[c] + yy[c];
}

template <typename T>
static SD_INLINE void _add_inplace(T* __restrict xx, const T* __restrict yy, const size_t& N) {
  PRAGMA_OMP_SIMD
  for (size_t c = 0; c < N; c++) xx[c] = xx[c] + yy[c];
}

template <typename T>
static SD_INLINE void _add_broadcast_inplace(T* __restrict xx, const T yy, const size_t& N) {
  PRAGMA_OMP_SIMD
  for (size_t c = 0; c < N; c++) xx[c] = xx[c] + yy;
}

template <typename T>
static SD_INLINE void _add_broadcast(const T* __restrict xx, const T yy, T* __restrict zz, const size_t& N) {
  PRAGMA_OMP_SIMD
  for (size_t c = 0; c < N; c++) zz[c] = xx[c] + yy;
}

static constexpr size_t MIN_NN = 32;
static constexpr size_t MIN_NN_K = 2;

// Helper function to validate shape info
static void validateShapeInfo(const sd::LongType* shapeInfo, const char* tensorName) {
  if (shapeInfo == nullptr) {
    std::stringstream ss;
    ss << "addBias: " << tensorName << " shapeInfo is null";
    THROW_EXCEPTION(ss.str().c_str());
  }
  
  const sd::LongType rank = shapeInfo[0];
  if (rank < 0 || rank > 32) {  // Reasonable upper limit for rank
    std::stringstream ss;
    ss << "addBias: Invalid rank for " << tensorName << ": " << rank;
    THROW_EXCEPTION(ss.str().c_str());
  }
  
  auto bases = &(shapeInfo[1]);
  auto strides = &(shapeInfo[rank + 1]);
  
  // Calculate total number of elements to check for empty tensors
  sd::LongType totalElements = 1;
  bool hasZeroDim = false;
  
  // Validate dimensions - improved logic
  for (sd::LongType i = 0; i < rank; i++) {
    // Check for negative dimensions (invalid)
    if (bases[i] < 0) {
      std::stringstream ss;
      ss << "addBias: Invalid dimension size for " << tensorName 
         << " at index " << i << ": " << bases[i];
      THROW_EXCEPTION(ss.str().c_str());
    }
    
    // Track if we have a zero dimension (empty tensor case)
    if (bases[i] == 0) {
      hasZeroDim = true;
    }
    
    // Check for overflow only for non-zero dimensions
    if (bases[i] > 0 && bases[i] > LLONG_MAX / 8) {
      std::stringstream ss;
      ss << "addBias: Dimension too large for " << tensorName 
         << " at index " << i << ": " << bases[i];
      THROW_EXCEPTION(ss.str().c_str());
    }
    
    totalElements *= bases[i];
  }
  
  // If tensor is empty (has a 0 dimension), log warning but don't throw
  // This allows operations to handle empty tensors gracefully
  if (hasZeroDim) {
    // Empty tensor is valid in many cases (e.g., dynamic batching)
    // Operations should handle this case appropriately
    return;
  }
  
  // Validate strides only for non-empty tensors
  for (sd::LongType i = 0; i < rank; i++) {
    if (strides[i] == 0 && bases[i] > 0) {
      std::stringstream ss;
      ss << "addBias: Invalid stride for " << tensorName 
         << " at index " << i << ": " << strides[i] 
         << " (dimension size: " << bases[i] << ")";
      THROW_EXCEPTION(ss.str().c_str());
    }
  }
}

// Helper function to safely calculate total number of elements
static size_t calculateTotalElements(const sd::LongType* bases, sd::LongType rank) {
  size_t total_num = 1;
  bool hasZeroDim = false;
  
  for (sd::LongType i = 0; i < rank; i++) {
    if (bases[i] == 0) {
      // Early return for empty tensor
      return 0;
    }
    
    // Check for overflow only for non-zero dimensions
    if (bases[i] > 0 && total_num > SIZE_MAX / static_cast<size_t>(bases[i])) {
      THROW_EXCEPTION("addBias: Tensor size overflow when calculating total elements");
    }
    
    total_num *= static_cast<size_t>(bases[i]);
  }
  
  return total_num;
}

// Helper function to validate parallel execution parameters
static void validateParallelParams(sd::LongType start, sd::LongType stop, sd::LongType inc, size_t total_num) {
  if (start < 0) {
    std::stringstream ss;
    ss << "addBias: Invalid start index: " << start;
    THROW_EXCEPTION(ss.str().c_str());
  }
  
  if (stop < start) {
    std::stringstream ss;
    ss << "addBias: Stop index (" << stop << ") is less than start index (" << start << ")";
    THROW_EXCEPTION(ss.str().c_str());
  }
  
  if (inc <= 0) {
    std::stringstream ss;
    ss << "addBias: Invalid increment: " << inc;
    THROW_EXCEPTION(ss.str().c_str());
  }
  
  // Allow stop > total_num for empty tensor case
  if (total_num > 0 && static_cast<size_t>(stop) > total_num) {
    std::stringstream ss;
    ss << "addBias: Stop index (" << stop << ") exceeds total elements (" << total_num << ")";
    THROW_EXCEPTION(ss.str().c_str());
  }
}

template <typename X, typename Y>
static typename std::enable_if<std::is_same<X, Y>::value, const X*>::type flattened_bias(const Y* b_real, X* b_stack,
                                                                                         const size_t b_stack_size,
                                                                                         std::unique_ptr<X[]>& b_heap,
                                                                                         const sd::LongType num,
                                                                                         sd::LongType yStrideC) {
  // Validate inputs
  if (b_real == nullptr) {
    THROW_EXCEPTION("addBias: flattened_bias received null bias pointer");
  }
  
  if (num <= 0) {
    std::stringstream ss;
    ss << "addBias: flattened_bias received invalid num: " << num;
    THROW_EXCEPTION(ss.str().c_str());
  }
  
  if (yStrideC == 0) {
    THROW_EXCEPTION("addBias: flattened_bias received zero stride");
  }
  
  // best results when buffer used much , may result bad perf if buffer is used once
  X* b_new = nullptr;
  if (yStrideC != 1) {
    if (static_cast<size_t>(num)  > b_stack_size) {
      b_heap.reset(new X[num]);
      b_new = b_heap.get();
    } else {
      b_new = b_stack;
    }
    for (size_t i = 0; i < static_cast<size_t>(num) ; i++) {
      b_new[i] = b_real[i * yStrideC];
    }
  } else {
    // no need , just pass normal bias
    return static_cast<const X*>(b_real);
  }
  return const_cast<const X*>(b_new);
}

template <typename X, typename Y>
static typename std::enable_if<!std::is_same<X, Y>::value, const X*>::type flattened_bias(const Y* b_real, X* b_stack,
                                                                                          const size_t b_stack_size,
                                                                                          std::unique_ptr<X[]>& b_heap,
                                                                                          const sd::LongType num,
                                                                                          sd::LongType yStrideC) {
  // Validate inputs
  if (b_real == nullptr) {
    THROW_EXCEPTION("addBias: flattened_bias received null bias pointer");
  }
  
  if (num <= 0) {
    std::stringstream ss;
    ss << "addBias: flattened_bias received invalid num: " << num;
    THROW_EXCEPTION(ss.str().c_str());
  }
  
  if (yStrideC == 0) {
    THROW_EXCEPTION("addBias: flattened_bias received zero stride");
  }
  
  // best results when buffer used much , may result bad perf if buffer is used once
  X* b_new = nullptr;
  if (static_cast<size_t>(num) > b_stack_size) {
    b_heap.reset(new X[num]);
    b_new = b_heap.get();
  } else {
    b_new = b_stack;
  }
  if (yStrideC != 1) {
    for (size_t i = 0; i < static_cast<size_t>(num) ; i++) {
      b_new[i] = static_cast<X>(b_real[i * yStrideC]);
    }
  } else {
    for (size_t i = 0; i < static_cast<size_t>(num) ; i++) {
      b_new[i] = static_cast<X>(b_real[i]);
    }
  }
  return const_cast<const X*>(b_new);
}

template <typename T, size_t constRank>
static void channel_atTheEnd_stride1_C(const sd::LongType*& x_strides, const sd::LongType*& bases, T* x, const T* b,
                                       T* z, const bool& inplace, const sd::LongType& start, const sd::LongType& stop,
                                       const sd::LongType& inc) {
  size_t loop_count = (stop - start) / inc;
  
  // Validate loop count
  if (loop_count == 0) {
    return;  // Nothing to do
  }
  
  sd::CoordsState<constRank - 1> cst;
  size_t offset = sd::init_coords<constRank>(cst, start, bases, x_strides);

  if (!inplace) {
    for (size_t i = 0; i < loop_count; i++) {
      _add(&(x[offset]), b, &(z[offset]), inc);
      offset = sd::inc_coords<constRank - 1>(cst, offset);
    }
  } else {
    for (size_t i = 0; i < loop_count; i++) {
      _add_inplace(&(x[offset]), b, inc);
      offset = sd::inc_coords<constRank - 1>(cst, offset);
    }
  }
}

template <typename T, size_t constRank>
static void channel_atTheEnd_generic_C(const sd::LongType* bases, const sd::LongType* x_strides,
                                       const sd::LongType* z_strides, const bool& inplaceOp, const bool same_stride,
                                       const bool same_order, T* x, const T* b, T* z, sd::LongType start,
                                       sd::LongType stop, sd::LongType inc) {
  // Validate parameters
  if (bases == nullptr || x_strides == nullptr || z_strides == nullptr) {
    THROW_EXCEPTION("addBias: channel_atTheEnd_generic_C received null pointers");
  }
  
  if (x == nullptr || z == nullptr || b == nullptr) {
    THROW_EXCEPTION("addBias: channel_atTheEnd_generic_C received null data pointers");
  }
  
  if (start >= stop) {
    return;  // Nothing to do
  }
  
  // just ensure that passed sameStride is correct,  because when bases are equal orders matters
  bool sameOrderStride = same_order && same_stride;
  if (sameOrderStride && x_strides[constRank - 1] == 1) {
    channel_atTheEnd_stride1_C<T, constRank>(x_strides, bases, x, b, z, inplaceOp, start, stop, inc);
  } else {
    size_t loop_count = (stop - start) / inc;
    
    if (loop_count == 0) {
      return;  // Nothing to do
    }
    
    sd::ZipCoordsState<constRank - 1> cst;
    sd::zip_size_t offset = sd::init_coords<constRank>(cst, start, bases, x_strides, z_strides);
    sd::LongType x_stride = ZIP_STRIDE1(cst, constRank - 1);
    sd::LongType z_stride = ZIP_STRIDE2(cst, constRank - 1);

    if (same_order && x_stride == 1 && z_stride == 1) {
      /* bases are equal with different strides , but the last one is 1. So we can still vectorize it  */
      for (size_t i = 0; i < loop_count; i++) {
        _add(&(x[offset.first]), b, &(z[offset.second]), inc);
        offset = sd::inc_coords<constRank - 1>(cst, offset);
      }
    } else {
      for (size_t i = 0; i < loop_count; i++) {
        T* xx = &(x[offset.first]);
        T* zz = &(z[offset.second]);
        for (size_t j = 0; j < static_cast<size_t>(inc) ; j++) zz[j * z_stride] = xx[j * x_stride] + b[j];
        offset = sd::inc_coords<constRank - 1>(cst, offset);
      }
    }
  }
}

/**
 * this is our main optimization which  benefits from everything for the continuous last_channel C order case
 * as it is intended for full continous we do not need any rank info
 */
template <typename T>
static void channel_atTheEnd_continous_C(T* x, const T* b, T* z, bool inplaceOp, sd::LongType start, sd::LongType stop,
                                         sd::LongType inc) {
  // Validate parameters
  if (x == nullptr || z == nullptr || b == nullptr) {
    THROW_EXCEPTION("addBias: channel_atTheEnd_continous_C received null data pointers");
  }
  
  if (start < 0 || stop < start) {
    std::stringstream ss;
    ss << "addBias: channel_atTheEnd_continous_C invalid range [" << start << ", " << stop << ")";
    THROW_EXCEPTION(ss.str().c_str());
  }
  
  if (inc <= 0) {
    std::stringstream ss;
    ss << "addBias: channel_atTheEnd_continous_C invalid increment: " << inc;
    THROW_EXCEPTION(ss.str().c_str());
  }
  
  sd::LongType  nums = (stop - start);
  sd::LongType  num_inc = nums - nums % inc;
  if (inplaceOp) {
    sd::LongType  offset_p = start;
    for (sd::LongType i = 0; i < num_inc; i += inc) {
      _add_inplace<T>(&(x[offset_p]), b, inc);
      offset_p += inc;
    }
    if (nums > num_inc) _add_inplace<T>(&(x[offset_p]), b, nums - num_inc);
  } else {
    sd::LongType offset_p = start;
    for (sd::LongType i = 0; i < num_inc; i += inc) {
      _add<T>(&(x[offset_p]), b, &(z[offset_p]), inc);
      offset_p += inc;
    }
    if (nums > num_inc) _add<T>(&(x[offset_p]), b, &(z[offset_p]), nums - num_inc);
  }
}

template <typename T, typename T2, size_t constRank>
static void channel_NC_stride1_C(const sd::LongType*& x_strides, const sd::LongType*& bases, T* x, const T2* b, T* z,
                                 const bool& inplace, const sd::LongType yStrideC, const sd::LongType& start,
                                 const sd::LongType& stop, const sd::LongType& inc) {
  // Validate stride
  if (yStrideC == 0) {
    THROW_EXCEPTION("addBias: channel_NC_stride1_C received zero yStrideC");
  }
  
  sd::LongType loop_count = (stop - start) / inc;
  
  if (loop_count <= 0) {
    return;  // Nothing to do
  }
  
  sd::CoordsState<constRank - 1> cst;
  sd::LongType offset = sd::init_coords<constRank>(cst, start, bases, x_strides);

  if (!inplace) {
    for (sd::LongType i = 0; i < loop_count; i++) {
      T yy = static_cast<T>(b[COORDS(cst, 1) * yStrideC]);
      _add_broadcast(&(x[offset]), yy, &(z[offset]), inc);
      offset = sd::inc_coords<constRank - 1>(cst, offset);
    }
  } else {
    for (sd::LongType i = 0; i < loop_count; i++) {
      T yy = static_cast<T>(b[COORDS(cst, 1) * yStrideC]);
      _add_broadcast_inplace(&(x[offset]), yy, inc);
      offset = sd::inc_coords<constRank - 1>(cst, offset);
    }
  }
}

template <typename T, typename T2, size_t constRank>
static void channel_NC_generic_C(const sd::LongType* bases, const sd::LongType* x_strides,
                                 const sd::LongType* z_strides, const bool& inplaceOp, const bool same_stride,
                                 const bool same_order, const sd::LongType yStrideC, T* x, const T2* b, T* z,
                                 sd::LongType start, sd::LongType stop, sd::LongType inc) {
  // Validate parameters
  if (bases == nullptr || x_strides == nullptr || z_strides == nullptr) {
    THROW_EXCEPTION("addBias: channel_NC_generic_C received null pointers");
  }
  
  if (x == nullptr || z == nullptr || b == nullptr) {
    THROW_EXCEPTION("addBias: channel_NC_generic_C received null data pointers");
  }
  
  if (yStrideC == 0) {
    THROW_EXCEPTION("addBias: channel_NC_generic_C received zero yStrideC");
  }
  
  if (start >= stop) {
    return;  // Nothing to do
  }
  
  // just ensure that passed sameStride is correct,  because when bases are equal orders matters
  bool sameOrderStride = same_order && same_stride;

  if (sameOrderStride && x_strides[constRank - 1] == 1) {
    channel_NC_stride1_C<T, T2, constRank>(x_strides, bases, x, b, z, inplaceOp, yStrideC, start, stop, inc);
  } else {
    // (stop-start) % inc == 0 because  we  handled inside partitioning using the channel size
    size_t loop_count = (stop - start) / inc;
    
    if (loop_count == 0) {
      return;  // Nothing to do
    }
    
    sd::ZipCoordsState<constRank - 1> cst;
    sd::zip_size_t offset = sd::init_coords<constRank>(cst, start, bases, x_strides, z_strides);
    sd::LongType x_stride = ZIP_STRIDE1(cst, constRank - 1);
    sd::LongType z_stride = ZIP_STRIDE2(cst, constRank - 1);
    if (same_order && z_stride == 1 && x_stride == 1) {
      /* bases are equal with different strides , but the last one is 1. So we can still vectorize it  */
      for (size_t i = 0; i < loop_count; i++) {
        T yy = static_cast<T>(b[ZIP_COORDS(cst, 1) * yStrideC]);
        _add_broadcast(&(x[offset.first]), yy, &(z[offset.second]), inc);
        offset = sd::inc_coords<constRank - 1>(cst, offset);
      }
    } else {
      for (size_t i = 0; i < loop_count; i++) {
        T* xx = &(x[offset.first]);
        T* zz = &(z[offset.second]);
        T yy = static_cast<T>(b[ZIP_COORDS(cst, 1) * yStrideC]);
        for (sd::LongType j = 0; j < inc; j++) zz[j * z_stride] = xx[j * x_stride] + yy;
        offset = sd::inc_coords<constRank - 1>(cst, offset);
      }
    }
  }
}

///
template <typename T, typename T2>
static void channel_NC_continous_numHW_C(sd::LongType rank, const sd::LongType* bases, const sd::LongType* x_strides,
                                         T* x, const T2* b, T* z, bool inplaceOp, const sd::LongType yStrideC,
                                         sd::LongType start, sd::LongType stop, sd::LongType inc) {
  // Validate parameters
  if (bases == nullptr || x_strides == nullptr) {
    THROW_EXCEPTION("addBias: channel_NC_continous_numHW_C received null pointers");
  }
  
  if (x == nullptr || z == nullptr || b == nullptr) {
    THROW_EXCEPTION("addBias: channel_NC_continous_numHW_C received null data pointers");
  }
  
  if (yStrideC == 0) {
    THROW_EXCEPTION("addBias: channel_NC_continous_numHW_C received zero yStrideC");
  }
  
  if (inc <= 0) {
    std::stringstream ss;
    ss << "addBias: channel_NC_continous_numHW_C invalid increment: " << inc;
    THROW_EXCEPTION(ss.str().c_str());
  }
  
  if (start >= stop) {
    return;  // Nothing to do
  }
  
  // (stop-start) % inc == 0 because  we  handled inside partitioning using the channel size
  size_t loop_count = (stop - start) / inc;
  
  if (loop_count == 0) {
    return;  // Nothing to do
  }

  sd::CoordsState<1> cst;
  // note: we had to manually pass index
  sd::LongType offset_p = sd::init_coords<2>(cst, start / inc, bases, x_strides);

  // partitioning was done using numHW, so we can increment from rank 2
  if (inplaceOp) {
    for (size_t i = 0; i < loop_count; i++) {
      T yy = static_cast<T>(b[COORDS(cst, 1) * yStrideC]);
      _add_broadcast_inplace(&(x[offset_p]), yy, inc);
      offset_p = sd::inc_coords<2>(cst, offset_p);
    }
  } else {
    if (yStrideC == 1) {
      for (size_t i = 0; i < loop_count; i++) {
        T yy = static_cast<T>(b[COORDS(cst, 1)]);
        _add_broadcast(&(x[offset_p]), yy, &(z[offset_p]), inc);
        offset_p = sd::inc_coords<2>(cst, offset_p);
      }
    } else {
      for (size_t i = 0; i < loop_count; i++) {
        T yy = static_cast<T>(b[COORDS(cst, 1) * yStrideC]);
        _add_broadcast(&(x[offset_p]), yy, &(z[offset_p]), inc);
        offset_p = sd::inc_coords<2>(cst, offset_p);
      }
    }
  }
}

//
template <typename T, typename T2, size_t constRank, size_t b_index, size_t skip>
static void channel_generic_stride_skip_F( sd::LongType*& x_strides,  sd::LongType*& bases, T* x, const T2* b,
                                           T* z, const bool& inplace, const sd::LongType yStrideC,
                                           const sd::LongType& start, const sd::LongType& stop,
                                           const sd::LongType& inc) {
  // Validate parameters
  if (yStrideC == 0) {
    THROW_EXCEPTION("addBias: channel_generic_stride_skip_F received zero yStrideC");
  }
  
  if (start >= stop) {
    return;  // Nothing to do
  }
  
  // (stop-start) % inc == 0 because  we  handled inside partitioning using the channel size
  sd::LongType loop_count = (stop - start) / inc;
  
  if (loop_count <= 0) {
    return;  // Nothing to do
  }
  
  sd::CoordsState<constRank - 1> cst;
  sd::LongType offset_p = sd::init_coords<constRank, 0, false>(cst, start, bases, x_strides);
  if (!inplace) {
    for (sd::LongType i = 0; i < loop_count; i++) {
      T yy = static_cast<T>(b[COORDS(cst, b_index) * yStrideC]);
      _add_broadcast(&(x[offset_p]), yy, &(z[offset_p]), inc);
      offset_p = sd::inc_coords<constRank, skip, false>(cst, offset_p);
    }
  } else {
    for (sd::LongType i = 0; i < loop_count; i++) {
      T yy = static_cast<T>(b[COORDS(cst, b_index) * yStrideC]);
      _add_broadcast_inplace(&(x[offset_p]), yy, inc);
      offset_p = sd::inc_coords<constRank, skip, false>(cst, offset_p);
    }
  }
}

///
template <typename T, typename T2, size_t constRank, size_t b_index>
static void channel_generic_F( sd::LongType* bases,  sd::LongType* x_strides,  sd::LongType* z_strides,
                               const bool& inplaceOp, const bool same_stride, const bool same_order,
                               sd::LongType yStrideC, T* x, const T2* b, T* z, sd::LongType start,
                               sd::LongType stop, sd::LongType inc) {
  // Validate parameters
  if (bases == nullptr || x_strides == nullptr || z_strides == nullptr) {
    THROW_EXCEPTION("addBias: channel_generic_F received null pointers");
  }
  
  if (x == nullptr || z == nullptr || b == nullptr) {
    THROW_EXCEPTION("addBias: channel_generic_F received null data pointers");
  }
  
  if (yStrideC == 0) {
    THROW_EXCEPTION("addBias: channel_generic_F received zero yStrideC");
  }
  
  if (start >= stop) {
    return;  // Nothing to do
  }
  
  // just ensure that passed sameStride is correct,  because when bases are equal orders matters
  bool sameOrderStride = same_order && same_stride;
  if (sameOrderStride && x_strides[0] == 1) {
    channel_generic_stride_skip_F<T, T2, constRank, b_index, 1>(x_strides, bases, x, b, z, inplaceOp, yStrideC, start,
                                                                stop, inc);
  } else {
    // (stop-start) % inc == 0 because  we  handled inside partitioning using the channel size
    sd::LongType loop_count = (stop - start) / inc;
    
    if (loop_count <= 0) {
      return;  // Nothing to do
    }
    
    sd::ZipCoordsState<constRank - 1> cst;
    sd::zip_size_t offset = sd::init_coords<constRank, 0, false>(cst, start, bases, x_strides, z_strides);
    sd::LongType x_stride = ZIP_STRIDE1(cst, 0);
    sd::LongType z_stride = ZIP_STRIDE2(cst, 0);
    if (same_order && z_stride == 1 && x_stride == 1) {
      for (size_t i = 0; i < static_cast<size_t>(loop_count) ; i++) {
        T yy = static_cast<T>(b[ZIP_COORDS(cst, b_index) * yStrideC]);
        _add_broadcast(&(x[offset.first]), yy, &(z[offset.second]), inc);
        offset = sd::inc_coords<constRank, 1, false>(cst, offset);
      }
    } else {
      for (sd::LongType i = 0; i < loop_count; i++) {
        T* xx = &(x[offset.first]);
        T* zz = &(z[offset.second]);
        T yy = static_cast<T>(b[ZIP_COORDS(cst, b_index) * yStrideC]);
        for (size_t j = 0; j < static_cast<size_t>(inc) ; j++) zz[j * z_stride] = xx[j * x_stride] + yy;
        offset = sd::inc_coords<constRank, 1, false>(cst, offset);
      }
    }
  }
}

template <typename X, typename Y>
static void addBias_(NDArray& input, NDArray& bias, NDArray& output, const bool isNCHW) {
  // Early return for empty tensors
  if (input.isEmpty() || output.isEmpty()) {
    // Empty tensor - nothing to do, this is valid for dynamic batching
    return;
  }
  
  // Input validation for non-empty case
  if (!input.bufferAsT<X>()) {
    THROW_EXCEPTION("addBias: Input buffer is null");
  }
  
  if (!output.bufferAsT<X>()) {
    THROW_EXCEPTION("addBias: Output buffer is null");
  }
  
  if (!bias.bufferAsT<Y>()) {
    THROW_EXCEPTION("addBias: Bias buffer is null");
  }
  
  auto x_shapeInfo = input.shapeInfo();
  auto z_shapeInfo = output.shapeInfo();
  
  // Validate shape info with improved validation
  validateShapeInfo(x_shapeInfo, "input");
  validateShapeInfo(z_shapeInfo, "output");
  
  auto x = input.bufferAsT<X>();
  auto z = output.bufferAsT<X>();
  auto b = bias.bufferAsT<Y>();
  
  const sd::LongType rank = x_shapeInfo[0];
  
  // Validate rank consistency
  if (rank != z_shapeInfo[0]) {
    std::stringstream ss;
    ss << "addBias: Input and output ranks don't match: " << rank << " vs " << z_shapeInfo[0];
    THROW_EXCEPTION(ss.str().c_str());
  }
  
  auto bases = &(x_shapeInfo[1]);
  auto x_strides = &(x_shapeInfo[rank + 1]);
  auto z_strides = &(z_shapeInfo[rank + 1]);
  
  // Check if tensor is actually empty (has 0 in any dimension)
  bool isEmptyTensor = false;
  for (sd::LongType i = 0; i < rank; i++) {
    if (bases[i] == 0) {
      isEmptyTensor = true;
      break;
    }
  }
  
  // If empty tensor, return early - this is valid for dynamic batching
  if (isEmptyTensor) {
    return;
  }
  
  // Validate shapes match for non-empty tensors
  for (sd::LongType i = 0; i < rank; i++) {
    if (bases[i] != z_shapeInfo[i + 1]) {
      std::stringstream ss;
      ss << "addBias: Input and output shapes don't match at dimension " << i 
         << ": " << bases[i] << " vs " << z_shapeInfo[i + 1];
      THROW_EXCEPTION(ss.str().c_str());
    }
  }
  
  const bool inplaceOp = (x == z);
  const bool same_order = inplaceOp || (input.ordering() == output.ordering());
  const bool channel_atTheEnd = !isNCHW;
  const bool same_stride = inplaceOp || shape::strideEquals(x_shapeInfo, z_shapeInfo);
  bool isContinuous = false;
  
  sd::LongType posOfNonUnityDim;
  bias.isCommonVector(posOfNonUnityDim);
  
  if (posOfNonUnityDim < 0 || posOfNonUnityDim >= bias.rankOf()) {
    // If bias is scalar or has all dimensions as 1, default to position 0
    posOfNonUnityDim = 0;
  }
  
  const sd::LongType yStrideC = bias.strideAt(posOfNonUnityDim);
  
  if (yStrideC == 0 && bias.lengthOf() > 1) {
    std::stringstream ss;
    ss << "addBias: Bias stride is zero at dimension " << posOfNonUnityDim 
       << " for non-scalar bias";
    THROW_EXCEPTION(ss.str().c_str());
  }
  
  char order = input.ordering();

  // for rank>5
  if (rank > 5) {
    const sd::LongType channelDim = isNCHW ? 1 : input.rankOf() - 1;  // second or last
    std::vector<sd::LongType> channelDimVec = {channelDim};
    const_cast<NDArray&>(input).applyBroadcast(sd::broadcast::Add,&channelDimVec , &bias, &output);
    return;
  }

  if (same_order && same_stride) {
    isContinuous = shape::elementWiseStride(x_shapeInfo) == 1 && shape::elementWiseStride(z_shapeInfo) == 1;
  }

  bool treat_as_lastC = false;
  
  if (rank == 2 && isNCHW) {
    // we believe we better treat it as channel at the end case;
    treat_as_lastC = true;
  }
  
  if (channel_atTheEnd || treat_as_lastC) {
    // N..HWC case here
    // flattened bias variables
    constexpr size_t BSIZE1 = 3 * MIN_NN * MIN_NN;
    constexpr size_t BSIZE2 = BSIZE1 + MIN_NN * MIN_NN;
    X flatBias_stack[BSIZE2] SD_ALIGN32;
    std::unique_ptr<X[]> flatBias_heap;
    const X* bias_new;
    X* bias_extra = nullptr;
    
    size_t total_num = calculateTotalElements(bases, rank);
    
    // Check again for empty tensor after calculation
    if (total_num == 0) {
      return;  // Empty tensor, nothing to do
    }
    
    size_t inc;
    size_t rank_skip = 1;
    if (order == 'c') {
      size_t b_stack_size = BSIZE2;
      inc = bases[rank - 1];
      
      if (inc <= 0) {
        // This shouldn't happen after our checks, but be defensive
        return;
      }
      
      if (isContinuous) {
        // for continous we need extra stack memory
        // to create vectorizable bias from small size
        b_stack_size = BSIZE1;
        bias_extra = &(flatBias_stack[BSIZE1]);
      }
      bias_new = flattened_bias(b, (X*)flatBias_stack, b_stack_size, flatBias_heap, inc, yStrideC);
      if (isContinuous && inc < MIN_NN_K * MIN_NN && total_num > inc * MIN_NN_K) {
        // for small size where total_num is sufficient  we need to recreate vectorizable buffer
        sd::LongType old_inc = inc;
        // sizeof bias_extra is MIN_NN * MIN_NN
        size_t new_inc = inc < MIN_NN ? inc * MIN_NN : inc * MIN_NN / MIN_NN_K;
        // if there is a room then lets multiply
        new_inc =
            (new_inc * MIN_NN_K <= total_num && new_inc < MIN_NN * MIN_NN / MIN_NN_K) ? MIN_NN_K * new_inc : new_inc;
        
        if (new_inc > MIN_NN * MIN_NN) {
          THROW_EXCEPTION("addBias: Buffer size exceeded for bias vectorization");
        }
        
        for (size_t i = 0; i < new_inc; i += inc) {
          // copy to our buffer
          X* cp = &(bias_extra[i]);
          for (size_t j = 0; j < inc; j++) {
            cp[j] = bias_new[j];
          }
        }
        // vectorizable buffer
        inc = new_inc;
        bias_new = bias_extra;
      }
    } else {
      inc = bases[0];
      
      if (inc <= 0) {
        // This shouldn't happen after our checks, but be defensive
        return;
      }
      
      if (isContinuous) {
        // we can choose other inc and index for that case
        // but for now lets choose all till the last one
        sd::LongType req_numThreads = sd::Environment::getInstance().maxMasterThreads();
        isContinuous = false;
        if (rank > 2) {
          if (req_numThreads < 2 || bases[rank - 1] >= req_numThreads) {
            if (bases[rank - 1] > 0) {
              inc = total_num / bases[rank - 1];
              isContinuous = true;
              rank_skip = rank - 1;
            }
          } else if (rank > 3 && bases[rank - 1] * bases[rank - 2] >= req_numThreads) {
            if (bases[rank - 1] > 0 && bases[rank - 2] > 0) {
              inc = total_num / bases[rank - 1] / bases[rank - 2];  // for continuous case it is its stride
              rank_skip = rank - 2;
              isContinuous = true;
            }
          }
        }
      }
    }

    // Final validation before parallel execution
    validateParallelParams(0, total_num, inc, total_num);

    FUNC_1D func = [order, isContinuous, rank, x, b, bias_new, z, x_shapeInfo, z_shapeInfo, same_stride, same_order,
        yStrideC, rank_skip](uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) -> void {
      auto bases = &(x_shapeInfo[1]);
      auto x_strides = &(x_shapeInfo[rank + 1]);
      auto z_strides = &(z_shapeInfo[rank + 1]);
      const bool inplaceOp = (x == z);
      if (order == 'c') {
        if (isContinuous) {
          channel_atTheEnd_continous_C(const_cast<X*>(x), bias_new, z, inplaceOp, start, stop, increment);
        }
          // rank is in [2,5]
        else if (rank == 4) {
          channel_atTheEnd_generic_C<X, 4>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order,
                                           const_cast<X*>(x), bias_new, z, start, stop, increment);

        } else if (rank == 5) {
          channel_atTheEnd_generic_C<X, 5>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order,
                                           const_cast<X*>(x), bias_new, z, start, stop, increment);
        } else if (rank == 2) {
          channel_atTheEnd_generic_C<X, 2>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order,
                                           const_cast<X*>(x), bias_new, z, start, stop, increment);
        } else if (rank == 3) {
          channel_atTheEnd_generic_C<X, 3>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order,
                                           const_cast<X*>(x), bias_new, z, start, stop, increment);
        }
      } else {
        // generic F case
        if (isContinuous) {
          if (rank == 4) {
            if (rank_skip == static_cast<size_t>(rank) - 2) {
              channel_generic_stride_skip_F<X, Y, 4, 3, 2>(x_strides, bases, const_cast<X*>(x), b, z, inplaceOp,
                                                           yStrideC, start, stop, increment);
            } else {
              channel_generic_stride_skip_F<X, Y, 4, 3, 3>(x_strides, bases, const_cast<X*>(x), b, z, inplaceOp,
                                                           yStrideC, start, stop, increment);
            }
          } else if (rank == 5) {
            if (static_cast<size_t>(rank_skip)  == static_cast<size_t>(rank)  - 2) {
              // skip==3
              channel_generic_stride_skip_F<X, Y, 5, 4, 3>(x_strides, bases, const_cast<X*>(x), b, z, inplaceOp,
                                                           yStrideC, start, stop, increment);
            } else {
              channel_generic_stride_skip_F<X, Y, 5, 4, 4>(x_strides, bases, const_cast<X*>(x), b, z, inplaceOp,
                                                           yStrideC, start, stop, increment);
            }
          } else if (rank == 3) {
            channel_generic_stride_skip_F<X, Y, 3, 2, 2>(x_strides, bases, const_cast<X*>(x), b, z, inplaceOp, yStrideC,
                                                         start, stop, increment);
          }
        } else if (rank == 4) {
          channel_generic_F<X, Y, 4, 3>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC,
                                        const_cast<X*>(x), b, z, start, stop, increment);
        } else if (rank == 5) {
          channel_generic_F<X, Y, 5, 4>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC,
                                        const_cast<X*>(x), b, z, start, stop, increment);
        } else if (rank == 2) {
          channel_generic_F<X, Y, 2, 1>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC,
                                        const_cast<X*>(x), b, z, start, stop, increment);
        } else if (rank == 3) {
          channel_generic_F<X, Y, 3, 2>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC,
                                        const_cast<X*>(x), b, z, start, stop, increment);
        }
      }
    };
    //
    samediff::Threads::parallel_aligned_increment(func, 0, total_num, inc);
  } else {
    // NC...HW case here
    size_t numNC = 1;
    size_t numHW = 1;
    
    // Validate dimensions for NCHW
    if (rank < 2) {
      std::stringstream ss;
      ss << "addBias: NCHW format requires rank >= 2, got " << rank;
      THROW_EXCEPTION(ss.str().c_str());
    }
    
    // Check for empty dimensions in NCHW case
    for (sd::LongType i = 0; i < 2; i++) {
      if (bases[i] == 0) {
        return;  // Empty tensor, nothing to do
      }
      if (numNC > SIZE_MAX / static_cast<size_t>(bases[i])) {
        THROW_EXCEPTION("addBias: NC dimensions overflow");
      }
      numNC *= bases[i];
    }
    for (sd::LongType i = 2; i < rank; i++) {
      if (bases[i] == 0) {
        return;  // Empty tensor, nothing to do
      }
      if (numHW > SIZE_MAX / static_cast<size_t>(bases[i])) {
        THROW_EXCEPTION("addBias: HW dimensions overflow");
      }
      numHW *= bases[i];
    }
    
    sd::LongType total_num = numNC * numHW;
    
    if (total_num <= 0) {
      // Empty or invalid tensor
      return;
    }
    
    sd::LongType inc = (order == 'c') ? bases[rank - 1] : bases[0];
    
    if (inc <= 0) {
      // Empty dimension, nothing to do
      return;
    }
    
    if (order == 'c' && isContinuous) {
      // sometimes last dimension is too big and multithreading could suffer using unfair partitioning
      // so we will do it only when inc is smaller our value or multithreading turned off
      sd::LongType req_numThreads = sd::Environment::getInstance().maxMasterThreads();
      if (req_numThreads < 2 || numNC >= static_cast<size_t>(req_numThreads)  || inc <= 2 * 8196 || rank == 3) {
        inc = numHW;
      } else {
        // treat it as stride1c case
        isContinuous = false;
      }
    }
    
    // Final validation before parallel execution
    validateParallelParams(0, total_num, inc, total_num);
    
    FUNC_1D func = [order, isContinuous, rank, x, b, z, x_shapeInfo, z_shapeInfo, same_stride, same_order, yStrideC](
        uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) -> void {
      sd::LongType* bases = &(x_shapeInfo[1]);
      sd::LongType* x_strides = &(x_shapeInfo[rank + 1]);
      sd::LongType* z_strides = &(z_shapeInfo[rank + 1]);
      const bool inplaceOp = (x == z);
      if (order == 'c') {
        if (isContinuous) {
          channel_NC_continous_numHW_C<X, Y>(rank, bases, x_strides, const_cast<X*>(x), b, z, inplaceOp, yStrideC,
                                             start, stop, increment);
        }
          // rank is in [3,5]
        else if (rank == 4) {
          channel_NC_generic_C<X, Y, 4>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC,
                                        const_cast<X*>(x), b, z, start, stop, increment);

        } else if (rank == 5) {
          channel_NC_generic_C<X, Y, 5>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC,
                                        const_cast<X*>(x), b, z, start, stop, increment);
        } else if (rank == 3) {
          channel_NC_generic_C<X, Y, 3>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC,
                                        const_cast<X*>(x), b, z, start, stop, increment);
        }
      } else {
        // the same can be applied for NCHW case
        // generic F case
        // continuous case is missing

        if (rank == 4) {
          channel_generic_F<X, Y, 4, 1>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC,
                                        const_cast<X*>(x), b, z, start, stop, increment);
        } else if (rank == 5) {
          channel_generic_F<X, Y, 5, 1>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC,
                                        const_cast<X*>(x), b, z, start, stop, increment);
        } else if (rank == 3) {
          channel_generic_F<X, Y, 3, 1>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC,
                                        const_cast<X*>(x), b, z, start, stop, increment);
        }
      }
    };
    //
    samediff::Threads::parallel_aligned_increment(func, 0, total_num, inc);
  }
}
//////////////////////////////////////////////////////////////////////////
void addBias(sd::graph::Context& block, NDArray& input, NDArray& bias, NDArray& output, const bool isNCHW) {
  // Early return for empty tensors - this is valid for dynamic batching
  if (input.isEmpty() || output.isEmpty()) {
    return;
  }
  
  // Input validation for non-empty case
  if (bias.isEmpty()) {
    THROW_EXCEPTION("addBias: Bias array is empty");
  }
  
  // Shape validation
  if (!input.isSameShape(output)) {
    THROW_EXCEPTION("addBias: Input and output shapes don't match");
  }
  
  auto inputDType = input.dataType();
  auto biasDType = bias.dataType();
  
  BUILD_DOUBLE_SELECTOR(input.dataType(), bias.dataType(), addBias_, (input, bias, output, isNCHW), SD_FLOAT_TYPES,
                        SD_FLOAT_TYPES);
}

BUILD_DOUBLE_TEMPLATE( void addBias_,
                      (NDArray& input, NDArray& bias, NDArray& output, const bool isNCHW), SD_FLOAT_TYPES,
                      SD_FLOAT_TYPES);
}  // namespace helpers
}  // namespace ops
}  // namespace sd
