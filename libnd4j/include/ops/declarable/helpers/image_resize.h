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
//  @author sgazeos@gmail.com
//
#ifndef __IMAGE_RESIZE_HELPERS__
#define __IMAGE_RESIZE_HELPERS__
#include <array/NDArray.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * ResizeBilinear: Bilinear interpolation. If 'antialias' is true, becomes a hat/tent filter function with radius 1 when
 * downsampling. ResizeLanczos5: Lanczos kernel with radius 5. Very-high-quality filter but may have stronger ringing.
 * ResizeBicubic: Cubic interpolant of Keys. Equivalent to Catmull-Rom kernel. Reasonably good quality and faster than
 * Lanczos3Kernel, particularly when upsampling. ResizeGaussian: Gaussian kernel with radius 3, sigma = 1.5 / 3.0.
 * ResizeNearest: Nearest neighbor interpolation. 'antialias' has no effect when used with nearest neighbor
 * interpolation. ResizeArea: Anti-aliased resampling with area interpolation. 'antialias' has no effect when used with
 * area interpolation; it always anti-aliases. ResizeMitchellcubic: Mitchell-Netravali Cubic non-interpolating filter.
 * For synthetic images (especially those lacking proper prefiltering), less ringing than Keys cubic kernel but less
 * sharp.
 */
enum ImageResizeMethods {
  kResizeBilinear = 0,
  kResizeNearest = 1,
  kResizeBicubic = 2,
  kResizeArea = 3,
  kResizeGaussian = 4,
  kResizeLanczos3 = 5,
  kResizeLanczos5 = 6,
  kResizeMitchellcubic = 7,
  kResizeFirst = kResizeBilinear,
  kResizeLast = kResizeMitchellcubic,
  kResizeOldLast = kResizeArea
};

/**
 * Effective only for the ResizeNearest interpolation.
 * Indicates how to get "nearest" pixel in NDArray from original coordinate
 * FLOOR = the largest integer value not greater than
 * ROUND_PREFER_FLOOR = round half down
 * ROUND_PREFER_CEIL = round half up
 * CEIL =  nearest integer not less than
 */
enum NearestMode {
  FLOOR = 0,
  ROUND_PREFER_FLOOR = 1,
  ROUND_PREFER_CEIL = 2,
  CEIL = 3,
};

/**
 * Transformation function of the coordinate in the resized NdArray to the coordinate in the original NdArray
 * ASYMMETRIC original = resized * inv_scale
 * HALF_PIXEL original = (resized + 0.5) * inv_scale - 0.5
 * HALF_PIXEL_NN original = (resized + 0.5) * inv_scale  It is used to retain old behaviour in ResizeNearest
 */
enum CoordinateTransformationMode {
  ASYMMETRIC = 0,  // LegacyScaler
  HALF_PIXEL = 1,
  HALF_PIXEL_NN = 2
};

#if !defined(__CUDACC__)
// An interface for integrated scale functors.
template <typename T = float>
struct IKernelFunc {
  virtual T operator()(T x) const = 0;
  virtual T radius() const = 0;
  // see: https://stackoverflow.com/questions/41552966/getting-new-delete-type-mismatch-from-asan
  virtual ~IKernelFunc() = default;
};
#endif

template <typename T = float>
struct KeysCubicKernelFunc
#if !defined(__CUDACC__)
    : public IKernelFunc<T>
#endif
{
  // http://ieeexplore.ieee.org/document/1163711/
  // R. G. Keys. Cubic convolution interpolation for digital image
  // processing. IEEE Transactions on Acoustics, Speech, and Signal
  // Processing, 29(6):1153–1160, 1981.

  static constexpr T KEYS_CUBIC_COEF = static_cast<T>(-0.5);
  static constexpr T ORDINARY_COEF = static_cast<T>(-0.75);

  SD_HOST_DEVICE KeysCubicKernelFunc() : _coef(KEYS_CUBIC_COEF) {}

  SD_HOST_DEVICE KeysCubicKernelFunc(T coef) : _coef(coef) {}

  SD_INLINE SD_HOST_DEVICE T calc_less2pt0(T x) const {
    // original: coef*|s|^3-5*coef*|s|^2+8*coef*|s| - 4coef
    // => ( (coef*|s|-5*coef)*|s|)+8*coef)*|s| - 4coef
    return ((_coef * x - T(5) * _coef) * x + T(8) * _coef) * x - T(4) * _coef;
  }

  SD_INLINE SD_HOST_DEVICE T calc_less1pt0(T x) const {
    // original: (coef+2)*|s|^3-(coef+3)*|s|^2 + 1
    // =>((coef + 2) * |s| - (coef + 3)) * |s| * |s| + 1
    return ((_coef + T(2)) * x - (_coef + T(3))) * x * x + T(1);
  }

  SD_HOST_DEVICE T operator()(T s) const {
    auto abs_s = math::sd_abs(s);
    if (abs_s >= T(2)) {
      return T(0.0);
    } else if (abs_s >= T(1)) {
      return calc_less2pt0(abs_s);
    } else {
      return calc_less1pt0(abs_s);
    }
  }

  SD_HOST_DEVICE T radius() const { return T(2); }

  T _coef = KEYS_CUBIC_COEF;
  // see: https://stackoverflow.com/questions/41552966/getting-new-delete-type-mismatch-from-asan
  virtual ~KeysCubicKernelFunc() = default;
};

struct LanczosKernelFunc
#if !defined(__CUDACC__)
    : public IKernelFunc<float>
#endif
{
  // Pass 1 for Lanczos1 kernel, 3 for Lanczos3 etc.
  explicit LanczosKernelFunc(float const radius) : _radius(radius) {}
  SD_HOST_DEVICE float operator()(float x) const {
    float const kPI = 3.141592653589793f;
    x = math::sd_abs(x);
    if (x > _radius) return 0.f;
    // Need to special case the limit case of sin(x) / x when x is zero.
    if (x <= 1.e-3f) {
      return 1.f;
    }
    return _radius * std::sin(kPI * x) * std::sin(kPI * x / _radius) / (kPI * kPI * x * x);
  }
  SD_HOST_DEVICE float radius() const { return _radius; }
  const float _radius;
  // see: https://stackoverflow.com/questions/41552966/getting-new-delete-type-mismatch-from-asan
  virtual ~LanczosKernelFunc() = default;
};

struct GaussianKernelFunc
#if !defined(__CUDACC__)
    : public IKernelFunc<float>
#endif
{
  static constexpr float kRadiusMultiplier = 3.0f;
  // https://en.wikipedia.org/wiki/Gaussian_function
  // We use sigma = 0.5, as suggested on p. 4 of Ken Turkowski's "Filters
  // for Common Resampling Tasks" for kernels with a support of 3 pixels:
  // www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
  // This implies a radius of 1.5,
  explicit GaussianKernelFunc(float radius = 1.5f) : _radius(radius), _sigma(radius / kRadiusMultiplier) {}
  SD_HOST_DEVICE float operator()(float x) const {
    x = math::sd_abs(x);
    if (x >= _radius) return 0.0f;
    return std::exp(-x * x / (2.0 * _sigma * _sigma));
  }
  SD_HOST_DEVICE float radius() const { return _radius; }
  const float _radius;
  const float _sigma;  // Gaussian standard deviation
  // see: https://stackoverflow.com/questions/41552966/getting-new-delete-type-mismatch-from-asan
  virtual ~GaussianKernelFunc() = default;
};

struct BoxKernelFunc
#if !defined(__CUDACC__)
    : public IKernelFunc<float>
#endif
{
  SD_HOST_DEVICE float operator()(float x) const {
    x = math::sd_abs(x);
    return x < 0.5f ? 1.f : x == 0.5f ? 0.5f : 0.f;
  }
  SD_HOST_DEVICE float radius() const { return 1.f; }

  // see: https://stackoverflow.com/questions/41552966/getting-new-delete-type-mismatch-from-asan
  virtual ~BoxKernelFunc() = default;
};

struct TriangleKernelFunc
#if !defined(__CUDACC__)
    : public IKernelFunc<float>
#endif
{
  // https://en.wikipedia.org/wiki/Triangle_function
  SD_HOST_DEVICE float operator()(float x) const {
    x = math::sd_abs(x);
    return x < 1.f ? 1.f - x : 0.f;
  }
  SD_HOST_DEVICE float radius() const { return 1.f; }

  // see: https://stackoverflow.com/questions/41552966/getting-new-delete-type-mismatch-from-asan
  virtual ~TriangleKernelFunc() = default;
};

struct MitchellCubicKernelFunc
#if !defined(__CUDACC__)
    : public IKernelFunc<float>
#endif
{
  // https://doi.org/10.1145/378456.378514
  // D. P. Mitchell and A. N. Netravali. Reconstruction filters in computer
  // graphics.  Computer Graphics (Proceedings of ACM SIGGRAPH 1988),
  // 22(4):221–228, 1988.
  SD_HOST_DEVICE float operator()(float x) const {
    x = math::sd_abs(x);
    if (x >= 2.f) {
      return 0.f;
    } else if (x >= 1.f) {
      return (((-7.f / 18.f) * x + 2.f) * x - 10.f / 3.f) * x + 16.f / 9.f;
    } else {
      return (((7.f / 6.f) * x - 2.f) * x) * x + 8.f / 9.f;
    }
  }
  SD_HOST_DEVICE float radius() const { return 2.f; }
};

// A pre-computed span of pixels along a single dimension.
// The output pixel will be the weighted sum of pixels starting from start.
struct Spans {
  // The maximum span size of any output pixel.
  int _spanSize;
  // int32 tensor with shape {outputSize}.
  NDArray _starts;

  // float32 tensor of size {outputSize, spanSize}.
  // The output pixel at x is computed as:
  //   dot_product(input[starts[x]:starts[x]+span_size], weights[x]).
  NDArray _weights;
  // see: https://stackoverflow.com/questions/41552966/getting-new-delete-type-mismatch-from-asan
  virtual ~Spans() = default;

};

template <typename I, typename F>
struct ImageResizerStateCommon {
  explicit SD_HOST_DEVICE ImageResizerStateCommon(bool alignCorners, bool halfPixelCenters)
      : _alignCorners(alignCorners), _halfPixelCenters(halfPixelCenters) {}

#if defined(__CUDACC__)
  explicit SD_HOST_DEVICE ImageResizerStateCommon(bool alignCorners, bool halfPixelCenters, cudaStream_t* cudaStream)
      : _alignCorners(alignCorners), _halfPixelCenters(halfPixelCenters), stream(cudaStream){};
#endif

  // calculateResizeScale determines the F scaling factor.
  static SD_HOST_DEVICE inline F calculateResizeScale(I inSize, I outSize, bool alignCorners) {
    return (alignCorners && outSize > 1) ? (inSize - 1) / static_cast<F>(outSize - 1)
                                         : inSize / static_cast<F>(outSize);
  }

  // ValidateAndCalculateOutputSize checks the bounds on the input tensors
  // and requested size, sets up some of the resizing state such as the
  // heightScale and widthScale, and calculates the output size.
  // If any of these operations fails, it sets an error status in
  // the context, which the caller must check.
  sd::Status validateAndCalculateOutputSize(NDArray const* input, int const width, int const height) {
    //
    batchSize = input->sizeAt(0);  //.dim_size(0);
    outHeight = static_cast<I>(height);
    outWidth = static_cast<I>(width);  // internal::SubtleMustCopy(Svec(1));
    inHeight = static_cast<I>(input->sizeAt(1));
    inWidth = static_cast<I>(input->sizeAt(2));
    channels = input->sizeAt(3);  //.dim_size(3);
    heightScale = calculateResizeScale(inHeight, outHeight, _alignCorners);
    widthScale = calculateResizeScale(inWidth, outWidth, _alignCorners);
    inputEws1 = input->ews() == 1;
    bStride = input->strideAt(0);
    hStride = input->strideAt(1);
    wStride = input->strideAt(2);
    cStride = input->strideAt(3);
    // Guard against overflows
    if (ceilf((outHeight - 1) * heightScale) > static_cast<float>(DataTypeUtils::max<int>())) {
      sd_printf("resize_bicubic: Upper overflow occurs for resize height (%f)\n", ceilf((outHeight - 1) * heightScale));
      return Logger::logStatusMsg(sd::Status::BAD_INPUT, "resize_bicubic: Upper overflow occurs for resize height");
    }
    if (ceilf((outWidth - 1) * heightScale) > static_cast<float>(DataTypeUtils::max<int>())) {
      sd_printf("resize_bicubic: Upper overflow occurs for resize height (%f)\n", ceilf((outHeight - 1) * heightScale));
      return Logger::logStatusMsg(sd::Status::BAD_INPUT, "resize_bicubic: Upper overflow occurs for resize width");
    }

    return sd::Status::OK;
  }

  // Calculates all the required variables, and allocates the output.
  sd::Status validateAndCreateOutput(NDArray const* input, int const width, int const height) {
    return validateAndCalculateOutputSize(input, width, height);
  }

  I batchSize;
  I outHeight;
  I outWidth;
  I inHeight;
  I inWidth;
  I channels;
  I bStride;
  I hStride;
  I wStride;
  I cStride;
  bool inputEws1;
  F heightScale;
  F widthScale;
  NDArray* output = nullptr;
#if defined(__CUDACC__)
  cudaStream_t* stream;
#endif
 private:
  bool _alignCorners;
  bool _halfPixelCenters;
};

using ImageResizerState = ImageResizerStateCommon<sd::LongType, float>;

struct BilinearInterpolationData {
  sd::LongType bottomIndex;  // Lower source index used in the interpolation
  sd::LongType topIndex;     // Upper source index used in the interpolation
  // 1-D linear iterpolation scale (see:
  // https://en.wikipedia.org/wiki/Bilinear_interpolation)
  double interpolarValue;
  // see: https://stackoverflow.com/questions/41552966/getting-new-delete-type-mismatch-from-asan
  virtual ~BilinearInterpolationData() = default;

};

SD_INLINE SD_HOST_DEVICE float legacy_scaler(const int x, const float scale) { return static_cast<float>(x) * scale; }

// Older incorrect scaling method that causes all resizes to have a slight
// translation leading to inconsistent results. For example, a flip then a
// resize gives different results then a resize then a flip.
struct LegacyScaler {
  SD_HOST_DEVICE LegacyScaler(){};
  SD_INLINE SD_HOST_DEVICE float operator()(const int x, const float scale) const {
    return static_cast<float>(x) * scale;
  }

  // see: https://stackoverflow.com/questions/41552966/getting-new-delete-type-mismatch-from-asan
  virtual ~LegacyScaler() = default;
};

// Half pixel scaler scales assuming that the pixel centers are at 0.5, i.e. the
// floating point coordinates of the top,left pixel is 0.5,0.5.
struct HalfPixelScaler {
  SD_HOST_DEVICE HalfPixelScaler(){};
  SD_INLINE SD_HOST_DEVICE float operator()(const int x, const float scale) const {
    // Note that we subtract 0.5 from the return value, as the existing bilinear
    // sampling code etc assumes pixels are in the old coordinate system.
    return (static_cast<float>(x) + 0.5f) * scale - 0.5f;
  }

  // see: https://stackoverflow.com/questions/41552966/getting-new-delete-type-mismatch-from-asan
  virtual ~HalfPixelScaler() = default;
};

// Half pixel scaler scales assuming that the pixel centers are at 0.5, i.e. the
// floating point coordinates of the top,left pixel is 0.5,0.5.
struct HalfPixelScalerNN {
  SD_HOST_DEVICE HalfPixelScalerNN(){};
  SD_INLINE SD_HOST_DEVICE float operator()(const int x, const float scale) const {
    // Note that we subtract 0.5 from the return value, as the existing bilinear
    // sampling code etc assumes pixels are in the old coordinate system.
    return (static_cast<float>(x) + 0.5f) * scale;
  }

  // see: https://stackoverflow.com/questions/41552966/getting-new-delete-type-mismatch-from-asan
  virtual ~HalfPixelScalerNN() = default;
};

constexpr sd::LongType kTableSize = (1 << 10);

struct WeightsAndIndices {
  float _weight0;
  float _weight1;
  float _weight2;
  float _weight3;
  sd::LongType _index0;
  sd::LongType _index1;
  sd::LongType _index2;
  sd::LongType _index3;

  int _advance;  // advance value.
  // see: https://stackoverflow.com/questions/41552966/getting-new-delete-type-mismatch-from-asan
  virtual ~WeightsAndIndices() = default;
};

SD_INLINE SD_HOST_DEVICE sd::LongType bound(sd::LongType val, sd::LongType limit) {
  return math::sd_min(limit - 1ll, math::sd_max(sd::LongType{0}, val));
}

template <typename T>
SD_INLINE SD_HOST_DEVICE float interpolate1D(const float weight0, const float weight1, const float weight2,
                                             const float weight3, const T value0, const T value1, const T value2,
                                             const T value3) {
  auto ret = static_cast<float>(value0) * weight0 + static_cast<float>(value1) * weight1 +
             static_cast<float>(value2) * weight2 + static_cast<float>(value3) * weight3;

  return ret;
}

// Compute the 1D interpolation for a given X index using the y_weights
static SD_HOST_DEVICE float compute(float values[4], const float xW0, const float xW1, const float xW2,
                                    const float xW3) {
  return interpolate1D(xW0, xW1, xW2, xW3, values[0], values[1], values[2], values[3]);
}

template <typename T>
static SD_INLINE SD_HOST_DEVICE float computeYInterpolation(int which, int channelNum, const WeightsAndIndices& yWai,
                                                            const T* pY0, const T* pY1, const T* pY2, const T* pY3,
                                                            const WeightsAndIndices& xWai) {
  int xIndex;
  switch (which) {
    case 0:
      xIndex = xWai._index0;
      break;
    case 1:
      xIndex = xWai._index1;
      break;
    case 2:
      xIndex = xWai._index2;
      break;
    default:
      xIndex = xWai._index3;
      break;
  }
  const int pt_index = xIndex + channelNum;

  return interpolate1D<T>(yWai._weight0, yWai._weight1, yWai._weight2, yWai._weight3, pY0[pt_index], pY1[pt_index],
                          pY2[pt_index], pY3[pt_index]);
}

template <typename Scaler>
SD_INLINE SD_HOST_DEVICE void getWeightsAndIndices(const float* coeffs_table, const float scale,
                                                   const sd::LongType out_loc, const sd::LongType limit,
                                                   WeightsAndIndices* out, bool exclude_outside) {
  const Scaler scaler;
  const float in_loc_f = scaler(out_loc, scale);
  const sd::LongType in_loc = math::sd_floor<float, sd::LongType>(in_loc_f);
  const float delta = in_loc_f - in_loc;
  const sd::LongType offset = math::sd_round<float, sd::LongType>(delta * kTableSize);

  if (exclude_outside) {
    // The legacy code placed more weight on the edge pixels, since bounding
    // the set of inputs to sample could cause an edge pixel to be repeated.
    // Here we change the behavior at borders to match that used by the
    // scale_and_translate_op, where sampling locations outside the image have
    // their weight set to 0, and the weights are renormalized so that their sum
    // is 1.0.
    out->_index0 = bound(in_loc - 1, limit);
    out->_weight0 = (out->_index0 == in_loc - 1 ? coeffs_table[offset * 2 + 1] : 0.0f);
    out->_index1 = bound(in_loc, limit);
    out->_weight1 = (out->_index1 == in_loc ? coeffs_table[offset * 2] : 0.0f);
    out->_index2 = bound(in_loc + 1, limit);
    out->_weight2 = (out->_index2 == in_loc + 1 ? coeffs_table[(kTableSize - offset) * 2] : 0.0f);
    out->_index3 = bound(in_loc + 2, limit);
    out->_weight3 = (out->_index3 == in_loc + 2 ? coeffs_table[(kTableSize - offset) * 2 + 1] : 0.0f);

    const float weight_sum = out->_weight0 + out->_weight1 + out->_weight2 + out->_weight3;
    if (math::sd_abs(weight_sum) >= 1000.0f * DataTypeUtils::min<float>()) {
      const float one_over_weight_sum = 1.0f / weight_sum;
      out->_weight0 *= one_over_weight_sum;
      out->_weight1 *= one_over_weight_sum;
      out->_weight2 *= one_over_weight_sum;
      out->_weight3 *= one_over_weight_sum;
    }
  } else {
    out->_weight0 = coeffs_table[offset * 2 + 1];
    out->_weight1 = coeffs_table[offset * 2];
    out->_weight2 = coeffs_table[(kTableSize - offset) * 2];
    out->_weight3 = coeffs_table[(kTableSize - offset) * 2 + 1];
    out->_index0 = bound(in_loc - 1, limit);
    out->_index1 = bound(in_loc, limit);
    out->_index2 = bound(in_loc + 1, limit);
    out->_index3 = bound(in_loc + 2, limit);
  }
}

class CachedInterpolationCalculator {
 public:
  SD_HOST_DEVICE CachedInterpolationCalculator() : _indexes{-1, -1, -1, -1} {}

  // Advances iteration. Returns the number of values that should be copied from
  // the current point to the next point. The copying should always be done by
  // copying the last <retval> values from the old point to the first <retval>
  // values of the new point.
  SD_INLINE SD_HOST_DEVICE int Advance(const sd::LongType x0, const sd::LongType x1, const sd::LongType x2,
                                       const sd::LongType x3) {
    // We use 2 hands and walk through, copying from one to another where
    // we already have values.
    // Invariant, new_indicies_hand <= cached_values_hand
    const sd::LongType new_x_indices[4] = {x0, x1, x2, x3};
    int cachedValuesHand = 0;
    int newIndiciesHand = 0;

    while (cachedValuesHand < 4) {
      if (_indexes[cachedValuesHand] == new_x_indices[newIndiciesHand]) {
        if (newIndiciesHand < cachedValuesHand) {
          _indexes[newIndiciesHand] = _indexes[cachedValuesHand];
        }
        newIndiciesHand++;
      }
      cachedValuesHand++;
    }
    switch (newIndiciesHand) {
      case 0:
        _indexes[0] = x0;
      case 1:
        _indexes[1] = x1;
      case 2:
        _indexes[2] = x2;
      case 3:
        _indexes[3] = x3;
        break;
    }
    return newIndiciesHand;
  }

 private:
  sd::LongType _indexes[4];
};

template <typename F, typename I>
struct CachedInterpolationT {
  I start;
  I end;
  F startScale;
  F endMinusOneScale;
  bool needsBounding;
};

using CachedInterpolation = CachedInterpolationT<float, sd::LongType>;
// ResizeArea
template <typename T>
struct ScaleCache {
  float yScale;
  T const* yPtr;
  using workType = float;
};

// Computes the sum of all x values defined by <x_interp> taken across
// the y offsets and scales defined by y_ptrs and y_scales, for channel c.
//
// Note that <NeedsXBounding> is a template parameter to avoid a performance
// penalty from dynamically checking it.
template <typename F, typename I, typename T = typename ScaleCache<F>::workType>
SD_HOST_DEVICE void computePatchSumOf3Channels(T scale, const ImageResizerState& st, const ScaleCache<F>* yScaleCache,
                                               I ptrsLen, const CachedInterpolationT<T, I>& xCache, T* outputPtr) {
  bool const needsXBounding = xCache.needsBounding;

  auto boundIfNeeded = [needsXBounding](sd::LongType x, sd::LongType y) -> sd::LongType {
    return (needsXBounding ? bound(x, y) : (x));
  };

  T sum_0 = T(0);
  T sum_1 = T(0);
  T sum_2 = T(0);
  auto cStride = st.cStride;
  auto cStrideX2 = st.cStride + st.cStride;
  for (int i = 0; i < ptrsLen; ++i) {
    const F* ptr = yScaleCache[i].yPtr;
    T scaleX = xCache.startScale;
    auto offset = st.wStride * boundIfNeeded(xCache.start, st.inWidth);
    T sum_y_0 = static_cast<T>(ptr[offset]) * scaleX;
    T sum_y_1 = static_cast<T>(ptr[offset + cStride]) * scaleX;
    T sum_y_2 = static_cast<T>(ptr[offset + cStrideX2]) * scaleX;

    if (xCache.start + 1 != xCache.end) {
      for (auto x = xCache.start + 1; x < xCache.end - 1; ++x) {
        auto offset = st.wStride * boundIfNeeded(x, st.inWidth);
        sum_y_0 += static_cast<T>(ptr[offset]);
        sum_y_1 += static_cast<T>(ptr[offset + cStride]);
        sum_y_2 += static_cast<T>(ptr[offset + cStrideX2]);
      }
      scaleX = xCache.endMinusOneScale;
      offset = st.wStride * boundIfNeeded(xCache.end - 1, st.inWidth);
      sum_y_0 += static_cast<T>(ptr[offset]) * scaleX;
      sum_y_1 += static_cast<T>(ptr[offset + cStride]) * scaleX;
      sum_y_2 += static_cast<T>(ptr[offset + cStrideX2]) * scaleX;
    }
    sum_0 += sum_y_0 * yScaleCache[i].yScale;
    sum_1 += sum_y_1 * yScaleCache[i].yScale;
    sum_2 += sum_y_2 * yScaleCache[i].yScale;
  }

  outputPtr[0] = sum_0 * scale;
  outputPtr[1] = sum_1 * scale;
  outputPtr[2] = sum_2 * scale;
}

// Computes the sum of all x values defined by <x_interp> taken across
// the y offsets and scales defined by y_ptrs and y_scales, for channel c.
//
// Note that <NeedsXBounding> is a template parameter to avoid a performance
// penalty from dynamically checking it.
template <typename F, typename I, typename T = typename ScaleCache<F>::workType>
SD_HOST_DEVICE void computePatchSum(T scale, const ImageResizerState& st, const ScaleCache<F>* yScaleCache, I ptrsLen,
                                    const CachedInterpolationT<T, I>& xCache, T* outputPtr) {
  bool const needsXBounding = xCache.needsBounding;

  auto boundIfNeeded = [needsXBounding](sd::LongType x, sd::LongType y) -> sd::LongType {
    return (needsXBounding ? bound(x, y) : (x));
  };

  const auto numChannels = st.channels;
  for (sd::LongType c = 0; c < numChannels; ++c) {
    T sum = T(0);
    for (int i = 0; i < ptrsLen; ++i) {
      F const* ptr = yScaleCache[i].yPtr;
      T scaleX = xCache.startScale;
      T sumY = static_cast<T>(ptr[st.wStride * boundIfNeeded(xCache.start, st.inWidth) + c * st.cStride]) * scaleX;
      if (xCache.start + 1 != xCache.end) {
        for (sd::LongType x = xCache.start + 1; x < xCache.end - 1; ++x) {
          sumY += static_cast<T>(ptr[st.wStride * boundIfNeeded(x, st.inWidth) + c * st.cStride]);
        }
        scaleX = xCache.endMinusOneScale;
        sumY += static_cast<T>(ptr[st.wStride * boundIfNeeded(xCache.end - 1, st.inWidth) + c * st.cStride]) * scaleX;
      }
      sum += sumY * yScaleCache[i].yScale;
    }
    outputPtr[c] = sum * scale;
  }
}

template <typename X, typename Z>
SD_HOST_DEVICE void gatherRows(int const spanSize, int const* starts, Z const* weights, X const* imagePtr,
                               sd::LongType const inputHeight, sd::LongType const inputWidth,
                               sd::LongType const outputHeight, sd::LongType const outputWidth,
                               sd::LongType const channels, Z* outputPtr, bool inputEws1, sd::LongType inRowStride,
                               sd::LongType wStride, sd::LongType cStride) {
  auto inRowSize = inputWidth * channels;
  auto outRowSize = outputWidth * channels;

  if (inputEws1) {
    auto addScaledVector = [](const X* inVector, int vectorLen, Z weight, Z* outVector) {
      Z* outVecEnd = outVector + vectorLen;
      for (; outVector != outVecEnd; ++outVector, ++inVector) {
        *outVector += weight * static_cast<Z>(*inVector);
      }
    };

    for (int y = 0; y < outputHeight; ++y) {
      Z* outRowData = outputPtr + outRowSize * y;
      memset(outRowData, '\0',
             outRowSize * sizeof(Z));  //            std::fill(outRowData, outRowData + outRowSize, 0.f);
      int inRow = starts[y];
      auto inRowData = imagePtr + inRowSize * inRow;
      auto weightsStart = weights + y * spanSize;
      auto realSpanSize = math::sd_min(starts[y] + spanSize, static_cast<int>(inputHeight)) - starts[y];
      auto weightsEnd = weightsStart + realSpanSize;
      for (auto weightPtr = weightsStart; weightPtr != weightsEnd; ++weightPtr) {
        addScaledVector(inRowData, inRowSize, *weightPtr, outRowData);
        inRowData += inRowSize;
      }
    }

  } else {
    auto addScaledVector = [](const X* inVector, int inputWidth, int channels, const sd::LongType wStride,
                              const sd::LongType cStride, Z weight, Z* outVector) {
      const X* inVec = inVector;
      for (int i = 0; i < inputWidth; i++) {
        for (int c = 0; c < channels; c++) {
          *outVector += weight * static_cast<Z>(inVec[c * cStride]);
          ++outVector;
        }
        inVec += wStride;
      }
    };

    for (int y = 0; y < outputHeight; ++y) {
      Z* outRowData = outputPtr + outRowSize * y;
      memset(outRowData, '\0',
             outRowSize * sizeof(Z));  //            std::fill(outRowData, outRowData + outRowSize, 0.f);
      int inRow = starts[y];
      auto inRowData = imagePtr + inRowStride * inRow;
      auto weightsStart = weights + y * spanSize;
      auto realSpanSize = math::sd_min(starts[y] + spanSize, static_cast<int>(inputHeight)) - starts[y];
      auto weightsEnd = weightsStart + realSpanSize;
      for (auto weightPtr = weightsStart; weightPtr != weightsEnd; ++weightPtr) {
        addScaledVector(inRowData, inputWidth, channels, wStride, cStride, *weightPtr, outRowData);
        inRowData += inRowStride;
      }
    }
  }
}

template <typename Z>
SD_HOST_DEVICE void gatherColumns(int const spanSize, int const* starts, Z const* weights, Z const* imagesPtr,
                                  sd::LongType const inputHeight, sd::LongType const inputWidth,
                                  sd::LongType const outputHeight, sd::LongType const outputWidth,
                                  sd::LongType channels, Z* outputPtr) {
  auto inRowSize = inputWidth * channels;
  auto outRowSize = outputWidth * channels;

  for (auto y = 0LL; y < outputHeight; ++y) {
    auto inputRowStart = imagesPtr + inRowSize * y;
    auto outPixels = outputPtr + outRowSize * y;
    for (auto x = 0LL; x < outputWidth; ++x, outPixels += channels) {
      auto inPixels = inputRowStart + starts[x] * channels;
      auto weightsStart = weights + x * spanSize;
      auto realSpanSize = math::sd_min(starts[x] + spanSize, static_cast<int>(inputWidth)) - starts[x];
      auto weightsEnd = weightsStart + realSpanSize;
      for (int c = 0; c < channels; ++c) {
        outPixels[c] = 0.0f;
      }
      for (auto weightPtr = weightsStart; weightPtr != weightsEnd; ++weightPtr) {
        Z w = *weightPtr;
        for (int c = 0; c < channels; ++c) {
          outPixels[c] += w * static_cast<Z>(inPixels[c]);
        }
        inPixels += channels;
      }
    }
  }
}

SD_LIB_HIDDEN sd::Status resizeBilinearFunctor(sd::LaunchContext* context, NDArray const* image, int const width,
                                               int const height, bool const alignCorners, bool const halfPixelCenter,
                                               NDArray* output);
SD_LIB_HIDDEN sd::Status resizeNeighborFunctor(sd::LaunchContext* context, NDArray const* images, int const width,
                                               int const height, CoordinateTransformationMode coorMode,
                                               NearestMode nearestMode, bool alignCorner, NDArray* output);
SD_LIB_HIDDEN sd::Status resizeBicubicFunctor(sd::LaunchContext* context, NDArray const* image, int const width,
                                              int const height, bool preserveAspectRatio, bool antialias,
                                              NDArray* output);
SD_LIB_HIDDEN sd::Status resizeBicubicFunctorA(sd::LaunchContext* context, NDArray const* image, int const width,
                                               int const height, bool const alignCorners,
                                               CoordinateTransformationMode coorMode, bool exclude_outside,
                                               double coefficient, NDArray* output);
SD_LIB_HIDDEN sd::Status resizeAreaFunctor(sd::LaunchContext* context, NDArray const* image, int const width,
                                           int const height, bool const alignCorners, NDArray* output);

SD_LIB_HIDDEN sd::Status resizeFunctor(sd::LaunchContext* context, NDArray const* image, int const width,
                                       int const height, ImageResizeMethods method,
                                       CoordinateTransformationMode coorMode, bool exclude_outside,
                                       NearestMode nearestMode, double coefficient, bool antialias, NDArray* output);

SD_LIB_HIDDEN sd::Status resizeImagesFunctor(sd::LaunchContext* context, NDArray const* image, int const width,
                                             int const height, ImageResizeMethods method, bool alignCorners,
                                             NDArray* output);
}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
