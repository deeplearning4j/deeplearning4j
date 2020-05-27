#include <array/NDArrayFactory.h>
#include <exceptions/cuda_exception.h>
#include <ops/declarable/helpers/image_resize.h>
#include <helpers/PointersManager.h>

namespace sd {
    namespace ops {
        namespace helpers {
// -------------------------------------------------------------------------------------------------------------- //
// resize v2 implementation                                                                                       //
// -------------------------------------------------------------------------------------------------------------- //
// A functional interface for a scale kernels.
//struct IKernelFunc {
//    _CUDA_HD virtual float operator()(float x) const = 0;
//    _CUDA_HD virtual float radius() const = 0;
//    _CUDA_HD virtual size_t size() const = 0;
//};

struct LanczosKernelFunc /*: public IKernelFunc*/ {
    // Pass 1 for Lanczos1 kernel, 3 for Lanczos3 etc.
    explicit LanczosKernelFunc(float const radius) : _radius(radius) {}
    _CUDA_HD float operator()(float x) const {
        float const kPI = 3.141592653589793f;
        x = math::nd4j_abs(x);
        if (x > _radius) return 0.f;
        // Need to special case the limit case of sin(x) / x when x is zero.
        if (x <= 1.e-3f) {
            return 1.f;
        }
        return _radius * std::sin(kPI * x) * std::sin(kPI * x / _radius) / (kPI * kPI * x * x);
    }
    _CUDA_HD float radius() const { return _radius; }
    const float _radius;
};

struct GaussianKernelFunc /*: public IKernelFunc*/ {
    static constexpr float kRadiusMultiplier = 3.0f;
    // https://en.wikipedia.org/wiki/Gaussian_function
    // We use sigma = 0.5, as suggested on p. 4 of Ken Turkowski's "Filters
    // for Common Resampling Tasks" for kernels with a support of 3 pixels:
    // www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
    // This implies a radius of 1.5,
    explicit GaussianKernelFunc(float radius = 1.5f)
            : _radius(radius), _sigma(radius / kRadiusMultiplier) {}
    _CUDA_HD float operator()(float x) const {
        x = math::nd4j_abs(x);
        if (x >= _radius) return 0.0f;
        return std::exp(-x * x / (2.0 * _sigma * _sigma));
    }
    _CUDA_HD float radius() const { return _radius; }
    const float _radius;
    const float _sigma;  // Gaussian standard deviation
};

struct BoxKernelFunc /*: public IKernelFunc*/ {
    _CUDA_HD float operator()(float x) const {
        x = math::nd4j_abs(x);
        return x < 0.5f ? 1.f : x == 0.5f ? 0.5f : 0.f;
    }
    _CUDA_HD float radius() const { return 1.f; }
    _CUDA_HD size_t size() const { return sizeof(BoxKernelFunc); }
};

struct TriangleKernelFunc /*: public IKernelFunc*/ {
    // https://en.wikipedia.org/wiki/Triangle_function
    _CUDA_HD float operator()(float x) const {
        x = math::nd4j_abs(x);
        return x < 1.f ? 1.f - x : 0.f;
    }
    _CUDA_HD float radius() const { return 1.f; }
};

struct KeysCubicKernelFunc /*: public IKernelFunc*/ {
    // http://ieeexplore.ieee.org/document/1163711/
    // R. G. Keys. Cubic convolution interpolation for digital image
    // processing. IEEE Transactions on Acoustics, Speech, and Signal
    // Processing, 29(6):1153–1160, 1981.
    _CUDA_HD float operator()(float x) const {
        x = math::nd4j_abs(x);
        if (x >= 2.0f) {
            return 0.0f;
        } else if (x >= 1.0f) {
            return ((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f;
        } else {
            return ((1.5f * x - 2.5f) * x) * x + 1.0f;
        }
    }
    _CUDA_HD float radius() const { return 2.f; }
};

struct MitchellCubicKernelFunc/* : public IKernelFunc*/ {
    // https://doi.org/10.1145/378456.378514
    // D. P. Mitchell and A. N. Netravali. Reconstruction filters in computer
    // graphics.  Computer Graphics (Proceedings of ACM SIGGRAPH 1988),
    // 22(4):221–228, 1988.
    _CUDA_HD float operator()(float x) const {
        x = math::nd4j_abs(x);
        if (x >= 2.f) {
            return 0.f;
        } else if (x >= 1.f) {
            return (((-7.f / 18.f) * x + 2.f) * x - 10.f / 3.f) * x + 16.f / 9.f;
        } else {
            return (((7.f / 6.f) * x - 2.f) * x) * x + 8.f / 9.f;
        }
    }
    _CUDA_HD float radius() const { return 2.f; }
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
};

static inline _CUDA_HD Nd4jLong boundsAmp(Nd4jLong  const low, Nd4jLong const high, Nd4jLong const value) {
    if (high < value) return high;
    if (value < low) return low;
    return value;
}

template <typename TKernelFunc>
static __global__ void computeSpansKernel(TKernelFunc* kernel, int* startsVec, float* weightsVector, Nd4jLong outSize, Nd4jLong  inSize, float kernelScale, int spanSize, float const invScale, float const invTranslate, float invKernelScale, float* tempWeightsBuf) {

    // return value if within bounds or bounds otherwise
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto step = blockDim.x * gridDim.x;
    __shared__ int maxSpanSize;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        maxSpanSize = 0;
    }
    __syncthreads();

    for (auto x = tid; x < outSize; x += step) {
        const float columnFloat = x + 0.5f;
        const float sampleFloat = columnFloat * invScale + invTranslate;

        // Don't sample when the sampling location is outside the source image.
        if (sampleFloat < 0 || sampleFloat > inSize) {
            // Add an empty span.
            startsVec[x] = 0;
            continue;
        }
        Nd4jLong spanStart = math::nd4j_ceil<float,float>(sampleFloat - kernel->radius() * kernelScale - 0.5f);
        Nd4jLong spanEnd = math::nd4j_floor<float, float>(sampleFloat + kernel->radius() * kernelScale - 0.5f);
        spanStart = boundsAmp(0LL, inSize - 1, spanStart);
        spanEnd = boundsAmp(0LL, inSize - 1, spanEnd) + 1;
        int const spanSize = spanEnd - spanStart;
        if (spanSize > spanSize) {
            return ; //throw "Exception"; ////return Status::CODE(ND4J_STATUS_BAD_INPUT, "Span is too large: "); // + spanSize + " vs " + spans._spanSize);//, spanSize, spans._spanSize));
        }
        float totalWeightSum = 0.f;
        auto tempWeights = &tempWeightsBuf[x];
        auto actualWeights = 0;
        for (int source = spanStart; source < spanEnd; ++source) {
            float kernelPos = static_cast<float>(source) + 0.5f - sampleFloat;
            float weight = (*kernel)(kernelPos * invKernelScale);
            totalWeightSum += weight;
            tempWeights[actualWeights++] = weight;
        }
        maxSpanSize = math::nd4j_max(maxSpanSize, spanSize);
        if (math::nd4j_abs(totalWeightSum) >= 1000.f * DataTypeUtils::min<float>()) { //
            auto totalWeightSumInverted = 1.0f / totalWeightSum;
            auto outIndex = spanSize * x;
            for (auto weightIndex = 0; weightIndex < actualWeights; ++weightIndex) {
                weightsVector[outIndex] = tempWeights[weightIndex] * totalWeightSumInverted;
                ++outIndex;
            }
        }
        startsVec[x] = spanStart;
    }

}

template <typename TKernelFunc>
static int computeSpans(LaunchContext* context, TKernelFunc& kernel, Nd4jLong const outSize, Nd4jLong const inSize, float const scale, float const translate, bool const antialias, Spans& spans) {
    // When sampling, we need the inverse scale and translation, to map from an
    // output to an input pixel.
    float const invScale = 1.f / scale;
    float const invTranslate = -invScale * translate;
    // When downsampling the kernel should be scaled since we want to low pass
    // filter and interpolate, but when upsampling it should not be since we only
    // want to interpolate.
    float  const kernelScale = antialias ? math::nd4j_max(invScale, 1.f) : 1.f;
    spans._spanSize = math::nd4j_min(2 * static_cast<int>(std::ceil(kernel.radius() * kernelScale)) + 1, static_cast<int>(inSize));
    spans._starts = NDArrayFactory::create<int>('c', {outSize}); spans._starts.syncToHost();
    spans._weights = NDArrayFactory::create<float>('c', {outSize, spans._spanSize}); spans._weights.syncToHost();

    auto startsVec = reinterpret_cast<int*>(spans._starts.buffer());
    auto weightsVector = reinterpret_cast<float*>(spans._weights.buffer());
    spans._weights.nullify();

    const float invKernelScale = 1.f / kernelScale;
//    NDArray tempWeights = NDArrayFactory::create<float>('c', {outSize, spans._spanSize});
//    auto tempWeightsBuf = reinterpret_cast<float*>(tempWeights.specialBuffer());
//    PointersManager mg(context, "ops::helpers::computeSpans");
//    auto specialKernel = reinterpret_cast<TKernelFunc*>(mg.replicatePointer(&kernel, sizeof(TKernelFunc)));
    auto stream = context->getCudaStream();
    //computeSpansKernel<TKernelFunc><<<1, 1, 128, *stream>>>(specialKernel, startsVec, weightsVector, outSize, inSize, kernelScale, spans._spanSize, invScale, invTranslate, invKernelScale, tempWeightsBuf);
    auto maxSpanSize = 0;
    std::vector<float> tempWeights;
    for (auto x = 0; x < outSize; x ++) {
        const float columnFloat = x + 0.5f;
        const float sampleFloat = columnFloat * invScale + invTranslate;

        // Don't sample when the sampling location is outside the source image.
        if (sampleFloat < 0 || sampleFloat > inSize) {
            // Add an empty span.
            startsVec[x] = 0;
            continue;
        }
        Nd4jLong spanStart = math::nd4j_ceil<float,float>(sampleFloat - kernel.radius() * kernelScale - 0.5f);
        Nd4jLong spanEnd = math::nd4j_floor<float, float>(sampleFloat + kernel.radius() * kernelScale - 0.5f);
        spanStart = boundsAmp(0LL, inSize - 1, spanStart);
        spanEnd = boundsAmp(0LL, inSize - 1, spanEnd) + 1;
        int const spanSize = spanEnd - spanStart;
        if (spanSize > spans._spanSize) {
            return Status::CODE(ND4J_STATUS_BAD_INPUT, "Span is too large: "); // + spanSize + " vs " + spans._spanSize);//, spanSize, spans._spanSize));
        }
        float totalWeightSum = 0.f;
        tempWeights.clear();

        for (int source = spanStart; source < spanEnd; ++source) {
            float kernelPos = static_cast<float>(source) + 0.5f - sampleFloat;
            float weight = kernel(kernelPos * invKernelScale);
            totalWeightSum += weight;
            tempWeights.push_back(weight);
        }
        maxSpanSize = math::nd4j_max(maxSpanSize, spanSize);
        if (math::nd4j_abs(totalWeightSum) >= 1000.f * DataTypeUtils::min<float>()) { //
            auto totalWeightSumInverted = 1.0f / totalWeightSum;
            auto outIndex = spans._spanSize * x;
            for (auto weightIndex = 0; weightIndex < tempWeights.size(); ++weightIndex) {
                weightsVector[outIndex++] = tempWeights[weightIndex] * totalWeightSumInverted;
//                ++outIndex;
            }
        }
        startsVec[x] = spanStart;
    }
    spans._starts.tickWriteHost(); spans._weights.tickWriteHost();
    spans._starts.syncToDevice();
    spans._weights.syncToDevice();
//    cudaStreamSynchronize(*stream);
    return Status::OK();
}

//template int computeSpans(LaunchContext* context, TriangleKernelFunc& kernel, Nd4jLong const outSize, Nd4jLong const inSize, float const scale, float const translate, bool const antialias, Spans& spans);


template <typename X, typename Z>
static __device__ void gatherRows(int const spanSize, int const* starts, Z const* weights, X const* imagePtr, Nd4jLong const inputHeight, Nd4jLong const inputWidth, Nd4jLong const outputHeight,
                       Nd4jLong const outputWidth, Nd4jLong const channels, Z* outputPtr) {
    auto inRowSize = inputWidth * channels;
    auto outRowSize = outputWidth * channels;

    auto addScaledVector = [](const X* inVector, int vectorLen, Z weight, Z* outVector) {
        Z* outVecEnd = outVector + vectorLen;
        for (; outVector != outVecEnd; ++outVector, ++inVector) {
            *outVector += weight * static_cast<Z>(*inVector);
        }
    };

    for (int y = 0; y < outputHeight; ++y) {
        Z* outRowData = outputPtr + outRowSize * y;
        memset(outRowData, '\0', outRowSize * sizeof(Z));//            std::fill(outRowData, outRowData + outRowSize, 0.f);
        int inRow = starts[y];
        auto inRowData = imagePtr + inRowSize * inRow;
        auto weightsStart = weights + y * spanSize;
        auto realSpanSize = math::nd4j_min(starts[y] + spanSize, static_cast<int>(inputHeight)) - starts[y];
        auto weightsEnd = weightsStart + realSpanSize;
        for (auto weightPtr = weightsStart; weightPtr != weightsEnd; ++weightPtr) {
            addScaledVector(inRowData, inRowSize, *weightPtr, outRowData);
            inRowData += inRowSize;
        }
    }
}

template <typename Z>
static __device__ void gatherColumns(int const spanSize, int const* starts, Z const* weights, Z const* imagesPtr, Nd4jLong const inputHeight, Nd4jLong const inputWidth, Nd4jLong const outputHeight, Nd4jLong const outputWidth, Nd4jLong channels, Z* outputPtr) {
    auto inRowSize = inputWidth * channels;
    auto outRowSize = outputWidth * channels;

    for (auto y = 0LL; y < outputHeight; ++y) {
        auto inputRowStart = imagesPtr + inRowSize * y;
        auto outPixels = outputPtr + outRowSize * y;
        for (auto x = 0LL; x < outputWidth; ++x, outPixels += channels) {
            auto inPixels = inputRowStart + starts[x] * channels;
            auto weightsStart = weights + x * spanSize;
            auto realSpanSize = math::nd4j_min(starts[x] + spanSize, static_cast<int>(inputWidth)) - starts[x];
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

template <typename X, typename Z>
static __global__ void batchedGatherSpan(Nd4jLong batchSize, Nd4jLong inputWidth, Nd4jLong inputHeight, Nd4jLong outputWidth, Nd4jLong outputHeight, Nd4jLong channels, int rowSpanSize, int const* rowStartsBuf, Z const* rowWeightBuf, int columnSpanSize, int const* columnStartsBuf, Z const* columnWeightBuf, X const* pImages, Z* pIntermediate, Z* pOutput,
        Nd4jLong inputPixPerBatch, Nd4jLong intermediatePixPerBatch, Nd4jLong outputPixPerBatch) {

    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto step = blockDim.x * gridDim.x;

    for (int b = tid; b < batchSize; b += step) {
        auto imagePtr = pImages + b * inputPixPerBatch;
        auto intermediatePtr = pIntermediate + b * intermediatePixPerBatch;
        auto outputPtr = pOutput + b * outputPixPerBatch;
        gatherRows<X, Z>(rowSpanSize, rowStartsBuf, rowWeightBuf,
                         imagePtr, inputHeight, inputWidth, outputHeight,
                         inputWidth, channels, intermediatePtr);
        gatherColumns<Z>(columnSpanSize, columnStartsBuf, columnWeightBuf,
                         intermediatePtr, outputHeight, inputWidth, outputHeight, outputWidth, channels, outputPtr);
    }
}

template <typename X, typename Z>
static void gatherSpans(LaunchContext* context, int const rowSpanSize, NDArray const& rowStarts, NDArray const& rowWeights, int const colSpanSize, NDArray const& columnStarts, NDArray const& columnWeights, NDArray const* images, NDArray& intermediate, NDArray* output) {
    auto batchSize = images->sizeAt(0);
    auto inputHeight = images->sizeAt(1);
    auto inputWidth = images->sizeAt(2);
    auto channels = images->sizeAt(3);

    auto outputHeight = output->sizeAt(1);
    auto outputWidth = output->sizeAt(2);

    auto inputPixPerBatch = inputWidth * inputHeight * channels;
    auto intermediatePixPerBatch = inputWidth * outputHeight * channels;
    auto outputPixPerBatch = outputWidth * outputHeight * channels;
    auto intermediatePtr = reinterpret_cast<Z*>(intermediate.specialBuffer());

    auto imagePtr = reinterpret_cast<X const*>(images->specialBuffer());
    auto outputPtr = reinterpret_cast<Z*>(output->specialBuffer());
    auto stream = context->getCudaStream();
    auto rowStartsBuf = reinterpret_cast<int const*>(rowStarts.specialBuffer());
    auto rowWeightBuf = reinterpret_cast<Z const*>(rowWeights.specialBuffer());
    auto columnStartsBuf = reinterpret_cast<int const*>(columnStarts.specialBuffer());
    auto columnWeightBuf = reinterpret_cast<Z const*>(columnWeights.specialBuffer());
    batchedGatherSpan<X,Z><<<128, 128, 256, *stream>>>(batchSize, inputWidth, inputHeight, outputWidth, outputHeight, channels, rowSpanSize, rowStartsBuf, rowWeightBuf, colSpanSize, columnStartsBuf, columnWeightBuf, imagePtr, intermediatePtr, outputPtr, inputPixPerBatch, intermediatePixPerBatch, outputPixPerBatch);
}

template <typename X, typename Z>
static int resizeKernel(LaunchContext* context, ImageResizeMethods method, NDArray const* input, Nd4jLong outWidth, Nd4jLong outHeight, bool antialias, NDArray* output) {
    Nd4jLong const batchSize = input->sizeAt(0);
    Nd4jLong const inputHeight = input->sizeAt(1);
    Nd4jLong const inputWidth = input->sizeAt(2);
    Nd4jLong const channels = input->sizeAt(3);
    NDArray::prepareSpecialUse({output}, {input});
    Z rowScale = Z(outHeight) / Z(inputHeight);
    Z columnScale = Z(outWidth) / Z(inputWidth);

    // Return if the output is empty.
    if (output->lengthOf() == 0) return Status::OK();

    Spans colSpans;
    Spans rowSpans;
    auto res = Status::OK();
    switch(method) {
        case kResizeBilinear: {
            TriangleKernelFunc kernel;
            res = computeSpans(context, kernel, outWidth, inputWidth, columnScale, 0.f, antialias,
                                                   colSpans);
            if (res != Status::OK()) return res;
            res = computeSpans(context, kernel, outHeight, inputHeight, rowScale, 0.f, antialias, rowSpans);

        }
            break;
        case kResizeBicubic: {
            KeysCubicKernelFunc kernel;
            res = computeSpans(context, kernel, outWidth, inputWidth, columnScale, 0.f, antialias,
                               colSpans);
            if (res != Status::OK()) return res;
            res = computeSpans(context, kernel, outHeight, inputHeight, rowScale, 0.f, antialias, rowSpans);
        } break;
        case kResizeLanczos3:{
            LanczosKernelFunc kernel(3.f);
            res = computeSpans(context, kernel, outWidth, inputWidth, columnScale, 0.f, antialias,
                               colSpans);
            if (res != Status::OK()) return res;
            res = computeSpans(context, kernel, outHeight, inputHeight, rowScale, 0.f, antialias, rowSpans);

        } break;

        case kResizeLanczos5: {
            LanczosKernelFunc kernel(5.f);
            res = computeSpans(context, kernel, outWidth, inputWidth, columnScale, 0.f, antialias,
                               colSpans);
            if (res != Status::OK()) return res;
            res = computeSpans(context, kernel, outHeight, inputHeight, rowScale, 0.f, antialias, rowSpans);

        } break;
        case kResizeGaussian: {
            GaussianKernelFunc kernel;
            res = computeSpans(context, kernel, outWidth, inputWidth, columnScale, 0.f, antialias,
                               colSpans);
            if (res != Status::OK()) return res;
            res = computeSpans(context, kernel, outHeight, inputHeight, rowScale, 0.f, antialias, rowSpans);

        } break;
        case kResizeMitchellcubic:{
            MitchellCubicKernelFunc kernel;
            res = computeSpans(context, kernel, outWidth, inputWidth, columnScale, 0.f, antialias,
                               colSpans);
            if (res != Status::OK()) return res;
            res = computeSpans(context, kernel, outHeight, inputHeight, rowScale, 0.f, antialias, rowSpans);

        } break;
    };

    NDArray intermediate = NDArrayFactory::create<Z>('c', {batchSize, outHeight, inputWidth, channels});

    //const functor::Spans& const_row_spans = row_spans;
    //typename TTypes<int32, 1>::ConstTensor row_starts(
    //const_row_spans.starts.tensor<int32, 1>());
    auto& rowStarts = rowSpans._starts; // shape {outWidth}
    auto& rowWeights = rowSpans._weights; // shape {outWidth, numSpans}
    auto& columnStarts = colSpans._starts; // shape {outHeights}
    auto& columnWeights = colSpans._weights; // shape {outHeights, numSpans}

    gatherSpans<X, Z>(context, rowSpans._spanSize, rowStarts, rowWeights, colSpans._spanSize, columnStarts, columnWeights, input, intermediate, output);

    NDArray::registerSpecialUse({output}, {input});
    return res;
}


static int resizeTriangle(sd::LaunchContext * context, NDArray const* image, int const width, int const height, bool const antialias, NDArray* output) {
//    std::unique_ptr<IKernelFunc> kernel(new TriangleKernelFunc);
    BUILD_DOUBLE_SELECTOR(image->dataType(), output->dataType(), return resizeKernel,(context, kResizeBilinear, image, width, height, antialias, output), NUMERIC_TYPES, FLOAT_TYPES_1);
    return Status::CODE(ND4J_STATUS_VALIDATION, "helpers::resizeTriangle: This resize method is avaliable in future versions");
}

static int resizeLanczos3(sd::LaunchContext * context, NDArray const* image, int const width, int const height, bool const antialias, NDArray* output) {
//    std::unique_ptr<IKernelFunc> kernel(new LanczosKernelFunc(3.f));
    BUILD_DOUBLE_SELECTOR(image->dataType(), output->dataType(), return resizeKernel,(context, kResizeLanczos3, image, width, height, antialias, output), NUMERIC_TYPES, FLOAT_TYPES_1);
    return Status::CODE(ND4J_STATUS_VALIDATION, "helpers::resizeLanczos3: This resize method is avaliable in future versions");
}

static int resizeLanczos5(sd::LaunchContext * context, NDArray const* image, int const width, int const height, bool const antialias, NDArray* output) {
//    std::unique_ptr<IKernelFunc> kernel(new LanczosKernelFunc(5.f));
    BUILD_DOUBLE_SELECTOR(image->dataType(), output->dataType(), return resizeKernel,(context, kResizeLanczos5, image, width, height, antialias, output), NUMERIC_TYPES, FLOAT_TYPES_1);
    return Status::CODE(ND4J_STATUS_VALIDATION, "helpers::resizeLanczos5: This resize method is avaliable in future versions");
}

static int resizeGaussian(sd::LaunchContext * context, NDArray const* image, int const width, int const height, bool const antialias, NDArray* output) {
    BUILD_DOUBLE_SELECTOR(image->dataType(), output->dataType(), return resizeKernel,(context, kResizeGaussian, image, width, height, antialias, output), NUMERIC_TYPES, FLOAT_TYPES_1);
    return Status::CODE(ND4J_STATUS_VALIDATION, "helpers::resizeGaussian: This resize method is avaliable in future versions");
}
static int resizeMitchellcubic(sd::LaunchContext * context, NDArray const* image, int const width, int const height, bool const antialias, NDArray* output) {
    BUILD_DOUBLE_SELECTOR(image->dataType(), output->dataType(), return resizeKernel,(context, kResizeMitchellcubic, image, width, height, antialias, output), NUMERIC_TYPES, FLOAT_TYPES_1);
    return Status::CODE(ND4J_STATUS_VALIDATION, "helpers::resizeMitchelcubic: This resize method is avaliable in future versions");
}
static int resizeKeycubic(sd::LaunchContext * context, NDArray const* image, int const width, int const height, bool const antialias, NDArray* output) {
    if (!antialias)
        return resizeBicubicFunctorA(context, image, width, height, false, true, output);
    BUILD_DOUBLE_SELECTOR(image->dataType(), output->dataType(), return resizeKernel,(context, kResizeBicubic, image, width, height, antialias, output), NUMERIC_TYPES, FLOAT_TYPES_1);
    return Status::CODE(ND4J_STATUS_VALIDATION, "helpers::resizeKeycubic: This resize method is avaliable in future versions");
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int resizeFunctor(sd::LaunchContext * context, NDArray const* image, int width, int height,
                  ImageResizeMethods method, bool antialias, NDArray* output) {
    switch (method) {
        case kResizeBilinear:     return resizeTriangle(context, image, width, height, antialias, output);
        case kResizeNearest:      return resizeNeighborFunctor(context, image, width, height, false, true, output);
        case kResizeBicubic:      return resizeKeycubic(context, image, width, height, antialias, output);
        case kResizeLanczos3:     return resizeLanczos3(context, image, width, height, antialias, output);
        case kResizeLanczos5:     return resizeLanczos5(context, image, width, height, antialias, output);
        case kResizeGaussian:     return resizeGaussian(context, image, width, height, antialias, output);
        case kResizeArea:         return resizeAreaFunctor(context, image, width, height, false, output);
        case kResizeMitchellcubic: return resizeMitchellcubic(context, image, width, height, antialias, output);
        default:
            nd4j_printf("helper::resizeFunctor: Wrong resize method %i\n", (int)method);
            throw std::runtime_error("helper::resizeFunctor: Wrong resize method.");
    }
    return ND4J_STATUS_OK;
}


        }
    }
}