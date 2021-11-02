#include <array/NDArrayFactory.h>
#include <exceptions/cuda_exception.h>
#include <ops/declarable/helpers/image_resize.h>
#include <helpers/PointersManager.h>

namespace sd {
    namespace ops {
        namespace helpers {


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
static __global__ void batchedGatherSpan(  Nd4jLong outputWidth, Nd4jLong outputHeight, int rowSpanSize, int const* rowStartsBuf, Z const* rowWeightBuf, int columnSpanSize, int const* columnStartsBuf, Z const* columnWeightBuf,
     X const* pImages, const Nd4jLong* imageSpecialShapeInfo, Z* pIntermediate, Z* pOutput , Nd4jLong outputPixPerBatch ) {

    auto batchSize = shape::sizeAt(imageSpecialShapeInfo, 0);
    auto inputHeight = shape::sizeAt(imageSpecialShapeInfo, 1);
    auto inputWidth = shape::sizeAt(imageSpecialShapeInfo, 2);
    auto channels = shape::sizeAt(imageSpecialShapeInfo, 3);
    bool inputEws1 = shape::elementWiseStride(imageSpecialShapeInfo)==1;
    auto inputPixPerBatch = shape::strideAt(imageSpecialShapeInfo, 0);
    auto inRowStride = shape::strideAt(imageSpecialShapeInfo, 1);
    auto wStride = shape::strideAt(imageSpecialShapeInfo, 2);
    auto cStride = shape::strideAt(imageSpecialShapeInfo, 3);
    auto intermediatePixPerBatch = inputWidth * outputHeight * channels;
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto step = blockDim.x * gridDim.x;

    for (int b = tid; b < batchSize; b += step) {
        auto imagePtr = pImages + b * inputPixPerBatch;
        auto intermediatePtr = pIntermediate + b * intermediatePixPerBatch;
        auto outputPtr = pOutput + b * outputPixPerBatch;
        gatherRows<X, Z>(rowSpanSize, rowStartsBuf, rowWeightBuf,
                         imagePtr, inputHeight, inputWidth, outputHeight,
                         inputWidth, channels, intermediatePtr,  inputEws1, inRowStride, wStride, cStride);
        gatherColumns<Z>(columnSpanSize, columnStartsBuf, columnWeightBuf,
                         intermediatePtr, outputHeight, inputWidth, outputHeight, outputWidth, channels, outputPtr);
    }
}

template <typename X, typename Z>
static void gatherSpans(LaunchContext* context, int const rowSpanSize, NDArray const& rowStarts, NDArray const& rowWeights, int const colSpanSize, NDArray const& columnStarts, NDArray const& columnWeights, NDArray const* images, NDArray& intermediate, NDArray* output) {
 
    const auto imageSpecialShapeInfo = images->specialShapeInfo();
    auto outputHeight = output->sizeAt(1);
    auto outputWidth = output->sizeAt(2);
    auto channels = images->sizeAt(3);
    auto outputPixPerBatch = outputWidth * outputHeight * channels;
    auto intermediatePtr = reinterpret_cast<Z*>(intermediate.specialBuffer());

    auto imagePtr = reinterpret_cast<X const*>(images->specialBuffer());
    auto outputPtr = reinterpret_cast<Z*>(output->specialBuffer());
    auto stream = context->getCudaStream();
    auto rowStartsBuf = reinterpret_cast<int const*>(rowStarts.specialBuffer());
    auto rowWeightBuf = reinterpret_cast<Z const*>(rowWeights.specialBuffer());
    auto columnStartsBuf = reinterpret_cast<int const*>(columnStarts.specialBuffer());
    auto columnWeightBuf = reinterpret_cast<Z const*>(columnWeights.specialBuffer());
    batchedGatherSpan<X,Z><<<128, 128, 256, *stream>>>( outputWidth, outputHeight, rowSpanSize, 
        rowStartsBuf, rowWeightBuf, colSpanSize, columnStartsBuf, columnWeightBuf, imagePtr, imageSpecialShapeInfo, intermediatePtr, outputPtr, outputPixPerBatch );
}

template <typename X, typename Z>
static int resizeKernel(LaunchContext* context, ImageResizeMethods method, NDArray const* input, Nd4jLong outWidth, Nd4jLong outHeight, bool antialias, double coefficient, NDArray* output) {
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
            KeysCubicKernelFunc<float> kernel(static_cast<float>(coefficient));
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

#if defined(HAS_FLOAT32)
#define FLOAT_TYPES_FLOAT32 SKIP_FIRST_COMMA(TTYPE_FLOAT32)

static int resizeTriangle(sd::LaunchContext * context, NDArray const* image, int const width, int const height, bool const antialias, NDArray* output) {
//    std::unique_ptr<IKernelFunc> kernel(new TriangleKernelFunc);
    BUILD_DOUBLE_SELECTOR(image->dataType(), output->dataType(), return resizeKernel,(context, kResizeBilinear, image, width, height, antialias, 0, output), NUMERIC_TYPES, FLOAT_TYPES_FLOAT32);
    return Status::CODE(ND4J_STATUS_VALIDATION, "helpers::resizeTriangle: This resize method is avaliable in future versions");
}

static int resizeLanczos3(sd::LaunchContext * context, NDArray const* image, int const width, int const height, bool const antialias, NDArray* output) {
//    std::unique_ptr<IKernelFunc> kernel(new LanczosKernelFunc(3.f));
    BUILD_DOUBLE_SELECTOR(image->dataType(), output->dataType(), return resizeKernel,(context, kResizeLanczos3, image, width, height, antialias, 0, output), NUMERIC_TYPES, FLOAT_TYPES_FLOAT32);
    return Status::CODE(ND4J_STATUS_VALIDATION, "helpers::resizeLanczos3: This resize method is avaliable in future versions");
}

static int resizeLanczos5(sd::LaunchContext * context, NDArray const* image, int const width, int const height, bool const antialias, NDArray* output) {
//    std::unique_ptr<IKernelFunc> kernel(new LanczosKernelFunc(5.f));
    BUILD_DOUBLE_SELECTOR(image->dataType(), output->dataType(), return resizeKernel,(context, kResizeLanczos5, image, width, height, antialias, 0, output), NUMERIC_TYPES, FLOAT_TYPES_FLOAT32);
    return Status::CODE(ND4J_STATUS_VALIDATION, "helpers::resizeLanczos5: This resize method is avaliable in future versions");
}

static int resizeGaussian(sd::LaunchContext * context, NDArray const* image, int const width, int const height, bool const antialias, NDArray* output) {
    BUILD_DOUBLE_SELECTOR(image->dataType(), output->dataType(), return resizeKernel,(context, kResizeGaussian, image, width, height, antialias, 0, output), NUMERIC_TYPES, FLOAT_TYPES_FLOAT32);
    return Status::CODE(ND4J_STATUS_VALIDATION, "helpers::resizeGaussian: This resize method is avaliable in future versions");
}
static int resizeMitchellcubic(sd::LaunchContext * context, NDArray const* image, int const width, int const height, bool const antialias, NDArray* output) {
    BUILD_DOUBLE_SELECTOR(image->dataType(), output->dataType(), return resizeKernel,(context, kResizeMitchellcubic, image, width, height, antialias, 0, output), NUMERIC_TYPES, FLOAT_TYPES_FLOAT32);
    return Status::CODE(ND4J_STATUS_VALIDATION, "helpers::resizeMitchelcubic: This resize method is avaliable in future versions");
}

static int resizeBicubicA(sd::LaunchContext * context, NDArray const* image, int const width, int const height, CoordinateTransformationMode coorMode, bool exclude_outside, double coefficient, NDArray* output) {
    constexpr bool alignCorners = false;
    return resizeBicubicFunctorA(context,  image, width, height, alignCorners, coorMode, exclude_outside, coefficient,  output);
}

static int resizeBicubicAntialias(sd::LaunchContext * context, NDArray const* image, int const width, int const height, bool const antialias, double coefficient, NDArray* output) {
    BUILD_DOUBLE_SELECTOR(image->dataType(), output->dataType(), return resizeKernel,(context, kResizeBicubic, image, width, height, antialias, coefficient, output), NUMERIC_TYPES, FLOAT_TYPES_FLOAT32);
    return Status::CODE(ND4J_STATUS_VALIDATION, "helpers::resizeMitchelcubic: This resize method is avaliable in future versions");
}

#endif
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ND4J_LOCAL int resizeFunctor(sd::LaunchContext * context, NDArray const* image, int const width, int const height,
                    ImageResizeMethods method, CoordinateTransformationMode coorMode,  bool exclude_outside,
                    NearestMode nearestMode, double coefficient, bool antialias, NDArray* output) {
    switch (method) {

        case kResizeNearest:      return resizeNeighborFunctor(context, image, width, height, coorMode, nearestMode, false, output);
        case kResizeArea:         return resizeAreaFunctor(context, image, width, height, false, output);

#if defined(HAS_FLOAT32)
        case kResizeBilinear:     return resizeTriangle(context, image, width, height, antialias, output);
        case kResizeLanczos3:     return resizeLanczos3(context, image, width, height, antialias, output);
        case kResizeLanczos5:     return resizeLanczos5(context, image, width, height, antialias, output);
        case kResizeGaussian:     return resizeGaussian(context, image, width, height, antialias, output);
        case kResizeMitchellcubic: return resizeMitchellcubic(context, image, width, height, antialias, output);
        case kResizeBicubic:{
                //if antialias then coorMode is HALF_PIXEL and exlude_outside is true 
                if(antialias){
                    return resizeBicubicAntialias(context, image, width, height, antialias, coefficient, output );
                }
                else{
                    //use modified v1
                    return resizeBicubicA(context, image, width, height,  coorMode, exclude_outside, coefficient, output);
                }
            }
#else
        case kResizeBilinear:
        case kResizeLanczos3:
        case kResizeLanczos5:
        case kResizeGaussian:
        case kResizeMitchellcubic:
        case kResizeBicubic:{
            nd4j_printf("helper::resizeFunctor: only float type is supported by this resize method %i\n", (int)method);
            return Status::CODE(ND4J_STATUS_BAD_INPUT, "helper::resizeFunctor: only float type supported");
        }
#endif
        default:
            nd4j_printf("helper::resizeFunctor: Wrong resize method %i\n", (int)method);
            throw std::runtime_error("helper::resizeFunctor: Wrong resize method.");
    }
    return ND4J_STATUS_OK;
}


        }
    }
}