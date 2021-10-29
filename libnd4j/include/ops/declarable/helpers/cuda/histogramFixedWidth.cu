/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 31.08.2018
//
#include <exceptions/cuda_exception.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/histogramFixedWidth.h>

namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
template <typename X, typename Z>
SD_KERNEL static void histogramFixedWidthCuda(const void* vx, const sd::LongType* xShapeInfo, void* vz,
                                              const sd::LongType* zShapeInfo, const X leftEdge, const X rightEdge) {
  const auto x = reinterpret_cast<const X*>(vx);
  auto z = reinterpret_cast<Z*>(vz);

  __shared__ sd::LongType xLen, zLen, totalThreads, nbins;
  __shared__ X binWidth, secondEdge, lastButOneEdge;

  if (threadIdx.x == 0) {
    xLen = shape::length(xShapeInfo);
    nbins = shape::length(zShapeInfo);  // nbins = zLen
    totalThreads = gridDim.x * blockDim.x;

    binWidth = (rightEdge - leftEdge) / nbins;
    secondEdge = leftEdge + binWidth;
    lastButOneEdge = rightEdge - binWidth;
  }

  __syncthreads();

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (sd::LongType i = tid; i < xLen; i += totalThreads) {
    const X value = x[shape::getIndexOffset(i, xShapeInfo)];

    sd::LongType zIndex;

    if (value < secondEdge)
      zIndex = 0;
    else if (value >= lastButOneEdge)
      zIndex = nbins - 1;
    else
      zIndex = static_cast<sd::LongType>((value - leftEdge) / binWidth);

    sd::math::atomics::sd_atomicAdd<Z>(&z[shape::getIndexOffset(zIndex, zShapeInfo)], 1);
  }
}

///////////////////////////////////////////////////////////////////
template <typename X, typename Z>
SD_HOST static void histogramFixedWidthCudaLauncher(const cudaStream_t* stream, const NDArray& input,
                                                    const NDArray& range, NDArray& output) {
  const X leftEdge = range.e<X>(0);
  const X rightEdge = range.e<X>(1);

  histogramFixedWidthCuda<X, Z><<<256, 256, 1024, *stream>>>(input.specialBuffer(), input.specialShapeInfo(),
                                                             output.specialBuffer(), output.specialShapeInfo(),
                                                             leftEdge, rightEdge);
}

////////////////////////////////////////////////////////////////////////
void histogramFixedWidth(sd::LaunchContext* context, const NDArray& input, const NDArray& range, NDArray& output) {
  // firstly initialize output with zeros
  output.nullify();

  PointersManager manager(context, "histogramFixedWidth");

  NDArray::prepareSpecialUse({&output}, {&input});
  BUILD_DOUBLE_SELECTOR(input.dataType(), output.dataType(), histogramFixedWidthCudaLauncher,
                        (context->getCudaStream(), input, range, output), SD_COMMON_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({&output}, {&input});

  manager.synchronize();
}

//     template <typename T>
//     SD_KERNEL static void copyBuffers(sd::LongType* destination, void const* source, sd::LongType* sourceShape,
//     sd::LongType bufferLength) {
//         const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
//         const auto step = gridDim.x * blockDim.x;
//         for (int t = tid; t < bufferLength; t += step) {
//             destination[t] = reinterpret_cast<T const*>(source)[shape::getIndexOffset(t, sourceShape)];
//         }
//     }

//     template <typename T>
//     SD_KERNEL static void returnBuffers(void* destination, sd::LongType const* source, sd::LongType*
//     destinationShape, sd::LongType bufferLength) {
//         const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
//         const auto step = gridDim.x * blockDim.x;
//         for (int t = tid; t < bufferLength; t += step) {
//             reinterpret_cast<T*>(destination)[shape::getIndexOffset(t, destinationShape)] = source[t];
//         }
//     }

//     template <typename T>
//     static SD_KERNEL void histogramFixedWidthKernel(void* outputBuffer, sd::LongType outputLength, void const*
//     inputBuffer, sd::LongType* inputShape, sd::LongType inputLength, double const leftEdge, double binWidth, double
//     secondEdge, double lastButOneEdge) {

//         __shared__ T const* x;
//         __shared__ sd::LongType* z; // output buffer

//         if (threadIdx.x == 0) {
//             z = reinterpret_cast<sd::LongType*>(outputBuffer);
//             x = reinterpret_cast<T const*>(inputBuffer);
//         }
//         __syncthreads();
//         auto tid = blockIdx.x * gridDim.x + threadIdx.x;
//         auto step = blockDim.x * gridDim.x;

//         for(auto i = tid; i < inputLength; i += step) {

//             const T value = x[shape::getIndexOffset(i, inputShape)];
//             sd::LongType currInd = static_cast<sd::LongType>((value - leftEdge) / binWidth);

//             if(value < secondEdge)
//                 currInd = 0;
//             else if(value >= lastButOneEdge)
//                 currInd = outputLength - 1;
//             sd::math::atomics::sd_atomicAdd(&z[currInd], 1LL);
//         }
//     }

//     template <typename T>
//     void histogramFixedWidth_(sd::LaunchContext * context, const NDArray& input, const NDArray& range, NDArray&
//     output) {
//         const int nbins = output.lengthOf();
//         auto stream = context->getCudaStream();
//         // firstly initialize output with zeros
//         //if(output.ews() == 1)
//         //    memset(output.buffer(), 0, nbins * output.sizeOfT());
//         //else
//         output.assign(0);
//         if (!input.isActualOnDeviceSide())
//             input.syncToDevice();

//         const double leftEdge  = range.e<double>(0);
//         const double rightEdge = range.e<double>(1);

//         const double binWidth       = (rightEdge - leftEdge ) / nbins;
//         const double secondEdge     = leftEdge + binWidth;
//         double lastButOneEdge = rightEdge - binWidth;
//         sd::LongType* outputBuffer;
//         cudaError_t err = cudaMalloc(&outputBuffer, output.lengthOf() * sizeof(sd::LongType));
//         if (err != 0)
//             throw cuda_exception::build("helpers::histogramFixedWidth: Cannot allocate memory for output", err);
//         copyBuffers<sd::LongType ><<<256, 512, 8192, *stream>>>(outputBuffer, output.specialBuffer(),
//         output.special(), output.lengthOf()); histogramFixedWidthKernel<T><<<256, 512, 8192, *stream>>>(outputBuffer,
//         output.lengthOf(), input.specialBuffer(), input.special(), input.lengthOf(), leftEdge, binWidth, secondEdge,
//         lastButOneEdge); returnBuffers<sd::LongType><<<256, 512, 8192, *stream>>>(output.specialBuffer(),
//         outputBuffer, output.special(), output.lengthOf());
//         //cudaSyncStream(*stream);
//         err = cudaFree(outputBuffer);
//         if (err != 0)
//             throw cuda_exception::build("helpers::histogramFixedWidth: Cannot deallocate memory for output buffer",
//             err);
//         output.tickWriteDevice();
// //#pragma omp parallel for schedule(guided)
// //        for(sd::LongType i = 0; i < input.lengthOf(); ++i) {
// //
// //            const T value = input.e<T>(i);
// //
// //            if(value < secondEdge)
// //#pragma omp critical
// //                output.p<sd::LongType>(0, output.e<sd::LongType>(0) + 1);
// //            else if(value >= lastButOneEdge)
// //#pragma omp critical
// //                output.p<sd::LongType>(nbins-1, output.e<sd::LongType>(nbins-1) + 1);
// //            else {
// //                sd::LongType currInd = static_cast<sd::LongType>((value - leftEdge) / binWidth);
// //#pragma omp critical
// //                output.p<sd::LongType>(currInd, output.e<sd::LongType>(currInd) + 1);
// //            }
// //        }
//     }

//     void histogramFixedWidth(sd::LaunchContext * context, const NDArray& input, const NDArray& range, NDArray&
//     output) {
//         BUILD_SINGLE_SELECTOR(input.dataType(), histogramFixedWidth_, (context, input, range, output),
//         SD_COMMON_TYPES);
//     }
//     BUILD_SINGLE_TEMPLATE(template void histogramFixedWidth_, (sd::LaunchContext * context, const NDArray& input,
//     const NDArray& range, NDArray& output), SD_COMMON_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
