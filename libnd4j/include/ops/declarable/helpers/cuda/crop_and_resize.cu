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
//  @author sgazeos@gmail.com (CUDA implementation)
//
#include <ops/declarable/helpers/crop_and_resize.h>
#include <helpers/PointersManager.h>
#include <execution/cuda/LaunchDims.h>
#include <exceptions/cuda_exception.h>
#include "helpers/DebugHelper.h"

namespace sd {
namespace ops {
namespace helpers {

// CUDA kernel for the bilinear interpolation method (method = 0)
template <typename T, typename F, typename I>
static SD_KERNEL void cropAndResizeBilinearKernel(void* vimages, const LongType* imagesShapeInfo,
                                                 void* vboxes, const LongType* boxesShapeInfo,
                                                 void* vindices, const LongType* indicesShapeInfo,
                                                 int method, double extrapolationVal,
                                                 int batchSize, int imageHeight, int imageWidth,
                                                 int numBoxes, int cropHeight, int cropWidth, int depth,
                                                 void* vcrops, const LongType* cropsShapeInfo) {
    // Load pointers to the correct types
    const auto images = reinterpret_cast<const T*>(vimages);
    const auto boxes = reinterpret_cast<const F*>(vboxes);
    const auto indices = reinterpret_cast<const I*>(vindices);
    auto crops = reinterpret_cast<T*>(vcrops);
    
    // Get thread ID and strides
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;
    
    // Calculate the total number of output pixels
    int totalPixels = numBoxes * cropHeight * cropWidth * depth;
    
    // Process pixels in a strided fashion
    for (int p = tid; p < totalPixels; p += totalThreads) {
        // Convert flat index to 4D coordinates
        int pixelIndex = p;
        int d = pixelIndex % depth;
        pixelIndex /= depth;
        int x = pixelIndex % cropWidth;
        pixelIndex /= cropWidth;
        int y = pixelIndex % cropHeight;
        int b = pixelIndex / cropHeight;
        
        // Get box parameters and image index
        int boxOffset = b * 4;
        float y1 = static_cast<float>(boxes[boxOffset]);
        float x1 = static_cast<float>(boxes[boxOffset + 1]);
        float y2 = static_cast<float>(boxes[boxOffset + 2]);
        float x2 = static_cast<float>(boxes[boxOffset + 3]);
        
        int bIn = static_cast<int>(indices[b]);
        if (bIn >= batchSize) {
            // Skip if the box index is out of bounds
            continue;
        }
        
        // Calculate scales for interpolation
        float heightScale = (cropHeight > 1) 
            ? ((y2 - y1) * (imageHeight - 1) / (cropHeight - 1))
            : 0.0f;
        float widthScale = (cropWidth > 1)
            ? ((x2 - x1) * (imageWidth - 1) / (cropWidth - 1))
            : 0.0f;
        
        // Calculate input image coordinates
        float inY = (cropHeight > 1) 
            ? y1 * (imageHeight - 1) + y * heightScale 
            : 0.5f * (y1 + y2) * (imageHeight - 1);
        
        float inX = (cropWidth > 1) 
            ? x1 * (imageWidth - 1) + x * widthScale 
            : 0.5f * (x1 + x2) * (imageWidth - 1);
        
        // Skip if outside the input image
        if (inY < 0 || inY > imageHeight - 1 || inX < 0 || inX > imageWidth - 1) {
            // Set to extrapolation value
            LongType cropsOffset = (((b * cropHeight + y) * cropWidth + x) * depth) + d;
            crops[cropsOffset] = static_cast<T>(extrapolationVal);
            continue;
        }
        
        // Index calculation for the crops output array
        LongType cropsOffset = (((b * cropHeight + y) * cropWidth + x) * depth) + d;
        
        if (method == 0) { // Bilinear interpolation
            int topYIndex = static_cast<int>(floorf(inY));
            int bottomYIndex = static_cast<int>(ceilf(inY));
            float yLerp = inY - topYIndex;
            
            int leftXIndex = static_cast<int>(floorf(inX));
            int rightXIndex = static_cast<int>(ceilf(inX));
            float xLerp = inX - leftXIndex;
            
            // Compute indices for getting values from the images array
            LongType topLeftOffset = (((bIn * imageHeight + topYIndex) * imageWidth + leftXIndex) * depth) + d;
            LongType topRightOffset = (((bIn * imageHeight + topYIndex) * imageWidth + rightXIndex) * depth) + d;
            LongType bottomLeftOffset = (((bIn * imageHeight + bottomYIndex) * imageWidth + leftXIndex) * depth) + d;
            LongType bottomRightOffset = (((bIn * imageHeight + bottomYIndex) * imageWidth + rightXIndex) * depth) + d;
            
            // Get pixel values and do bilinear interpolation
            float topLeft = static_cast<float>(images[topLeftOffset]);
            float topRight = static_cast<float>(images[topRightOffset]);
            float bottomLeft = static_cast<float>(images[bottomLeftOffset]);
            float bottomRight = static_cast<float>(images[bottomRightOffset]);
            
            float top = topLeft + (topRight - topLeft) * xLerp;
            float bottom = bottomLeft + (bottomRight - bottomLeft) * xLerp;
            float value = top + (bottom - top) * yLerp;
            
            crops[cropsOffset] = static_cast<T>(value);
        } else { // Nearest neighbor interpolation
            int closestXIndex = static_cast<int>(roundf(inX));
            int closestYIndex = static_cast<int>(roundf(inY));
            
            // Compute index for getting value from the images array
            LongType imageOffset = (((bIn * imageHeight + closestYIndex) * imageWidth + closestXIndex) * depth) + d;
            
            crops[cropsOffset] = images[imageOffset];
        }
    }
}

template <typename T, typename F, typename I>
static void cropAndResizeFunctor_(LaunchContext* context, NDArray* images, NDArray* boxes, NDArray* indices,
                                  NDArray* cropSize, int method, double extrapolationVal, NDArray* crops) {
    // Prepare arrays for CUDA
    NDArray::prepareSpecialUse({crops}, {images, boxes, indices, cropSize});
    
    // Get dimensions from arrays
    const int batchSize = images->sizeAt(0);
    const int imageHeight = images->sizeAt(1);
    const int imageWidth = images->sizeAt(2);
    
    const int numBoxes = crops->sizeAt(0);
    const int cropHeight = crops->sizeAt(1);
    const int cropWidth = crops->sizeAt(2);
    const int depth = crops->sizeAt(3);
    
    // Calculate CUDA launch parameters
    dim3 launchDims = getLaunchDims("cropAndResize");
    
    // Launch CUDA kernel
    cropAndResizeBilinearKernel<T, F, I><<<launchDims.y, launchDims.x, launchDims.z, *context->getCudaStream()>>>(
        images->specialBuffer(), images->specialShapeInfo(),
        boxes->specialBuffer(), boxes->specialShapeInfo(),
        indices->specialBuffer(), indices->specialShapeInfo(),
        method, extrapolationVal,
        batchSize, imageHeight, imageWidth,
        numBoxes, cropHeight, cropWidth, depth,
        crops->specialBuffer(), crops->specialShapeInfo()
    );
    
    // Check for errors
    DebugHelper::checkErrorCode(context->getCudaStream(), "cropAndResizeFunctor CUDA kernel failed");
    
    // Finalize arrays after CUDA use
    NDArray::registerSpecialUse({crops}, {images, boxes, indices, cropSize});
}

// Wrapper function that dispatches to the correct typed implementation
void cropAndResizeFunctor(LaunchContext* context, NDArray* images, NDArray* boxes,
                          NDArray* indices, NDArray* cropSize, int method,
                          double extrapolationVal, NDArray* crops) {
    BUILD_TRIPLE_SELECTOR(images->dataType(), boxes->dataType(), indices->dataType(), cropAndResizeFunctor_,
                          (context, images, boxes, indices, cropSize, method, extrapolationVal, crops), 
                          SD_NUMERIC_TYPES, SD_FLOAT_TYPES, SD_INTEGER_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd