/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
//  @author GS <sgazeos@gmail.com>
//

#include <ops/declarable/helpers/bds.h>
#include <Status.h>


namespace nd4j {
namespace ops {
namespace helpers {


    template <typename T>
    static __global__ void bdsLoopKernel(void const* inputX, Nd4jLong const* inputXshape, void const* inputY, Nd4jLong const* inputYshape, void* output, Nd4jLong* outputShape) {
        __shared__ T const* x;
        __shared__ T const* y;
        __shared__ T* z;
        __shared__ bool speedWay;
        //__shared__ int indexX, indexY;
        __shared__ Nd4jLong xLen, yLen, outputLen;
        if (threadIdx.x == 0) {
            x = reinterpret_cast<T const*>(inputX);
            y = reinterpret_cast<T const*>(inputY);
            z = reinterpret_cast<T*>(output);
            xLen = shape::length(inputXshape);
            yLen = shape::length(inputYshape);
            outputLen = shape::length(outputShape);
            speedWay = speedWay && shape::elementWiseStride(inputXshape) == 1;
            speedWay = speedWay && shape::elementWiseStride(inputYshape) == 1;
            speedWay = speedWay && shape::elementWiseStride(outputShape) == 1;

        }
        __syncthreads();

        auto tid = threadIdx.x + blockIdx.x * blockDim.x;
        auto step = blockDim.x * gridDim.x;
        for (int e = tid; e < outputLen; e += step) {
            T val;
            if (speedWay) {
                if (e < nd4j::math::nd4j_min(yLen, xLen)) {
                    val = nd4j::math::nd4j_max(x[e], y[e]);
                } else if (e < xLen) {
                    val = nd4j::math::nd4j_max(x[e], y[yLen - 1]);
                } else {
                    val = nd4j::math::nd4j_max(x[xLen - 1], y[e]);
                }
                z[e] = val;
            }
            else {
                auto xIndex = e < xLen?shape::getIndexOffset(e, inputXshape, xLen):shape::getIndexOffset(xLen, inputXshape, xLen);
                auto yIndex = e < yLen?shape::getIndexOffset(e, inputYshape, yLen):shape::getIndexOffset(yLen - 1, inputYshape, yLen);
                auto zIndex = shape::getIndexOffset(e, outputShape, outputLen);
                z[zIndex] = nd4j::math::nd4j_max(x[xIndex], y[yIndex]);
            }
        }
    }
    template <typename T>
    static void bdsLoopH(cudaStream_t* stream, void const* inputX, Nd4jLong const* inputXshape, void const* inputY, Nd4jLong const* inputYshape, void* output, Nd4jLong* outputShape) {
        bdsLoopKernel<T><<<128, 256, 512, *stream>>>(inputX, inputXshape, inputY, inputYshape, output, outputShape);

    }

    Nd4jStatus bdsFunctor(graph::LaunchContext* context, NDArray* x_shape, NDArray* y_shape, NDArray* output) {
        //int e = 0, x = 0, y = 0;

        if (x_shape->lengthOf() == 1 || y_shape->lengthOf() == 1) {// except case
            auto lesser = (x_shape->lengthOf() == 1 ? x_shape: y_shape);
            auto greater = (x_shape->lengthOf() == 1 ? y_shape: x_shape);
            output->assign(greater);
            output->syncToHost();
            output->p(output->lengthOf() - 1, *lesser);
            output->syncToDevice();
        }
        else {
            //bdsLoopH(context->getCudaStream(), x->getSpecialBuffer(), x->getSpecialShapeInfo(), y->getSpecialBuffer(), y->getSpecialShape(), output->specialBuffer(), output->specialShapeInfo())
            BUILD_SINGLE_SELECTOR(output->dataType(), bdsLoopH, (context->getCudaStream(), x_shape->getSpecialBuffer(), x_shape->getSpecialShapeInfo(), y_shape->getSpecialBuffer(), y_shape->getSpecialShapeInfo(), output->specialBuffer(), output->specialShapeInfo()), NUMERIC_TYPES);
        }
        return Status::OK();
        return Status::OK();
    }
    BUILD_SINGLE_TEMPLATE(template void bdsLoopH, (cudaStream_t* stream, void const* inputX, Nd4jLong const* inputXshape, void const* inputY, Nd4jLong const* inputYshape, void* output, Nd4jLong* outputShape), NUMERIC_TYPES);

}
}
}