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
// @author raver119@gmail.com
//


#include<ops/declarable/helpers/meshgrid.h>
#include <helpers/PointersManager.h>
#include <helpers/ConstantTadHelper.h>
#include <array/ResultSet.h>
#include <numeric>

namespace sd 	  {
namespace ops 	  {
namespace helpers {

    template <typename T>
    static _CUDA_D void assign_(void *vx, Nd4jLong *xShapeInfo, void *vz, Nd4jLong *zShapeInfo) {
        auto x = reinterpret_cast<T*>(vx);
        auto z = reinterpret_cast<T*>(vz);

        auto tid = threadIdx.x + blockIdx.x * blockDim.x;

        auto xEws = shape::elementWiseStride(xShapeInfo);
        auto zEws = shape::elementWiseStride(zShapeInfo);

        auto xOrder = shape::order(xShapeInfo);
        auto zOrder = shape::order(zShapeInfo);

        __shared__ Nd4jLong length;

        if (threadIdx.x == 0) {
            length = shape::length(xShapeInfo);
        }
        __syncthreads();

        if (xEws > 0 && zEws > 0 && xOrder == zOrder) {
            for (int i = threadIdx.x; i < length; i += blockDim.x) {
                z[i * zEws] = x[i * xEws];
            }
        } else {
            for (int i = threadIdx.x; i < length; i += blockDim.x) {
                auto xOffset = shape::getIndexOffset(i, xShapeInfo);
                auto zOffset = shape::getIndexOffset(i, zShapeInfo);

                z[zOffset] = x[xOffset];
            }
        }

    }

    template <typename T>
    static _CUDA_G void meshgridKernel(int rank, void **outBuffers, Nd4jLong **tadShapes, Nd4jLong **tadOffsets, Nd4jLong *numTads, void **inBuffers, Nd4jLong **inShapes) {
        // for all arrays
        for (int i = blockIdx.x; i < rank; i += gridDim.x) {

            // for all tads in this array
            for(Nd4jLong j = 0; j < numTads[i]; j++) {
                assign_<T>(inBuffers[i], inShapes[i], reinterpret_cast<T*>(outBuffers[i]) + tadOffsets[i][j], tadShapes[i]);
            }
            __syncthreads();
        }
    }

    template <typename T>
    static void meshgrid_(sd::LaunchContext * context, const std::vector<NDArray*>& inArrs, const std::vector<NDArray*>& outArrs, const bool swapFirst2Dims) {
        const int rank = inArrs.size();
        int inIndices[MAX_RANK];
        std::iota(inIndices, inIndices + rank, 0);
        if(swapFirst2Dims && rank > 1) {
            inIndices[0] = 1;
            inIndices[1] = 0;
        }

        PointersManager pm(context, "meshgrid");
        std::vector<const void *> hInBuffers(rank);
        std::vector<void *> hOutBuffers(rank);
        std::vector<const Nd4jLong *> hInShapes(rank);

        std::vector<const Nd4jLong *> hOutTadShapes(rank);
        std::vector<const Nd4jLong *> hOutTadOffsets(rank);

        std::vector<Nd4jLong> hNumTads(rank);

        for(int i = 0; i < rank; ++i) {
            hInBuffers[i] = inArrs[i]->specialBuffer();
            hInShapes[i] = inArrs[i]->specialShapeInfo();

            hOutBuffers[i] = outArrs[i]->specialBuffer();


            auto pack = ConstantTadHelper::getInstance().tadForDimensions(outArrs[i]->shapeInfo(), {inIndices[i]});
            hOutTadShapes[i] = pack.specialShapeInfo();
            hOutTadOffsets[i] = pack.specialOffsets();
            hNumTads[i] = pack.numberOfTads();


            //auto list = outArrs[i]->allTensorsAlongDimension({inIndices[i]});
            //for(int j = 0; j < list->size(); ++j)
            //    list->at(j)->assign(inArrs[i]);

            //delete list;
        }

        auto dInBuffers = reinterpret_cast<void **>(pm.replicatePointer(hInBuffers.data(), hInBuffers.size() * sizeof(void *)));
        auto dOutBuffers = reinterpret_cast<void **>(pm.replicatePointer(hOutBuffers.data(), hOutBuffers.size() * sizeof(void *)));


        auto dInShapes = reinterpret_cast<Nd4jLong **>(pm.replicatePointer(hInShapes.data(), hInShapes.size() * sizeof(Nd4jLong *)));
        auto dOutTadShapes = reinterpret_cast<Nd4jLong **>(pm.replicatePointer(hOutTadShapes.data(), hOutTadShapes.size() * sizeof(Nd4jLong *)));
        auto dOutTadOffsets = reinterpret_cast<Nd4jLong **>(pm.replicatePointer(hOutTadOffsets.data(), hOutTadOffsets.size() * sizeof(Nd4jLong *)));

        auto dNumTads = reinterpret_cast<Nd4jLong *>(pm.replicatePointer(hNumTads.data(), hNumTads.size() * sizeof(Nd4jLong)));


        meshgridKernel<T><<<256, 256, 1024, *context->getCudaStream()>>>(rank, dOutBuffers, dOutTadShapes, dOutTadOffsets, dNumTads, dInBuffers, dInShapes);

        pm.synchronize();
    }

    //////////////////////////////////////////////////////////////////////////
    void meshgrid(sd::LaunchContext * context, const std::vector<NDArray*>& inArrs, const std::vector<NDArray*>& outArrs, const bool swapFirst2Dims) {

        BUILD_SINGLE_SELECTOR(inArrs.at(0)->dataType(), meshgrid_, (context, inArrs, outArrs, swapFirst2Dims), NUMERIC_TYPES);

        for (auto v:outArrs)
            v->tickWriteDevice();
    }

}
}
}

