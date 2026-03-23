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

#include <execution/Threads.h>
#include <helpers/MmulHelper.h>
#include <array/NDArrayFactory.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/gated_delta_net_block.h>
#include <ops/declarable/helpers/gated_delta_rule.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void gatedDeltaNetBlock_(LaunchContext* context,
                                 NDArray* x, NDArray* Wqkv, NDArray* Wbeta, NDArray* Wgate,
                                 NDArray* Wout, NDArray* convWeight, NDArray* convBias,
                                 NDArray* stateIn,
                                 NDArray* output, NDArray* recurrentStateOut, NDArray* convStateOut,
                                 int numHeads, int headDimK, int headDimV, float rmsEps) {
    const auto B = x->sizeAt(0);
    const auto L = x->sizeAt(1);
    const auto D = x->sizeAt(2);
    const LongType BL = B * L;

    // Linear projections: x[BL,D] @ W -> projected
    std::vector<LongType> xReshapeVec = {BL, D};
    auto xReshaped = x->reshape('c', xReshapeVec);
    auto qkvProjected = MmulHelper::mmul(xReshaped, Wqkv);
    auto betaProjected = MmulHelper::mmul(xReshaped, Wbeta);
    auto gateProjected = MmulHelper::mmul(xReshaped, Wgate);

    // Reshape beta/gate to [B, L, numHeads]
    std::vector<LongType> bgShape = {B, L, static_cast<LongType>(numHeads)};
    auto betaReshaped = betaProjected->reshape('c', bgShape);
    auto gateReshaped = gateProjected->reshape('c', bgShape);

    // Split QKV projection into Q, K, V per head
    const LongType qkvDim = numHeads * (2 * headDimK + headDimV);
    std::vector<LongType> qkvReshapeVec = {B, L, qkvDim};
    auto qkvReshaped = qkvProjected->reshape('c', qkvReshapeVec);

    std::vector<LongType> qkShape = {B, L, static_cast<LongType>(numHeads), static_cast<LongType>(headDimK)};
    std::vector<LongType> vShape = {B, L, static_cast<LongType>(numHeads), static_cast<LongType>(headDimV)};
    auto Q = NDArrayFactory::create<T>('c', qkShape);
    auto Kmat = NDArrayFactory::create<T>('c', qkShape);
    auto V = NDArrayFactory::create<T>('c', vShape);

    const T* qkvBuf = qkvReshaped->bufferAsT<T>();
    T* qBuf = Q->bufferAsT<T>();
    T* kBuf = Kmat->bufferAsT<T>();
    T* vBuf = V->bufferAsT<T>();

    // QKV split: interleaved [q0,k0,v0, q1,k1,v1, ...] per head
    auto splitFunc = PRAGMA_THREADS_FOR {
        for (auto idx = start; idx < stop; ++idx) {
            LongType offset = 0;
            for (LongType h = 0; h < numHeads; ++h) {
                for (LongType dk = 0; dk < headDimK; ++dk)
                    qBuf[(idx * numHeads + h) * headDimK + dk] = qkvBuf[idx * qkvDim + offset++];
                for (LongType dk = 0; dk < headDimK; ++dk)
                    kBuf[(idx * numHeads + h) * headDimK + dk] = qkvBuf[idx * qkvDim + offset++];
                for (LongType dv = 0; dv < headDimV; ++dv)
                    vBuf[(idx * numHeads + h) * headDimV + dv] = qkvBuf[idx * qkvDim + offset++];
            }
        }
    };
    samediff::Threads::parallel_tad(splitFunc, 0, BL);

    // Sigmoid on beta
    T* betaBuf = betaReshaped->bufferAsT<T>();
    const LongType betaLen = betaReshaped->lengthOf();
    auto sigmoidFunc = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; ++i)
            betaBuf[i] = static_cast<T>(1) / (static_cast<T>(1) + sd::math::sd_exp<T, T>(-betaBuf[i]));
    };
    samediff::Threads::parallel_for(sigmoidFunc, 0, betaLen);

    // L2-normalize K per vector
    auto normFunc = PRAGMA_THREADS_FOR {
        for (auto idx = start; idx < stop; ++idx) {
            const LongType base = idx * headDimK;
            T normSq = static_cast<T>(0);
            for (LongType dk = 0; dk < headDimK; ++dk)
                normSq += kBuf[base + dk] * kBuf[base + dk];
            T invNorm = static_cast<T>(1) / sd::math::sd_sqrt<T, T>(normSq + static_cast<T>(1e-12));
            for (LongType dk = 0; dk < headDimK; ++dk)
                kBuf[base + dk] *= invNorm;
        }
    };
    samediff::Threads::parallel_tad(normFunc, 0, BL * numHeads);

    // Gated delta rule recurrence
    auto gdnOutput = NDArrayFactory::create<T>('c', vShape);
    gatedDeltaRule(context, Q, Kmat, V, betaReshaped, gateReshaped,
                   stateIn, gdnOutput, recurrentStateOut);

    // RMS normalization on output
    const LongType hdv = static_cast<LongType>(numHeads) * headDimV;
    std::vector<LongType> flatShape = {BL, hdv};
    auto gdnFlat = gdnOutput->reshape('c', flatShape);
    T* flatBuf = gdnFlat->bufferAsT<T>();

    auto rmsFunc = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; ++row) {
            T* rowPtr = flatBuf + row * hdv;
            T sumSq = static_cast<T>(0);
            for (LongType j = 0; j < hdv; ++j) sumSq += rowPtr[j] * rowPtr[j];
            T invRms = static_cast<T>(1) / sd::math::sd_sqrt<T, T>(sumSq / static_cast<T>(hdv) + static_cast<T>(rmsEps));
            for (LongType j = 0; j < hdv; ++j) rowPtr[j] *= invRms;
        }
    };
    samediff::Threads::parallel_tad(rmsFunc, 0, BL);

    // Output projection
    auto finalOutput = MmulHelper::mmul(gdnFlat, Wout);
    std::vector<LongType> finalReshapeVec = {B, L, D};
    auto finalReshaped = finalOutput->reshape('c', finalReshapeVec);
    output->assign(finalReshaped);

    convStateOut->nullify();

    delete xReshaped;
    delete qkvProjected;
    delete betaProjected;
    delete gateProjected;
    delete betaReshaped;
    delete gateReshaped;
    delete qkvReshaped;
    delete Q;
    delete Kmat;
    delete V;
    delete gdnOutput;
    delete gdnFlat;
    delete finalOutput;
    delete finalReshaped;
}

void gatedDeltaNetBlock(LaunchContext* context,
                         NDArray* x, NDArray* Wqkv, NDArray* Wbeta, NDArray* Wgate,
                         NDArray* Wout, NDArray* convWeight, NDArray* convBias,
                         NDArray* stateIn,
                         NDArray* output, NDArray* recurrentStateOut, NDArray* convStateOut,
                         int numHeads, int headDimK, int headDimV, float rmsEps) {
    NDArray::preparePrimaryUse({output, recurrentStateOut, convStateOut}, {x, Wqkv, Wbeta, Wgate, Wout, convWeight, convBias});
    if (stateIn != nullptr) NDArray::preparePrimaryUse({}, {stateIn});

    BUILD_SINGLE_SELECTOR(x->dataType(), gatedDeltaNetBlock_,
        (context, x, Wqkv, Wbeta, Wgate, Wout, convWeight, convBias, stateIn,
         output, recurrentStateOut, convStateOut,
         numHeads, headDimK, headDimV, rmsEps), SD_FLOAT_TYPES);

    NDArray::registerPrimaryUse({output, recurrentStateOut, convStateOut}, {x, Wqkv, Wbeta, Wgate, Wout, convWeight, convBias});
    if (stateIn != nullptr) NDArray::registerPrimaryUse({}, {stateIn});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
