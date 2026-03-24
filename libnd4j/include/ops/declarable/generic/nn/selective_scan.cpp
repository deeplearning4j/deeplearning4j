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
// @author Adam Gibson
//
// selective_scan - Mamba selective state space scan
//
// Implements the selective scan (S6) operation from the Mamba architecture.
// For each time step t:
//   h_t = A_t * h_{t-1} + B_t * x_t
//   y_t = C_t * h_t + D * x_t
//
// Where A, B, C are input-dependent (selective), enabling content-based reasoning.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_selective_scan)

#include <system/common.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(selective_scan, 5, 1, false, 0, 0) {
    auto x = INPUT_VARIABLE(0);       // [batch, seq_len, dim]
    auto A = INPUT_VARIABLE(1);       // [batch, seq_len, state_dim] discretized A
    auto B = INPUT_VARIABLE(2);       // [batch, seq_len, state_dim] discretized B
    auto C = INPUT_VARIABLE(3);       // [batch, seq_len, state_dim] output projection
    auto D = INPUT_VARIABLE(4);       // [dim] skip connection (feed-through)
    auto output = OUTPUT_VARIABLE(0); // [batch, seq_len, dim]

    // Optional: initial hidden state
    NDArray* h0 = block.width() > 5 ? INPUT_VARIABLE(5) : nullptr;

    auto batch = x->sizeAt(0);
    auto seqLen = x->sizeAt(1);
    auto dim = x->sizeAt(2);
    auto stateDim = A->sizeAt(2);

    // CPU fallback: sequential scan
    // h: [batch, dim, state_dim]
    auto h = NDArrayFactory::create<float>('c', {batch, dim, stateDim});
    if (h0 != nullptr) {
        h->assign(h0);
    }

    // Sync all inputs to host ONCE before the loop to avoid per-element sync overhead
    x->syncToHost();
    A->syncToHost();
    B->syncToHost();
    C->syncToHost();
    D->syncToHost();
    h->syncToHost();

    // Get typed buffers for direct access — eliminates O(n^2) sync from p()/e() in loops
    const float* xBuf = x->bufferAsT<float>();
    const float* aBuf = A->bufferAsT<float>();
    const float* bBuf = B->bufferAsT<float>();
    const float* cBuf = C->bufferAsT<float>();
    const float* dBuf = D->bufferAsT<float>();
    float* hBuf = h->bufferAsT<float>();
    float* outBuf = output->bufferAsT<float>();

    // Cache strides for offset computation — inputs may be non-contiguous (views)
    // x, A, B, C are rank-3: [batch, seq_len, dim/state_dim]
    const auto xStride0 = x->strideAt(0);  const auto xStride1 = x->strideAt(1);  const auto xStride2 = x->strideAt(2);
    const auto aStride0 = A->strideAt(0);  const auto aStride1 = A->strideAt(1);  const auto aStride2 = A->strideAt(2);
    const auto bStride0 = B->strideAt(0);  const auto bStride1 = B->strideAt(1);  const auto bStride2 = B->strideAt(2);
    const auto cStride0 = C->strideAt(0);  const auto cStride1 = C->strideAt(1);  const auto cStride2 = C->strideAt(2);
    // D is rank-1: [dim]
    const auto dStride0 = D->strideAt(0);
    // h is locally created 'c' order rank-3: [batch, dim, state_dim]
    const auto hStride0 = h->strideAt(0);  const auto hStride1 = h->strideAt(1);  const auto hStride2 = h->strideAt(2);
    // output is rank-3: [batch, seq_len, dim]
    const auto oStride0 = output->strideAt(0);  const auto oStride1 = output->strideAt(1);  const auto oStride2 = output->strideAt(2);

    for (sd::LongType t = 0; t < seqLen; ++t) {
        for (sd::LongType b_idx = 0; b_idx < batch; ++b_idx) {
            for (sd::LongType d = 0; d < dim; ++d) {
                float x_val = xBuf[b_idx * xStride0 + t * xStride1 + d * xStride2];
                float y_val = 0.0f;

                for (sd::LongType s = 0; s < stateDim; ++s) {
                    float a_val = aBuf[b_idx * aStride0 + t * aStride1 + s * aStride2];
                    float b_val = bBuf[b_idx * bStride0 + t * bStride1 + s * bStride2];
                    float c_val = cBuf[b_idx * cStride0 + t * cStride1 + s * cStride2];

                    // State update: h = A * h + B * x
                    sd::LongType hOffset = b_idx * hStride0 + d * hStride1 + s * hStride2;
                    float h_prev = hBuf[hOffset];
                    float h_new = a_val * h_prev + b_val * x_val;
                    hBuf[hOffset] = h_new;

                    // Output contribution: y += C * h
                    y_val += c_val * h_new;
                }

                // Skip connection: y += D * x
                y_val += dBuf[d * dStride0] * x_val;

                outBuf[b_idx * oStride0 + t * oStride1 + d * oStride2] = y_val;
            }
        }
    }

    // Mark host-side writes and sync to device ONCE after all loops complete
    h->tickWriteHost();
    h->syncToDevice();
    output->tickWriteHost();
    output->syncToDevice();

    delete h;

    return sd::Status::OK;
}

DECLARE_TYPES(selective_scan) {
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(selective_scan) {
    auto xShape = inputShape->at(0);

    auto batch = shape::sizeAt(xShape, 0);
    auto seqLen = shape::sizeAt(xShape, 1);
    auto dim = shape::sizeAt(xShape, 2);

    auto outputShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(xShape), 'c', {batch, seqLen, dim});

    return SHAPELIST(outputShape);
}

}  // namespace ops
}  // namespace sd

#endif
