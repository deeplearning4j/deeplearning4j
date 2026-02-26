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

    for (sd::LongType t = 0; t < seqLen; ++t) {
        for (int b_idx = 0; b_idx < batch; ++b_idx) {
            for (sd::LongType d = 0; d < dim; ++d) {
                float x_val = x->e<float>(b_idx, t, d);
                float y_val = 0.0f;

                for (sd::LongType s = 0; s < stateDim; ++s) {
                    float a_val = A->e<float>(b_idx, t, s);
                    float b_val = B->e<float>(b_idx, t, s);
                    float c_val = C->e<float>(b_idx, t, s);

                    // State update: h = A * h + B * x
                    float h_prev = h->e<float>(b_idx, d, s);
                    float h_new = a_val * h_prev + b_val * x_val;
                    h->p(b_idx, d, s, h_new);

                    // Output contribution: y += C * h
                    y_val += c_val * h_new;
                }

                // Skip connection: y += D * x
                y_val += D->e<float>(d) * x_val;

                output->p(b_idx, t, d, y_val);
            }
        }
    }

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
