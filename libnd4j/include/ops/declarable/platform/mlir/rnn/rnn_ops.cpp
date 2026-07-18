/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// MLIR-accelerated RNN operations (LSTM, GRU, SimpleRNN)
//

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <system/platform_boilerplate.h>
#include <ops/declarable/platform/mlir/mlirUtils.h>

#if defined(HAVE_MLIR)

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
// lstmLayer MLIR implementation
// Full LSTM layer with configurable time direction
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(lstmLayer, ENGINE_CPU)

PLATFORM_IMPL(lstmLayer, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);          // [time, batch, input] or [batch, time, input]
    auto* Wx = INPUT_VARIABLE(1);         // input weights [input, 4*hidden]
    auto* Wr = INPUT_VARIABLE(2);         // recurrent weights [hidden, 4*hidden]
    auto* b = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;   // biases [4*hidden]
    auto* hI = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;  // initial hidden [batch, hidden]
    auto* cI = block.width() > 5 ? INPUT_VARIABLE(5) : nullptr;  // initial cell [batch, hidden]
    auto* Wp = block.width() > 6 ? INPUT_VARIABLE(6) : nullptr;  // peephole weights

    auto* h = OUTPUT_VARIABLE(0);         // all hidden states
    auto* hL = block.outputWidth() > 1 ? OUTPUT_VARIABLE(1) : nullptr;  // last hidden
    auto* cL = block.outputWidth() > 2 ? OUTPUT_VARIABLE(2) : nullptr;  // last cell

    // dataFormat: 0 = [time, batch, input], 1 = [batch, time, input]
    int dataFormat = INT_ARG(0);
    // directionMode: 0 = forward, 1 = backward, 2 = bidirectional sum, 3 = bidirectional concat
    int directionMode = INT_ARG(1);
    // gateActivation: 0 = tanh, 1 = relu, 2 = sigmoid
    int gateAct = INT_ARG(2);
    int cellAct = INT_ARG(3);
    int outAct = INT_ARG(4);

    std::vector<NDArray*> inputs = {x, Wx, Wr};
    if (b != nullptr) inputs.push_back(b);
    if (hI != nullptr) inputs.push_back(hI);
    if (cI != nullptr) inputs.push_back(cI);
    if (Wp != nullptr) inputs.push_back(Wp);

    std::vector<NDArray*> outputs = {h};
    if (hL != nullptr) outputs.push_back(hL);
    if (cL != nullptr) outputs.push_back(cL);

    auto status = executeMlirEx("lstm_layer", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(lstmLayer, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);

    Requirements req("MLIR LSTM_LAYER");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(x->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// lstmLayer_bp MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(lstmLayer_bp, ENGINE_CPU)

PLATFORM_IMPL(lstmLayer_bp, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* Wx = INPUT_VARIABLE(1);
    auto* Wr = INPUT_VARIABLE(2);
    auto* b = INPUT_VARIABLE(3);
    auto* hI = INPUT_VARIABLE(4);
    auto* cI = INPUT_VARIABLE(5);
    auto* gradH = INPUT_VARIABLE(6);
    auto* gradHL = block.width() > 7 ? INPUT_VARIABLE(7) : nullptr;
    auto* gradCL = block.width() > 8 ? INPUT_VARIABLE(8) : nullptr;

    auto* gradX = OUTPUT_VARIABLE(0);
    auto* gradWx = OUTPUT_VARIABLE(1);
    auto* gradWr = OUTPUT_VARIABLE(2);
    auto* gradB = OUTPUT_VARIABLE(3);
    auto* gradHI = block.outputWidth() > 4 ? OUTPUT_VARIABLE(4) : nullptr;
    auto* gradCI = block.outputWidth() > 5 ? OUTPUT_VARIABLE(5) : nullptr;

    std::vector<NDArray*> inputs = {x, Wx, Wr, b, hI, cI, gradH};
    if (gradHL != nullptr) inputs.push_back(gradHL);
    if (gradCL != nullptr) inputs.push_back(gradCL);

    std::vector<NDArray*> outputs = {gradX, gradWx, gradWr, gradB};
    if (gradHI != nullptr) outputs.push_back(gradHI);
    if (gradCI != nullptr) outputs.push_back(gradCI);

    auto status = executeMlir("lstm_layer_bp", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(lstmLayer_bp, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);

    Requirements req("MLIR LSTM_LAYER_BP");

    // Backward ops fall through to native C++ — no MLIR backward support
    req.expectTrue(false, "Backward ops use native C++ implementation");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// gruCell MLIR implementation
// Single GRU cell computation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(gruCell, ENGINE_CPU)

PLATFORM_IMPL(gruCell, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);     // [batch, input]
    auto* hPrev = INPUT_VARIABLE(1); // [batch, hidden]
    auto* Wru = INPUT_VARIABLE(2);   // [input, 2*hidden] - reset/update gates
    auto* Wc = INPUT_VARIABLE(3);    // [input, hidden] - candidate
    auto* Rru = INPUT_VARIABLE(4);   // [hidden, 2*hidden]
    auto* Rc = INPUT_VARIABLE(5);    // [hidden, hidden]
    auto* bru = block.width() > 6 ? INPUT_VARIABLE(6) : nullptr;
    auto* bc = block.width() > 7 ? INPUT_VARIABLE(7) : nullptr;

    auto* h = OUTPUT_VARIABLE(0);    // [batch, hidden]

    std::vector<NDArray*> inputs = {x, hPrev, Wru, Wc, Rru, Rc};
    if (bru != nullptr) inputs.push_back(bru);
    if (bc != nullptr) inputs.push_back(bc);

    std::vector<NDArray*> outputs = {h};

    auto status = executeMlirEx("gru_cell", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(gruCell, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);

    Requirements req("MLIR GRU_CELL");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(x->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// gru MLIR implementation (full sequence)
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(gru, ENGINE_CPU)

PLATFORM_IMPL(gru, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);     // [time, batch, input] or [batch, time, input]
    auto* hI = INPUT_VARIABLE(1);    // initial hidden [batch, hidden]
    auto* Wx = INPUT_VARIABLE(2);    // [input, 3*hidden]
    auto* Wh = INPUT_VARIABLE(3);    // [hidden, 3*hidden]
    auto* b = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;

    auto* h = OUTPUT_VARIABLE(0);    // all hidden states
    auto* hL = block.outputWidth() > 1 ? OUTPUT_VARIABLE(1) : nullptr;  // last hidden

    int dataFormat = block.numI() > 0 ? INT_ARG(0) : 0;

    std::vector<NDArray*> inputs = {x, hI, Wx, Wh};
    if (b != nullptr) inputs.push_back(b);

    std::vector<NDArray*> outputs = {h};
    if (hL != nullptr) outputs.push_back(hL);

    auto status = executeMlirEx("gru", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(gru, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);

    Requirements req("MLIR GRU");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(x->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// gru_bp MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(gru_bp, ENGINE_CPU)

PLATFORM_IMPL(gru_bp, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* hI = INPUT_VARIABLE(1);
    auto* Wx = INPUT_VARIABLE(2);
    auto* Wh = INPUT_VARIABLE(3);
    auto* b = INPUT_VARIABLE(4);
    auto* gradH = INPUT_VARIABLE(5);

    auto* gradX = OUTPUT_VARIABLE(0);
    auto* gradHI = OUTPUT_VARIABLE(1);
    auto* gradWx = OUTPUT_VARIABLE(2);
    auto* gradWh = OUTPUT_VARIABLE(3);
    auto* gradB = OUTPUT_VARIABLE(4);

    std::vector<NDArray*> inputs = {x, hI, Wx, Wh, b, gradH};
    std::vector<NDArray*> outputs = {gradX, gradHI, gradWx, gradWh, gradB};

    auto status = executeMlir("gru_bp", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(gru_bp, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);

    Requirements req("MLIR GRU_BP");

    // Backward ops fall through to native C++ — no MLIR backward support
    req.expectTrue(false, "Backward ops use native C++ implementation");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// simple_rnn MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(simple_rnn, ENGINE_CPU)

PLATFORM_IMPL(simple_rnn, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);     // [time, batch, input] or [batch, time, input]
    auto* hI = INPUT_VARIABLE(1);    // initial hidden [batch, hidden]
    auto* Wx = INPUT_VARIABLE(2);    // [input, hidden]
    auto* Wh = INPUT_VARIABLE(3);    // [hidden, hidden]
    auto* b = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;

    auto* h = OUTPUT_VARIABLE(0);    // all hidden states
    auto* hL = block.outputWidth() > 1 ? OUTPUT_VARIABLE(1) : nullptr;  // last hidden

    int dataFormat = block.numI() > 0 ? INT_ARG(0) : 0;

    std::vector<NDArray*> inputs = {x, hI, Wx, Wh};
    if (b != nullptr) inputs.push_back(b);

    std::vector<NDArray*> outputs = {h};
    if (hL != nullptr) outputs.push_back(hL);

    auto status = executeMlirEx("simple_rnn", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(simple_rnn, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);

    Requirements req("MLIR SIMPLE_RNN");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(x->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// lstmCell MLIR implementation
// Single LSTM cell computation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(lstmCell, ENGINE_CPU)

PLATFORM_IMPL(lstmCell, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);     // [batch, input]
    auto* hPrev = INPUT_VARIABLE(1); // [batch, hidden]
    auto* cPrev = INPUT_VARIABLE(2); // [batch, hidden]
    auto* Wx = INPUT_VARIABLE(3);    // [input, 4*hidden]
    auto* Wh = INPUT_VARIABLE(4);    // [hidden, 4*hidden]
    auto* Wc = block.width() > 5 ? INPUT_VARIABLE(5) : nullptr;  // peephole
    auto* b = block.width() > 6 ? INPUT_VARIABLE(6) : nullptr;

    auto* h = OUTPUT_VARIABLE(0);    // [batch, hidden]
    auto* c = OUTPUT_VARIABLE(1);    // [batch, hidden]

    std::vector<NDArray*> inputs = {x, hPrev, cPrev, Wx, Wh};
    if (Wc != nullptr) inputs.push_back(Wc);
    if (b != nullptr) inputs.push_back(b);

    std::vector<NDArray*> outputs = {h, c};

    auto status = executeMlirEx("lstm_cell", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(lstmCell, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);

    Requirements req("MLIR LSTM_CELL");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(x->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

} // namespace platforms
} // namespace ops
} // namespace sd

#endif // HAVE_MLIR
