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
// Scaled GEMM over operands of any storage type (FP8 included), composed from
// backend-dispatched array operations.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_fp8_matmul) || NOT_EXCLUDED(OP_smooth_quant) || NOT_EXCLUDED(OP_awq_matmul) || \
    NOT_EXCLUDED(OP_decoder_masked_mha)
#include <array/DataTypeUtils.h>
#include <helpers/MmulHelper.h>
#include <ops/declarable/helpers/int8_gemm.h>
#include <ops/op_types.h>
#if NOT_EXCLUDED(OP_smooth_quant)
#include <ops/declarable/helpers/symmetric_quant.h>
#endif

#include <vector>

namespace sd {
namespace ops {
namespace helpers {

// Returns operand itself when it already holds dataType in shape; otherwise a
// fresh c-order copy of dataType in shape, which holds operand's element count.
// The copy is staged in operand's own shape, where the central NDArray::assign
// converts (an FP8 operand widens exactly) and follows operand's strides, and
// the contiguous copy then relabels into shape in c index order.
static NDArray* scaledGemmOperand(NDArray* operand, const std::vector<LongType>& shape, DataType dataType,
                                  LaunchContext* context) {
  if (operand->dataType() == dataType && operand->isSameShape(shape)) return operand;
  std::vector<LongType> operandShape(operand->shapeOf(), operand->shapeOf() + operand->rankOf());
  auto* copy = new NDArray('c', operandShape, dataType, context);
  copy->assign(operand);
  if (!copy->isSameShape(shape)) copy->reshapei('c', shape);
  return copy;
}

// Retires a copy made by scaledGemmOperand behind its last consumer on the stream.
static void scaledGemmRetire(NDArray* operand, NDArray* original) {
  if (operand != original) MmulHelper::deleteTemporary(operand);
}

// The product accumulates in the aggregate type of the output: operands convert
// into it (FP8 exactly), products sum and scale at that precision, and the
// result converts once into the output. An output of the aggregate type in the
// product's layout receives the product directly.
template <typename Z>
static void scaledGemm_(LaunchContext* context, NDArray* A, NDArray* B, NDArray* scaleA, NDArray* scaleB,
                        NDArray* bias, NDArray* output, bool transposeA, bool transposeB) {
  using AccT = typename simdOps::AggregateType<Z>::type;
  const DataType accType = DataTypeUtils::fromT<AccT>();
  const LongType columns = transposeB ? B->sizeAt(0) : B->sizeAt(1);
  const LongType depth = transposeB ? B->sizeAt(1) : B->sizeAt(0);
  const LongType rows = output->lengthOf() / columns;
  std::vector<LongType> productShape = {rows, columns};

  NDArray* product = output->dataType() == accType && output->isSameShape(productShape)
                         ? output
                         : new NDArray('c', productShape, accType, context);
  if (depth == 0) {
    // An empty contraction sums nothing.
    product->nullify();
  } else {
    // A's leading axes flatten into the rows of the product.
    const std::vector<LongType> aShape =
        transposeA ? std::vector<LongType>{depth, rows} : std::vector<LongType>{rows, depth};
    const std::vector<LongType> bShape = {B->sizeAt(0), B->sizeAt(1)};
    NDArray* a = scaledGemmOperand(A, aShape, accType, context);
    NDArray* b = scaledGemmOperand(B, bShape, accType, context);
    MmulHelper::matmul(a, b, product, transposeA, transposeB, 1.0, 0.0);
    scaledGemmRetire(a, A);
    scaledGemmRetire(b, B);
  }

  // scaleA runs down the rows of the product, scaleB and bias along its columns.
  if (scaleA != nullptr) {
    NDArray* rowScale = scaledGemmOperand(scaleA, {scaleA->lengthOf(), 1}, accType, context);
    *product *= *rowScale;
    scaledGemmRetire(rowScale, scaleA);
  }
  if (scaleB != nullptr) {
    NDArray* columnScale = scaledGemmOperand(scaleB, {1, scaleB->lengthOf()}, accType, context);
    *product *= *columnScale;
    scaledGemmRetire(columnScale, scaleB);
  }
  if (bias != nullptr) {
    NDArray* biasRow = scaledGemmOperand(bias, {1, columns}, accType, context);
    *product += *biasRow;
    scaledGemmRetire(biasRow, bias);
  }
  if (product != output) {
    output->assign(product);
    MmulHelper::deleteTemporary(product);
  }
}

void scaledGemm(LaunchContext* context, NDArray* A, NDArray* B, NDArray* scaleA, NDArray* scaleB, NDArray* bias,
                NDArray* output, bool transposeA, bool transposeB) {
  if (output->isEmpty()) return;
  BUILD_SINGLE_SELECTOR(output->dataType(), scaledGemm_,
                        (context, A, B, scaleA, scaleB, bias, output, transposeA, transposeB), SD_FLOAT_TYPES);
}

#if NOT_EXCLUDED(OP_smooth_quant)
// Each channel of the last axis divides by its smoothing scale in the aggregate
// type of the output, and the quotient converts once into the output.
template <typename Z>
static void smoothActivation_(LaunchContext* context, NDArray* input, NDArray* smoothScale, NDArray* output) {
  using AccT = typename simdOps::AggregateType<Z>::type;
  const DataType accType = DataTypeUtils::fromT<AccT>();
  const LongType depth = input->sizeAt(-1);
  std::vector<LongType> rowsShape = {input->lengthOf() / depth, depth};
  const std::vector<LongType> divisorShape = {1, depth};

  NDArray* smoothed = output->dataType() == accType && output->isSameShape(rowsShape)
                          ? output
                          : new NDArray('c', rowsShape, accType, context);
  NDArray* rows = scaledGemmOperand(input, rowsShape, accType, context);
  if (rows != smoothed) smoothed->assign(rows);
  scaledGemmRetire(rows, input);
  NDArray* divisor = scaledGemmOperand(smoothScale, divisorShape, accType, context);
  *smoothed /= *divisor;
  scaledGemmRetire(divisor, smoothScale);
  if (smoothed != output) {
    output->assign(smoothed);
    MmulHelper::deleteTemporary(smoothed);
  }
}

// SmoothQuant: the activation divides by the smoothing scale s, quantizes by the
// activation scale a onto the weight's grid (A8 with W8, INT8 or FP8) and
// dequantizes by a; the weight codes dequantize by the weight scale in the
// scaled GEMM. q(x / s / a) = q(x / c) with the channel scale c = s * a.
template <typename Z>
static void smoothQuantGemm_(LaunchContext* context, NDArray* input, NDArray* weight, NDArray* smoothScale,
                             NDArray* actScale, NDArray* weightScale, NDArray* bias, NDArray* output,
                             bool transposeWeight) {
  using AccT = typename simdOps::AggregateType<Z>::type;
  const DataType accType = DataTypeUtils::fromT<AccT>();
  const LongType depth = input->sizeAt(-1);
  if (depth == 0) {
    // An empty contraction sums nothing, so no activation quantizes.
    scaledGemm_<Z>(context, input, weight, nullptr, weightScale, bias, output, false, !transposeWeight);
    return;
  }
  const LongType columns = output->sizeAt(-1);
  const LongType rows = output->lengthOf() / columns;
  const LongType actLength = actScale->lengthOf();
  std::vector<LongType> channelShape = {depth};
  const std::vector<LongType> actShape = {actLength};
  const std::vector<LongType> actRowShape = {1, actLength};
  std::vector<LongType> rowsShape = {rows, depth};
  std::vector<LongType> inputShape(input->shapeOf(), input->shapeOf() + input->rankOf());

  NDArray* smooth = scaledGemmOperand(smoothScale, channelShape, accType, context);
  auto* channelScale = new NDArray('c', channelShape, accType, context);
  channelScale->assign(smooth);
  scaledGemmRetire(smooth, smoothScale);
  NDArray* act = scaledGemmOperand(actScale, actShape, accType, context);
  *channelScale *= *act;
  scaledGemmRetire(act, actScale);

  auto* codes = new NDArray('c', inputShape, weight->dataType(), context);
  symmetricQuantize(context, input, channelScale, codes);
  MmulHelper::deleteTemporary(channelScale);

  codes->reshapei('c', rowsShape);
  auto* activation = new NDArray('c', rowsShape, accType, context);
  activation->assign(codes);
  MmulHelper::deleteTemporary(codes);
  NDArray* actRow = scaledGemmOperand(actScale, actRowShape, accType, context);
  *activation *= *actRow;
  scaledGemmRetire(actRow, actScale);

  scaledGemm_<Z>(context, activation, weight, nullptr, weightScale, bias, output, false, !transposeWeight);
  MmulHelper::deleteTemporary(activation);
}

void smoothActivation(LaunchContext* context, NDArray* input, NDArray* smoothScale, NDArray* output) {
  if (output->isEmpty()) return;
  BUILD_SINGLE_SELECTOR(output->dataType(), smoothActivation_, (context, input, smoothScale, output),
                        SD_FLOAT_TYPES);
}

void smoothQuantGemm(LaunchContext* context, NDArray* input, NDArray* weight, NDArray* smoothScale,
                     NDArray* actScale, NDArray* weightScale, NDArray* bias, NDArray* output, bool transposeWeight) {
  if (output->isEmpty()) return;
  BUILD_SINGLE_SELECTOR(output->dataType(), smoothQuantGemm_,
                        (context, input, weight, smoothScale, actScale, weightScale, bias, output, transposeWeight),
                        SD_FLOAT_TYPES);
}
#endif

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
