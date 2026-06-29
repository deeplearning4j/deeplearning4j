/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Paul Dubs
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_standardize)

#include <ops/declarable/headers/transforms.h>
#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/reverse.h>
#include <helpers/ShapeUtils.h>

namespace sd {
namespace ops {

CONFIGURABLE_OP_IMPL(standardize, 1, 1, true, 0, -2) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // Handle empty input gracefully - nothing to standardize
  if (input->isEmpty() || input->lengthOf() == 0) {
    return sd::Status::OK;
  }

  output->nullify();
  std::vector<sd::LongType> axis;

  if (block.width() > 1)
    axis = INPUT_VARIABLE(1)->template asVectorT<sd::LongType>();
  else if (block.numI() > 0)
    axis = *block.getIArguments();

  REQUIRE_TRUE(!axis.empty(), 0, "STANDARDIZE OP: axis has to be non-empty")

  shape::checkDimensions(input->rankOf(), &axis);

  // Pre-compute shape for mean/stdev with keepDims=true to avoid heap allocations
  // This creates the shape [1, ..., 1, dim, 1, ..., 1] where reduced dims are 1
  auto reducedShapeInfo = ShapeUtils::evalReduceShapeInfo('c', &axis, input->shapeInfo(), true, false, block.getWorkspace());

  // Create mean and stdev arrays on stack with pre-computed shape
  NDArray means(reducedShapeInfo, true, block.launchContext());
  NDArray stdev(reducedShapeInfo, true, block.launchContext());

  // Compute mean directly into pre-allocated array
  input->reduceAlongDimension(reduce::Mean, &means, &axis, true, false);

  // Compute standard deviation directly into stdev array
  // Using SummaryStatsStandardDeviation to get std = sqrt(variance)
  // For plain standardization, we do NOT add epsilon (unlike layer_norm)
  input->varianceAlongDimension(variance::SummaryStatsStandardDeviation, stdev, false, &axis);

  // When stdev == 0 (all values in the group are identical), replace 0 with 1
  // so the result is (x - mean) / 1 = 0 (since x == mean when stdev == 0).
  // CompareAndSetTransform params: [compare_value, set_value, eps, mode]
  // mode=0 means "equals": if |elem - compare| <= eps, replace with set_value
  ExtraArguments stdevArgs({0.0, 1.0, 1e-6, 0.0});
  stdev.applyTransform(transform::SameOps::CompareAndSetTransform, &stdev, &stdevArgs);

  // output = (input - mean) / stdev
  input->applyTrueBroadcast(sd::BroadcastOpsTuple::Subtract(), &means, output, false);
  output->applyTrueBroadcast(sd::BroadcastOpsTuple::Divide(), &stdev, output, false);

  // Flush the default context stream before the stack NDArrays (means, stdev) are freed.
  // During DSP REPLAY this op may run as a gap op: tl_graphExecutionActive=true but
  // tl_graphCaptureStream=nullptr. CudaMemoryPool::free() calls cudaFreeAsync on
  // tl_dspExecutionStream, which is a different stream than the default context stream
  // where the CUDA kernels above are enqueued. Without this sync, the frees can race
  // with the kernel reads. synchronizeExecStream() skips only during CAPTURE (where
  // cudaStreamSynchronize is illegal).
  output->synchronizeExecStream("standardize: sync before stack NDArray destruction");

  return sd::Status::OK;
}

DECLARE_TYPES(standardize) {
  getOpDescriptor()->setAllowedInputTypes(0, {ALL_FLOATS});
  getOpDescriptor()->setAllowedInputTypes(1, {DataType::INT32, DataType::INT64});
  getOpDescriptor()->setAllowedOutputTypes(0, DataType::INHERIT);
}

CUSTOM_OP_IMPL(standardize_bp, 2, 1, false, 0, -2) {
  auto input = INPUT_VARIABLE(0);
  auto eps = block.width() == 3 ? INPUT_VARIABLE(2) : INPUT_VARIABLE(1);

  auto output = OUTPUT_VARIABLE(0);

  // Handle empty input gracefully - nothing to backpropagate
  if (input->isEmpty() || input->lengthOf() == 0) {
    return sd::Status::OK;
  }

  std::vector<sd::LongType> axis;

  if (block.width() == 3)
    axis = INPUT_VARIABLE(1)->template asVectorT<sd::LongType>();
  else if (block.numI() > 0)
    axis = *block.getIArguments();

  REQUIRE_TRUE(!axis.empty(), 0, "STANDARDIZE OP: axis has to be non-empty")

  shape::checkDimensions(input->rankOf(), &axis);
  auto longAxis = ArrayUtils::toLongVector(axis);

  auto means = input->reduceAlongDimension(reduce::Mean, &axis, true);
  REQUIRE_TRUE(means != nullptr, 0, "STANDARDIZE_BP OP: failed to compute mean along dimension");

  auto stdevRaw = input->varianceAlongDimension(variance::SummaryStatsStandardDeviation, false, &axis);
  REQUIRE_TRUE(stdevRaw != nullptr, 0, "STANDARDIZE_BP OP: failed to compute standard deviation along dimension");

  // Reshape stdev to match means shape for broadcasting (use reshape instead of reshapei)
  auto meansShape = means->getShapeAsVector();
  auto stdev = stdevRaw->reshape(stdevRaw->ordering(), *meansShape);
  delete meansShape;
  delete stdevRaw;

  // When stdev == 0, replace with 1 to match the forward pass zero-stdev handling.
  // This prevents Inf in the backward pass when dividing by zero stdev.
  ExtraArguments zeroStdevBpArgs({0.0, 1.0, 1e-6, 0.0});
  stdev->applyTransform(transform::SameOps::CompareAndSetTransform, stdev, &zeroStdevBpArgs);

  eps->applyTrueBroadcast(sd::BroadcastOpsTuple::Divide(), stdev, output, false);

  auto sum = output->reduceAlongDimension(reduce::Sum, &axis, true);
  REQUIRE_TRUE(sum != nullptr, 0, "STANDARDIZE_BP OP: failed to compute sum along dimension");

  NDArray dldu_sum = -(*sum);

  NDArray dldx_u(input->shapeInfo(), false, block.launchContext());
  std::vector<NDArray *> meanBpArgs = {input, &dldu_sum};
  std::vector<NDArray *> meanBpOutput = {&dldx_u};
  std::vector<double> meanBpTArgs = {};
  std::vector<bool> meanBpBArgs = {};

  sd::ops::reduce_mean_bp meanBp;
  meanBp.execute(meanBpArgs, meanBpOutput, meanBpTArgs, longAxis, meanBpBArgs);
  *output += dldx_u;

  // (eps * (means - input) / (stdev * stdev))
  NDArray tmp(eps->shapeInfo(), false, block.launchContext());
  means->applyTrueBroadcast(sd::BroadcastOpsTuple::Subtract(), input, &tmp, false);
  tmp.applyPairwiseTransform(sd::pairwise::Multiply, eps, &tmp);
  stdev->applyPairwiseTransform(sd::pairwise::Multiply, stdev, stdev);
  tmp.applyTrueBroadcast(sd::BroadcastOpsTuple::Divide(), stdev, &tmp, false);

  auto dlds_sum = tmp.reduceAlongDimension(reduce::Sum, &axis, true);
  REQUIRE_TRUE(dlds_sum != nullptr, 0, "STANDARDIZE_BP OP: failed to compute dlds_sum along dimension");

  NDArray dldx_s(input->shapeInfo(), false, block.launchContext());
  std::vector<NDArray *> stdevBpArgs = {input, dlds_sum};
  std::vector<NDArray *> stdevBpOutput = {&dldx_s};
  std::vector<double> stdevBpTArgs = {};
  std::vector<bool> stdevBpBArgs = {};
  sd::ops::reduce_stdev_bp stdevBp;
  stdevBp.execute(stdevBpArgs, stdevBpOutput, stdevBpTArgs, longAxis, stdevBpBArgs);
  *output += dldx_s;

  output->applyScalar(sd::scalar::ReplaceNans, 0, output);

  // Flush the default context stream BEFORE freeing any NDArrays.
  // During DSP REPLAY this op runs as a gap op: tl_graphExecutionActive=true,
  // tl_graphCaptureStream=nullptr. CudaMemoryPool::free() calls cudaFreeAsync on
  // tl_dspExecutionStream — a different stream than the default context stream where
  // the CUDA kernels above were enqueued. Without this sync, cudaFreeAsync can reclaim
  // the device memory of sum/means/stdev/dlds_sum while the default stream kernels
  // are still reading it. synchronizeExecStream() skips only during CAPTURE.
  output->synchronizeExecStream("standardize_bp: sync before NDArray destruction");

  delete sum;
  delete means;
  delete stdev;
  delete dlds_sum;

  return sd::Status::OK;
}

DECLARE_TYPES(standardize_bp) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(standardize_bp) {
  auto in = inputShape->at(0);
  sd::LongType *out;
  COPY_SHAPE(in, out);
  auto result = CONSTANT(out);
  delete[] out;

  return SHAPELIST(result);
}

}  // namespace ops
}  // namespace sd

#endif
