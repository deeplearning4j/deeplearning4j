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
// Created by raver119 on 13.10.2017.
//
#include "ops/declarable/BooleanOp.h"

#include <array/NDArrayFactory.h>
#include <ops/declarable/OpRegistrator.h>
#include <initializer_list>
#include <vector>

namespace sd {
namespace ops {
BooleanOp::BooleanOp(const char *name, int numInputs, bool scalar)
    : DeclarableOp(name, numInputs, scalar) {
  //
}

/**
 * Output shape of any BooleanOp is ALWAYS scalar
 */
ShapeList *BooleanOp::calculateOutputShape(ShapeList *inputShape, Context &block) {
  return SHAPELIST(ConstantShapeHelper::getInstance().scalarShapeInfo(DataType::BOOL));
}

bool BooleanOp::verify(Context &block) {
  // check if scalar or not

  // validation?

  Status status = this->validateNonEmptyInput(block);
  if (status != Status::OK) {
    THROW_EXCEPTION("Bad inputs");
  }

  status = this->validateAndExecute(block);
  if (status == Status::EQ_TRUE)
    return true;
  else if (status == Status::EQ_FALSE)
    return false;
  else {
    sd_printf("Got error %i during [%s] evaluation: ", (int)status, this->getOpDescriptor()->getOpName()->c_str());
    THROW_EXCEPTION("Internal error");
  }

  return false;
}

bool BooleanOp::prepareOutputs(Context &ctx) {
  auto variableSpace = ctx.getVariableSpace();
  if (ctx.isFastPath()) return true;

  for (int e = 0; e < this->getOpDescriptor()->getNumberOfOutputs(); e++) {
    std::pair<int, int> pair(ctx.nodeId(), e);

    if (!variableSpace->hasVariable(pair)) variableSpace->putVariable(pair, new Variable());

    auto var = ctx.variable(pair);

    if (!var->hasNDArray()) {
      var->setNDArray(NDArrayFactory::create_<bool>(false, ctx.launchContext()));
      var->markRemovable(true);
    }
  }

  return true;
}

Status BooleanOp::execute(Context *block) {
  // basic validation: ensure inputs are set
  REQUIRE_OK(this->validateNonEmptyInput(*block));

  // ensure number of IArgs, TArgs match our expectations
  REQUIRE_OK(this->validateArguments(*block));

  // this method will allocate output NDArrays for this op
  this->prepareOutputs(*block);

  auto timeStart = std::chrono::system_clock::now();

  Status status = this->validateAndExecute(*block);

  auto timeEnd = std::chrono::system_clock::now();
  auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart).count();
  block->setInnerTime(outerTime);

  // basically we're should be putting 0.0 as FALSE, and any non-0.0 value will be treated as TRUE
  std::pair<int, int> p(block->nodeId(), 0);
  auto var = block->isFastPath() ? block->fastpath_out()[0] : block->variable(p)->getNDArray();
  if(!var->isEmpty())
    var->p(LongType(0), status == Status::EQ_TRUE ? 1.0f : 0.0f);

  // for CPU backend that's nop, but for CUDA-like archs this will update special buffer
  var->syncToDevice();

  if (status == Status::EQ_FALSE || status == Status::EQ_TRUE) return Status::OK;

  sd_printf("%s: node_%i got unexpected result instead of boolean: [%i]\n", this->getOpName()->c_str(), block->nodeId(),
            status);


  traceExecIfNeeded(*block);


  return Status::KERNEL_FAILURE;
}

bool BooleanOp::verify(const std::vector<NDArray *> &args) {
  VariableSpace variableSpace;

  int cnt = -1;
  std::vector<int> in;
  for (auto v : args) {
    auto var = new Variable(v);
    var->markRemovable(false);
    in.push_back(cnt);
    variableSpace.putVariable(cnt--, var);
  }

  Context block(1, &variableSpace, false);
  block.fillInputs(in);

  return this->verify(block);
}
}  // namespace ops
}  // namespace sd
