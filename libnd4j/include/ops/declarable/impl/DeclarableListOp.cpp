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
#include <graph/Context.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <ops/declarable/DeclarableListOp.h>
#include <ops/declarable/OpDescriptor.h>

namespace sd {
namespace ops {
DeclarableListOp::DeclarableListOp(int numInputs, int numOutputs, const char* opName, int tArgs, int iArgs)
    : DeclarableOp(numInputs, numOutputs, opName, false, tArgs, iArgs) {
  // This kind of operations work with sets: NDArrayList
  this->getOpDescriptor()->setInputType(InputType_NUMERIC_SET);
}

/**
 * This method just outputs scalar buffer
 *
 * @tparam T
 * @param inputShape
 * @param block
 * @return
 */
ShapeList* DeclarableListOp::calculateOutputShape(ShapeList* inputShape, Context& block) {
  // TODO: ensure this method isn't ever called

  auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(block.dataType(), 'c', {1, 1});
  return SHAPELIST(newShape);
}

NDArray* DeclarableListOp::getZ(sd::graph::Context& block, int inputId) {
  return nullptr;
}

void DeclarableListOp::setupResult(NDArray* array, Context& block) {
  block.pushNDArrayToVariableSpace(block.getNodeId(), 0, array);
}

void DeclarableListOp::setupResultList(NDArrayList* arrayList, Context& block) {
  block.pushNDArrayListToVariableSpace(block.getNodeId(), 0, arrayList);
}

ResultSet DeclarableListOp::execute(NDArrayList* list, std::initializer_list<NDArray*> inputs,
                                    std::initializer_list<double> tArgs, std::initializer_list<int> iArgs) {
  std::vector<NDArray*> ins(inputs);
  std::vector<double> tas(tArgs);
  std::vector<int> ias(iArgs);
  return this->execute(list, ins, tas, ias);
}

Status DeclarableListOp::execute(Context* block) {
  if (block == nullptr) THROW_EXCEPTION("Block is NULL");

  sd_debug("Executing list op: [%s]\n", this->getOpName()->c_str());

  // ensure number of IArgs, TArgs match our expectations
  REQUIRE_OK(this->validateArguments(*block));

  // we shouldn't call for this in ListOp
  // this->prepareOutputs(*block);

  auto timeStart = std::chrono::system_clock::now();

  Status status = this->validateAndExecute(*block);

  auto timeEnd = std::chrono::system_clock::now();
  auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart).count();
  block->setInnerTime(outerTime);
  traceExecIfNeeded(*block);

  return status;
}

ResultSet DeclarableListOp::execute(NDArrayList* list, std::vector<NDArray*>& inputs, std::vector<double>& tArgs,
                                    std::vector<int>& iArgs) {
  VariableSpace varSpace;
  int nodeId = 119;

  // should be never used in practice, since in-graph NDArrayList should have id set
  int cnt = -1;
  std::vector<int> in;
  if (list != nullptr) {
    if (list->id().first == 0) list->id().first = -1;

    auto listVar = new Variable(nullptr, nullptr, -119, 0);
    listVar->setNDArrayList(list);
    varSpace.putVariable(-1, listVar);
    in.push_back(-1);
    cnt--;
  }

  for (auto v : inputs) {
    auto var = new Variable(v);
    var->markRemovable(false);
    in.push_back(cnt);
    varSpace.putVariable(cnt--, var);
  }

  Context block(1, &varSpace, false);
  block.fillInputs(in);

  for (size_t e = 0; e < tArgs.size(); e++) block.getTArguments()->emplace_back(tArgs.at(e));

  for (size_t e = 0; e < iArgs.size(); e++) block.getIArguments()->emplace_back(iArgs.at(e));

  Status result = this->validateAndExecute(block);
  ResultSet res;
  res.setStatus(result);

  for (int e = 0; e < DataTypeUtils::max<int>(); e++) {
    std::pair<int, int> pair(1, e);
    if (varSpace.hasVariable(pair)) {
      auto var = varSpace.getVariable(pair);
      if (var->hasNDArray()) {
        auto arr = var->getNDArray();
        if (arr->isAttached()) {
          auto d = arr->detach();
          res.push_back(d);
        } else {
          var->markRemovable(false);
          res.push_back(arr);
        }
      }
    } else
      break;
  }

  return res;
}
}  // namespace ops
}  // namespace sd
