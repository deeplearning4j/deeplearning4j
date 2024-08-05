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
//  @author raver119@gmail.com
//

#ifndef LIBND4J_DECLARABLE_LIST_OP_H
#define LIBND4J_DECLARABLE_LIST_OP_H
#include <array/ResultSet.h>
#include <graph/Context.h>
#include <ops/declarable/DeclarableOp.h>
#include <ops/declarable/OpRegistrator.h>

using namespace sd::graph;

namespace sd {
namespace ops {
class SD_LIB_EXPORT DeclarableListOp : public DeclarableOp {
 protected:
  Status validateAndExecute(sd::graph::Context& block) override = 0;

  NDArray* getZ(sd::graph::Context& block, int inputId);
  void setupResult(NDArray* array, Context& block);
  void setupResultList(NDArrayList* arrayList, Context& block);

 public:
  DeclarableListOp(int numInputs, int numOutputs, const char* opName, int tArgs, int iArgs);

  Status execute(Context* block) override;

  ResultSet execute(NDArrayList* list, std::initializer_list<NDArray*> inputs, std::initializer_list<double> tArgs,
                    std::initializer_list<int> iArgs);
  ResultSet execute(NDArrayList* list, std::vector<NDArray*>& inputs, std::vector<double>& tArgs,
                    std::vector<int>& iArgs);

  ShapeList* calculateOutputShape(ShapeList* inputShape, Context& block) override;
};
}  // namespace ops
}  // namespace sd

#endif
