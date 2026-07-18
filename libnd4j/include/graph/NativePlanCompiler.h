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

#ifndef LIBND4J_NATIVE_PLAN_COMPILER_H
#define LIBND4J_NATIVE_PLAN_COMPILER_H

#include <graph/NativeDynamicShapePlan.h>
#include <graph/generated/graph_generated.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace sd {
namespace graph {

/**
 * Compiles a FlatGraph into a NativeDynamicShapePlan for native C++ execution.
 *
 * This mirrors the Java DynamicShapePlanCompiler but operates on C++ FlatBuffer
 * types directly. Steps:
 *
 * 1. Topological sort of FlatNode entries (using inputPaired dependencies)
 * 2. Filter out VARIABLE_INIT/PLACEHOLDER_SET nodes
 * 3. Build external input index maps (constants, variables, placeholders → negative indices)
 * 4. Assign sequential output slot indices
 * 5. Wire inputs per slot using inputSourceIndices/inputSourceTypes
 * 6. Compute liveness analysis → releaseAtStep schedule
 * 7. Resolve op hashes via OpRegistrator::getInstance().getOperation()
 * 8. Freeze i/t/b/d args from FlatNode fields
 * 9. Return NativeDynamicShapePlan ready for execution
 */
class SD_LIB_EXPORT NativePlanCompiler {
 public:
  /**
   * Compile a NativeDynamicShapePlan from a FlatGraph and pre-loaded variables.
   *
   * @param graph  FlatBuffer graph with nodes and variables
   * @param variables  Pre-loaded variable data (constants and initialized variables)
   * @param requestedOutputs  Output variable names the plan should produce
   * @return Compiled plan ready for execution, or nullptr on failure
   */
  static NativeDynamicShapePlan* compile(
      const ::graph::FlatGraph* graph,
      const std::unordered_map<std::string, NDArray*>& variables,
      const std::vector<std::string>& requestedOutputs);

 private:
  // Op classification is driven directly by each registered op's OpDescriptor traits.
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_NATIVE_PLAN_COMPILER_H
