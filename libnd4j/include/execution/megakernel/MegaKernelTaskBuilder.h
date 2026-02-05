/* ******************************************************************************
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
// Host-side builder for mega-kernel tasks.
//

#ifndef LIBND4J_MEGAKERNEL_TASK_BUILDER_H
#define LIBND4J_MEGAKERNEL_TASK_BUILDER_H

#include <execution/megakernel/MegaKernelTask.h>
#include <array/NDArray.h>
#include <graph/Graph.h>
#include <vector>
#include <unordered_map>
#include <memory>

namespace sd {
namespace execution {

/**
 * Builder for constructing mega-kernel tasks from a subgraph.
 *
 * This class takes a set of operations from a computation graph and
 * converts them into a compact task representation suitable for
 * device-side interpretation.
 */
class MegaKernelTaskBuilder {
public:
    MegaKernelTaskBuilder();
    ~MegaKernelTaskBuilder();

    /**
     * Begin building a new task.
     * @param taskId Unique identifier for this task
     * @param graphId The graph this task belongs to
     */
    void beginTask(int32_t taskId, int32_t graphId);

    /**
     * Add an operation to the current task.
     * @param opType The operation type
     * @param inputs Input NDArrays
     * @param outputs Output NDArrays
     * @param params Operation-specific parameters (can be nullptr)
     * @param paramSize Size of parameter data in bytes
     */
    void addOp(MegaKernelOpType opType,
               const std::vector<NDArray*>& inputs,
               const std::vector<NDArray*>& outputs,
               const void* params = nullptr,
               size_t paramSize = 0);

    /**
     * Add a barrier wait to the current task.
     * @param barrierId The barrier to wait on
     */
    void addBarrierWait(int32_t barrierId);

    /**
     * Add a barrier signal to the current task.
     * @param barrierId The barrier to signal after completion
     */
    void addBarrierSignal(int32_t barrierId);

    /**
     * Configure the CUDA grid/block dimensions for this task.
     * If not called, dimensions will be auto-computed based on workload.
     */
    void setGridConfig(int32_t gridX, int32_t gridY, int32_t gridZ,
                       int32_t blockX, int32_t blockY, int32_t blockZ,
                       int32_t sharedMem);

    /**
     * Finalize the current task and return it.
     * The returned task is owned by the builder and will be released
     * when the builder is destroyed or when finalizeGraph() is called.
     */
    MegaKernelTask* endTask();

    /**
     * Finalize the entire task graph and prepare for execution.
     * This allocates device memory and transfers all task data.
     * @return A complete task graph ready for execution
     */
    std::unique_ptr<MegaKernelTaskGraph> finalizeGraph();

    /**
     * Convert an operation name to the corresponding MegaKernelOpType.
     * Returns CUSTOM if the operation is not recognized.
     */
    static MegaKernelOpType opNameToType(const std::string& opName);

    /**
     * Check if an operation can be fused into a mega-kernel.
     */
    static bool canFuseOp(const std::string& opName);

private:
    struct BuilderState;
    std::unique_ptr<BuilderState> state_;

    // Auto-compute grid dimensions based on workload
    void autoComputeGridConfig(MegaKernelTask* task);

    // Register a buffer and return its index
    int32_t registerBuffer(NDArray* arr);

    // Add parameters to the parameter buffer
    int32_t addParams(const void* params, size_t size);
};

/**
 * Utility class for analyzing a graph and extracting fusible subgraphs.
 */
class MegaKernelGraphAnalyzer {
public:
    /**
     * Analyze a graph and identify convex subgraphs suitable for mega-kernel fusion.
     *
     * @param graph The graph to analyze
     * @return List of node groups, each representing a fusible subgraph
     */
    static std::vector<std::vector<int>> findFusibleSubgraphs(graph::Graph* graph);

    /**
     * Check if a subgraph is convex (no external dependencies within the subgraph).
     */
    static bool isConvexSubgraph(graph::Graph* graph, const std::vector<int>& nodes);

    /**
     * Estimate the benefit of fusing a subgraph.
     * Returns a score where higher is better.
     */
    static float estimateFusionBenefit(graph::Graph* graph, const std::vector<int>& nodes);
};

}  // namespace execution
}  // namespace sd

#endif  // LIBND4J_MEGAKERNEL_TASK_BUILDER_H
