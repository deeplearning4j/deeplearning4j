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
// Mega-kernel task scheduler with dependency tracking.
//

#ifndef LIBND4J_MEGAKERNEL_SCHEDULER_H
#define LIBND4J_MEGAKERNEL_SCHEDULER_H

#include <execution/megakernel/MegaKernelTask.h>
#include <execution/LaunchContext.h>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>

namespace sd {
namespace execution {

/**
 * Scheduler for mega-kernel task execution.
 *
 * The scheduler manages:
 * - Task dependency tracking
 * - Ready queue management
 * - Persistent kernel dispatch
 * - Task completion callbacks
 */
class MegaKernelScheduler {
public:
    explicit MegaKernelScheduler(LaunchContext* context);
    ~MegaKernelScheduler();

    /**
     * Submit a task graph for execution.
     * @param graph The task graph to execute
     * @param async If true, returns immediately; otherwise blocks until complete
     */
    void submitGraph(MegaKernelTaskGraph* graph, bool async = false);

    /**
     * Wait for all submitted tasks to complete.
     */
    void synchronize();

    /**
     * Check if all tasks have completed.
     */
    bool isComplete() const;

    /**
     * Get the number of tasks currently in the ready queue.
     */
    int32_t getReadyQueueSize() const;

    /**
     * Get execution statistics.
     */
    struct Statistics {
        int64_t tasksSubmitted;
        int64_t tasksCompleted;
        int64_t totalExecutionTimeNs;
        int64_t averageTaskLatencyNs;
        int64_t maxTaskLatencyNs;
        int64_t kernelLaunchCount;
    };

    Statistics getStatistics() const;

    /**
     * Reset statistics counters.
     */
    void resetStatistics();

    /**
     * Enable or disable the persistent kernel mode.
     * When disabled, each task launches a separate kernel.
     */
    void setPersistentKernelMode(bool enabled);

    /**
     * Set the maximum number of concurrent tasks.
     * Only relevant when persistent kernel mode is disabled.
     */
    void setMaxConcurrentTasks(int32_t maxTasks);

private:
    struct SchedulerState;
    std::unique_ptr<SchedulerState> state_;

    LaunchContext* context_;
    bool persistentMode_;
    int32_t maxConcurrentTasks_;

    // Statistics
    mutable std::mutex statsMutex_;
    Statistics stats_;

    // Internal methods
    void processReadyTasks();
    void launchPersistentKernel();
    void launchIndividualKernels();
    void updateDependencies(int32_t completedTaskId);
    void markTaskReady(int32_t taskId);
};

/**
 * Device-side task executor for persistent kernel mode.
 * This is launched once and continuously processes tasks from the queue.
 */
class PersistentKernelExecutor {
public:
    /**
     * Launch the persistent kernel.
     * The kernel runs until signaled to stop.
     */
    static void launch(DeviceTaskQueue* queue, LaunchContext* context);

    /**
     * Signal the persistent kernel to stop.
     */
    static void stop(DeviceTaskQueue* queue);

    /**
     * Check if the persistent kernel is running.
     */
    static bool isRunning(DeviceTaskQueue* queue);

private:
    // CUDA kernel implementation is in the .cu file
};

}  // namespace execution
}  // namespace sd

#endif  // LIBND4J_MEGAKERNEL_SCHEDULER_H
