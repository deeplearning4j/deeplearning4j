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
// Pipeline Scheduler - Coordinate pipeline execution across devices
//

#ifndef SD_PIPELINE_SCHEDULER_H
#define SD_PIPELINE_SCHEDULER_H

#include <system/common.h>
#include <array/NDArray.h>
#include <execution/ModelParallelConfig.h>
#include <execution/DeviceManager.h>
#include <execution/DataTransferManager.h>
#include <graph/Context.h>

#include <vector>
#include <memory>
#include <mutex>
#include <queue>
#include <atomic>
#include <future>
#include <functional>
#include <condition_variable>
#include <unordered_map>

namespace sd {
namespace modelparallel {

/**
 * Pipeline schedule type
 */
enum class PipelineSchedule {
    GPIPE = 0,             // GPipe: Fill-drain schedule with gradient accumulation
    ONE_F_ONE_B,           // 1F1B: Interleaved forward-backward
    INTERLEAVED_1F1B,      // Interleaved 1F1B with virtual pipeline stages
    ZERO_BUBBLE,           // Zero-bubble pipeline (minimal idle time)
    ASYNC                  // Fully asynchronous (may have staleness)
};

/**
 * Stage execution state
 */
enum class StageState {
    IDLE = 0,
    WAITING_INPUT,
    EXECUTING_FORWARD,
    EXECUTING_BACKWARD,
    TRANSFERRING,
    COMPLETED
};

/**
 * Micro-batch descriptor
 */
struct SD_LIB_EXPORT MicroBatch {
    int microBatchId = 0;
    int globalBatchId = 0;

    // Input/output tensors
    std::vector<NDArray*> inputs;
    std::vector<NDArray*> outputs;
    std::vector<NDArray*> activations;  // For backward pass

    // Gradient tensors (for backward pass)
    std::vector<NDArray*> gradInputs;
    std::vector<NDArray*> gradOutputs;

    // Tracking
    bool forwardComplete = false;
    bool backwardComplete = false;
    int currentStage = 0;
    double forwardStartTime = 0.0;
    double forwardEndTime = 0.0;
    double backwardStartTime = 0.0;
    double backwardEndTime = 0.0;
};

/**
 * Pipeline stage descriptor
 */
struct SD_LIB_EXPORT PipelineStage {
    int stageId = 0;
    int deviceId = 0;
    DeviceType deviceType = DeviceType::ANY;

    // Layer range for this stage
    int startLayer = 0;
    int endLayer = -1;

    // Execution function
    std::function<void(MicroBatch&, bool forward)> executeFunc;

    // Memory estimates
    size_t parameterMemory = 0;
    size_t activationMemory = 0;
    size_t gradientMemory = 0;

    // State
    StageState state = StageState::IDLE;
    int currentMicroBatch = -1;

    // Timing stats
    double totalForwardTime = 0.0;
    double totalBackwardTime = 0.0;
    int forwardCount = 0;
    int backwardCount = 0;
};

/**
 * Schedule entry for a single time step
 */
struct SD_LIB_EXPORT ScheduleEntry {
    int timeStep = 0;
    int stageId = 0;
    int microBatchId = 0;
    bool isForward = true;           // Forward or backward pass
    bool isTransfer = false;         // Data transfer between stages
    int transferSrc = -1;
    int transferDst = -1;
};

/**
 * Pipeline execution result
 */
struct SD_LIB_EXPORT PipelineResult {
    bool success = false;
    std::string errorMessage;

    // Timing
    double totalTimeMs = 0.0;
    double computeTimeMs = 0.0;
    double transferTimeMs = 0.0;
    double idleTimeMs = 0.0;

    // Efficiency metrics
    float pipelineEfficiency = 0.0f;  // Compute time / Total time
    float bubbleRatio = 0.0f;         // Idle time / Total time

    // Per-stage stats
    std::vector<double> stageForwardTimes;
    std::vector<double> stageBackwardTimes;
};

/**
 * Activation checkpoint info
 */
struct SD_LIB_EXPORT CheckpointInfo {
    int layerId = 0;
    int stageId = 0;
    std::vector<NDArray*> activations;
    bool saved = false;
};

/**
 * Pipeline Scheduler - Coordinates pipeline parallel execution
 *
 * This class provides:
 * - Multiple pipeline schedules (GPipe, 1F1B, etc.)
 * - Micro-batch management
 * - Activation checkpointing
 * - Automatic scheduling optimization
 * - Cross-stage communication
 */
class SD_LIB_EXPORT PipelineScheduler {
public:
    /**
     * Create a pipeline scheduler with configuration
     */
    explicit PipelineScheduler(const ModelParallelConfig& config);

    ~PipelineScheduler();

    // =========================================================================
    // Setup
    // =========================================================================

    /**
     * Initialize the scheduler with stages
     */
    bool initialize(const std::vector<PipelineStage>& stages);

    /**
     * Set the pipeline schedule type
     */
    void setSchedule(PipelineSchedule schedule);

    /**
     * Get current schedule type
     */
    PipelineSchedule getSchedule() const { return _schedule; }

    /**
     * Set number of micro-batches
     */
    void setNumMicroBatches(int count);

    /**
     * Get number of micro-batches
     */
    int getNumMicroBatches() const { return _numMicroBatches; }

    /**
     * Set execution function for a stage
     */
    void setStageExecutor(int stageId, std::function<void(MicroBatch&, bool)> executor);

    /**
     * Enable/disable activation checkpointing
     */
    void setActivationCheckpointing(bool enabled, const std::vector<int>& layers = {});

    // =========================================================================
    // Execution
    // =========================================================================

    /**
     * Execute forward pass through pipeline
     * @param inputs Input tensors (per micro-batch)
     * @return Output tensors from last stage
     */
    std::vector<std::vector<NDArray*>> forward(
        const std::vector<std::vector<NDArray*>>& inputs
    );

    /**
     * Execute backward pass through pipeline
     * @param gradOutputs Gradient of loss w.r.t. outputs
     * @return Gradient of loss w.r.t. inputs
     */
    std::vector<std::vector<NDArray*>> backward(
        const std::vector<std::vector<NDArray*>>& gradOutputs
    );

    /**
     * Execute forward and backward passes
     */
    PipelineResult execute(
        const std::vector<std::vector<NDArray*>>& inputs,
        const std::vector<std::vector<NDArray*>>& targets,
        std::function<NDArray*(const NDArray*, const NDArray*)> lossFunc
    );

    /**
     * Execute a single micro-batch forward
     */
    std::vector<NDArray*> forwardMicroBatch(int microBatchId, const std::vector<NDArray*>& inputs);

    /**
     * Execute a single micro-batch backward
     */
    std::vector<NDArray*> backwardMicroBatch(int microBatchId, const std::vector<NDArray*>& gradOutputs);

    // =========================================================================
    // Scheduling
    // =========================================================================

    /**
     * Generate the execution schedule
     */
    std::vector<ScheduleEntry> generateSchedule();

    /**
     * Get the generated schedule entries
     */
    const std::vector<ScheduleEntry>& getScheduleEntries() const { return _scheduleEntries; }

    /**
     * Estimate pipeline bubble ratio
     */
    float estimateBubbleRatio() const;

    /**
     * Get estimated pipeline efficiency
     */
    float estimateEfficiency() const;

    /**
     * Find optimal number of micro-batches
     */
    int findOptimalMicroBatches(size_t batchSize, size_t memoryBudget);

    // =========================================================================
    // Communication
    // =========================================================================

    /**
     * Send activations from one stage to next
     */
    void sendActivations(int srcStage, int dstStage, MicroBatch& batch);

    /**
     * Receive activations from previous stage
     */
    void receiveActivations(int srcStage, int dstStage, MicroBatch& batch);

    /**
     * Send gradients from one stage to previous
     */
    void sendGradients(int srcStage, int dstStage, MicroBatch& batch);

    /**
     * Receive gradients from next stage
     */
    void receiveGradients(int srcStage, int dstStage, MicroBatch& batch);

    /**
     * Synchronize a pipeline stage
     */
    void synchronizeStage(int stageId);

    /**
     * Synchronize all stages
     */
    void synchronizeAll();

    // =========================================================================
    // Activation Checkpointing
    // =========================================================================

    /**
     * Checkpoint activations for backward pass
     */
    void checkpointActivations(int microBatchId, int layerId, const std::vector<NDArray*>& activations);

    /**
     * Retrieve checkpointed activations
     */
    std::vector<NDArray*> getCheckpointedActivations(int microBatchId, int layerId);

    /**
     * Recompute activations from checkpoint
     */
    std::vector<NDArray*> recomputeActivations(int microBatchId, int startLayer, int endLayer);

    /**
     * Clear all checkpoints
     */
    void clearCheckpoints();

    /**
     * Get memory used by checkpoints
     */
    size_t getCheckpointMemory() const;

    // =========================================================================
    // State Management
    // =========================================================================

    /**
     * Get stage info
     */
    const PipelineStage& getStage(int stageId) const;

    /**
     * Get all stages
     */
    const std::vector<PipelineStage>& getStages() const { return _stages; }

    /**
     * Get number of stages
     */
    int getNumStages() const { return static_cast<int>(_stages.size()); }

    /**
     * Get current stage state
     */
    StageState getStageState(int stageId) const;

    /**
     * Reset scheduler state
     */
    void reset();

    /**
     * Check if scheduler is busy
     */
    bool isBusy() const;

    // =========================================================================
    // Statistics
    // =========================================================================

    /**
     * Get pipeline execution result
     */
    PipelineResult getLastResult() const { return _lastResult; }

    /**
     * Get total forward time for a stage
     */
    double getStageForwardTime(int stageId) const;

    /**
     * Get total backward time for a stage
     */
    double getStageBackwardTime(int stageId) const;

    /**
     * Get pipeline throughput (micro-batches/second)
     */
    double getThroughput() const;

    /**
     * Print pipeline schedule visualization
     */
    void printSchedule() const;

    /**
     * Print pipeline statistics
     */
    void printStats() const;

private:
    // Schedule generation for different types
    std::vector<ScheduleEntry> generateGPipeSchedule();
    std::vector<ScheduleEntry> generate1F1BSchedule();
    std::vector<ScheduleEntry> generateInterleaved1F1BSchedule();
    std::vector<ScheduleEntry> generateZeroBubbleSchedule();
    std::vector<ScheduleEntry> generateAsyncSchedule();

    // Execution helpers
    void executeStageForward(int stageId, MicroBatch& batch);
    void executeStageBackward(int stageId, MicroBatch& batch);
    void transferBetweenStages(int srcStage, int dstStage, MicroBatch& batch, bool isGradient);

    // Micro-batch management
    MicroBatch& getMicroBatch(int id);
    void initializeMicroBatches(const std::vector<std::vector<NDArray*>>& inputs);
    void clearMicroBatches();

    // Member variables
    ModelParallelConfig _config;
    PipelineSchedule _schedule = PipelineSchedule::ONE_F_ONE_B;
    int _numMicroBatches = 1;

    std::vector<PipelineStage> _stages;
    std::vector<ScheduleEntry> _scheduleEntries;
    std::vector<MicroBatch> _microBatches;

    // Checkpointing
    bool _checkpointingEnabled = false;
    std::vector<int> _checkpointLayers;
    std::unordered_map<int, std::unordered_map<int, CheckpointInfo>> _checkpoints;

    // Synchronization
    mutable std::mutex _mutex;
    std::vector<std::unique_ptr<std::mutex>> _stageMutexes;
    std::vector<std::unique_ptr<std::condition_variable>> _stageConditions;

    // State
    std::atomic<bool> _initialized{false};
    std::atomic<bool> _executing{false};
    PipelineResult _lastResult;
};

/**
 * Pipeline context - RAII wrapper for pipeline execution
 */
class SD_LIB_EXPORT PipelineContext {
public:
    PipelineContext(
        PipelineScheduler& scheduler,
        const std::vector<std::vector<NDArray*>>& inputs
    );
    ~PipelineContext();

    /**
     * Execute forward pass
     */
    std::vector<std::vector<NDArray*>> forward();

    /**
     * Execute backward pass
     */
    std::vector<std::vector<NDArray*>> backward(
        const std::vector<std::vector<NDArray*>>& gradOutputs
    );

    /**
     * Get results
     */
    const PipelineResult& getResult() const;

private:
    PipelineScheduler& _scheduler;
    std::vector<std::vector<NDArray*>> _inputs;
    std::vector<std::vector<NDArray*>> _outputs;
    bool _forwardDone = false;
};

/**
 * Helper to create a pipeline scheduler
 */
inline std::unique_ptr<PipelineScheduler> createPipelineScheduler(
    const ModelParallelConfig& config
) {
    return std::make_unique<PipelineScheduler>(config);
}

/**
 * Helper to create stages from layer assignments
 */
inline std::vector<PipelineStage> createStagesFromAssignments(
    const std::vector<LayerAssignment>& assignments
) {
    std::vector<PipelineStage> stages;
    for (size_t i = 0; i < assignments.size(); ++i) {
        PipelineStage stage;
        stage.stageId = static_cast<int>(i);
        stage.startLayer = assignments[i].startLayer;
        stage.endLayer = assignments[i].endLayer;
        stage.deviceId = assignments[i].device.deviceIndex;
        stage.deviceType = assignments[i].device.type;
        stages.push_back(stage);
    }
    return stages;
}

}  // namespace modelparallel
}  // namespace sd

#endif  // SD_PIPELINE_SCHEDULER_H
