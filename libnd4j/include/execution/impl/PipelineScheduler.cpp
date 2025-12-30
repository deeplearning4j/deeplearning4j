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

#include <execution/PipelineScheduler.h>
#include <execution/DataTransferManager.h>

#include <chrono>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace sd {
namespace modelparallel {

PipelineScheduler::PipelineScheduler(const ModelParallelConfig& config)
    : _config(config)
    , _schedule(PipelineSchedule::ONE_F_ONE_B)
    , _numMicroBatches(config.numMicroBatches) {

    if (config.enableInterleaved) {
        _schedule = PipelineSchedule::INTERLEAVED_1F1B;
    }
}

PipelineScheduler::~PipelineScheduler() {
    clearCheckpoints();
    clearMicroBatches();
}

bool PipelineScheduler::initialize(const std::vector<PipelineStage>& stages) {
    std::lock_guard<std::mutex> lock(_mutex);

    if (stages.empty()) {
        return false;
    }

    _stages = stages;
    _stageMutexes.resize(stages.size());
    _stageConditions.resize(stages.size());

    // Generate initial schedule
    _scheduleEntries = generateSchedule();

    _initialized.store(true);
    return true;
}

void PipelineScheduler::setSchedule(PipelineSchedule schedule) {
    std::lock_guard<std::mutex> lock(_mutex);
    _schedule = schedule;
    _scheduleEntries = generateSchedule();
}

void PipelineScheduler::setNumMicroBatches(int count) {
    std::lock_guard<std::mutex> lock(_mutex);
    _numMicroBatches = count;
    _scheduleEntries = generateSchedule();
}

void PipelineScheduler::setStageExecutor(int stageId, std::function<void(MicroBatch&, bool)> executor) {
    if (stageId >= 0 && stageId < static_cast<int>(_stages.size())) {
        _stages[stageId].executeFunc = executor;
    }
}

void PipelineScheduler::setActivationCheckpointing(bool enabled, const std::vector<int>& layers) {
    _checkpointingEnabled = enabled;
    _checkpointLayers = layers;
}

std::vector<ScheduleEntry> PipelineScheduler::generateSchedule() {
    switch (_schedule) {
        case PipelineSchedule::GPIPE:
            return generateGPipeSchedule();
        case PipelineSchedule::ONE_F_ONE_B:
            return generate1F1BSchedule();
        case PipelineSchedule::INTERLEAVED_1F1B:
            return generateInterleaved1F1BSchedule();
        case PipelineSchedule::ZERO_BUBBLE:
            return generateZeroBubbleSchedule();
        case PipelineSchedule::ASYNC:
            return generateAsyncSchedule();
        default:
            return generateGPipeSchedule();
    }
}

std::vector<ScheduleEntry> PipelineScheduler::generateGPipeSchedule() {
    std::vector<ScheduleEntry> schedule;
    int numStages = static_cast<int>(_stages.size());

    // GPipe: All forward passes, then all backward passes
    int timeStep = 0;

    // Forward phase
    for (int mb = 0; mb < _numMicroBatches; ++mb) {
        for (int stage = 0; stage < numStages; ++stage) {
            ScheduleEntry entry;
            entry.timeStep = timeStep++;
            entry.stageId = stage;
            entry.microBatchId = mb;
            entry.isForward = true;
            schedule.push_back(entry);
        }
    }

    // Backward phase
    for (int mb = _numMicroBatches - 1; mb >= 0; --mb) {
        for (int stage = numStages - 1; stage >= 0; --stage) {
            ScheduleEntry entry;
            entry.timeStep = timeStep++;
            entry.stageId = stage;
            entry.microBatchId = mb;
            entry.isForward = false;
            schedule.push_back(entry);
        }
    }

    return schedule;
}

std::vector<ScheduleEntry> PipelineScheduler::generate1F1BSchedule() {
    std::vector<ScheduleEntry> schedule;
    int numStages = static_cast<int>(_stages.size());

    // 1F1B: Warm-up, steady state, cool-down
    int timeStep = 0;

    // Warm-up phase: forward passes to fill pipeline
    for (int warmup = 0; warmup < numStages; ++warmup) {
        for (int stage = 0; stage <= warmup && stage < numStages; ++stage) {
            ScheduleEntry entry;
            entry.timeStep = timeStep;
            entry.stageId = stage;
            entry.microBatchId = warmup - stage;
            entry.isForward = true;

            if (entry.microBatchId >= 0 && entry.microBatchId < _numMicroBatches) {
                schedule.push_back(entry);
            }
        }
        timeStep++;
    }

    // Steady state: 1F1B pattern
    for (int steady = 0; steady < _numMicroBatches - numStages; ++steady) {
        for (int stage = 0; stage < numStages; ++stage) {
            // Forward
            ScheduleEntry fwd;
            fwd.timeStep = timeStep;
            fwd.stageId = stage;
            fwd.microBatchId = numStages + steady - stage;
            fwd.isForward = true;

            if (fwd.microBatchId < _numMicroBatches) {
                schedule.push_back(fwd);
            }

            // Backward
            ScheduleEntry bwd;
            bwd.timeStep = timeStep;
            bwd.stageId = stage;
            bwd.microBatchId = steady + stage;
            bwd.isForward = false;

            if (bwd.microBatchId < _numMicroBatches) {
                schedule.push_back(bwd);
            }
        }
        timeStep++;
    }

    // Cool-down phase: remaining backward passes
    for (int cooldown = 0; cooldown < numStages; ++cooldown) {
        for (int stage = numStages - 1 - cooldown; stage < numStages; ++stage) {
            ScheduleEntry entry;
            entry.timeStep = timeStep;
            entry.stageId = stage;
            entry.microBatchId = _numMicroBatches - numStages + cooldown + (stage - (numStages - 1 - cooldown));
            entry.isForward = false;

            if (entry.microBatchId >= 0 && entry.microBatchId < _numMicroBatches) {
                schedule.push_back(entry);
            }
        }
        timeStep++;
    }

    return schedule;
}

std::vector<ScheduleEntry> PipelineScheduler::generateInterleaved1F1BSchedule() {
    // Interleaved 1F1B with virtual pipeline stages
    // For now, fall back to regular 1F1B
    return generate1F1BSchedule();
}

std::vector<ScheduleEntry> PipelineScheduler::generateZeroBubbleSchedule() {
    // Zero-bubble pipeline scheduling
    // This requires careful overlap of communication and computation
    // For now, fall back to 1F1B
    return generate1F1BSchedule();
}

std::vector<ScheduleEntry> PipelineScheduler::generateAsyncSchedule() {
    // Fully asynchronous schedule (may have staleness)
    // Each stage processes micro-batches independently
    std::vector<ScheduleEntry> schedule;
    int numStages = static_cast<int>(_stages.size());

    int timeStep = 0;
    for (int mb = 0; mb < _numMicroBatches; ++mb) {
        // Forward pass
        for (int stage = 0; stage < numStages; ++stage) {
            ScheduleEntry entry;
            entry.timeStep = timeStep++;
            entry.stageId = stage;
            entry.microBatchId = mb;
            entry.isForward = true;
            schedule.push_back(entry);
        }

        // Backward pass (immediately after forward for each micro-batch)
        for (int stage = numStages - 1; stage >= 0; --stage) {
            ScheduleEntry entry;
            entry.timeStep = timeStep++;
            entry.stageId = stage;
            entry.microBatchId = mb;
            entry.isForward = false;
            schedule.push_back(entry);
        }
    }

    return schedule;
}

std::vector<std::vector<NDArray*>> PipelineScheduler::forward(
    const std::vector<std::vector<NDArray*>>& inputs
) {
    if (!_initialized.load()) {
        return {};
    }

    _executing.store(true);

    // Initialize micro-batches
    initializeMicroBatches(inputs);

    auto startTime = std::chrono::high_resolution_clock::now();

    // Execute forward schedule entries
    for (const auto& entry : _scheduleEntries) {
        if (entry.isForward) {
            auto& batch = getMicroBatch(entry.microBatchId);
            executeStageForward(entry.stageId, batch);
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    _lastResult.totalTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    // Collect outputs from last stage
    std::vector<std::vector<NDArray*>> outputs;
    for (int mb = 0; mb < _numMicroBatches; ++mb) {
        outputs.push_back(_microBatches[mb].outputs);
    }

    _executing.store(false);
    return outputs;
}

std::vector<std::vector<NDArray*>> PipelineScheduler::backward(
    const std::vector<std::vector<NDArray*>>& gradOutputs
) {
    if (!_initialized.load()) {
        return {};
    }

    _executing.store(true);

    // Set gradient outputs for each micro-batch
    for (int mb = 0; mb < _numMicroBatches; ++mb) {
        _microBatches[mb].gradOutputs = gradOutputs[mb];
    }

    // Execute backward schedule entries
    for (const auto& entry : _scheduleEntries) {
        if (!entry.isForward) {
            auto& batch = getMicroBatch(entry.microBatchId);
            executeStageBackward(entry.stageId, batch);
        }
    }

    // Collect gradient inputs from first stage
    std::vector<std::vector<NDArray*>> gradInputs;
    for (int mb = 0; mb < _numMicroBatches; ++mb) {
        gradInputs.push_back(_microBatches[mb].gradInputs);
    }

    _executing.store(false);
    return gradInputs;
}

PipelineResult PipelineScheduler::execute(
    const std::vector<std::vector<NDArray*>>& inputs,
    const std::vector<std::vector<NDArray*>>& targets,
    std::function<NDArray*(const NDArray*, const NDArray*)> lossFunc
) {
    PipelineResult result;
    auto startTime = std::chrono::high_resolution_clock::now();

    // Forward pass
    auto outputs = forward(inputs);

    // Compute loss and gradients
    std::vector<std::vector<NDArray*>> gradOutputs;
    for (int mb = 0; mb < _numMicroBatches; ++mb) {
        std::vector<NDArray*> grads;
        for (size_t i = 0; i < outputs[mb].size(); ++i) {
            auto* loss = lossFunc(outputs[mb][i], targets[mb][i]);
            // Gradient of loss w.r.t. output is computed by loss function
            grads.push_back(loss);
        }
        gradOutputs.push_back(grads);
    }

    // Backward pass
    backward(gradOutputs);

    auto endTime = std::chrono::high_resolution_clock::now();
    result.totalTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    // Calculate efficiency
    result.computeTimeMs = 0;
    for (const auto& stage : _stages) {
        result.computeTimeMs += stage.totalForwardTime + stage.totalBackwardTime;
    }
    result.pipelineEfficiency = static_cast<float>(result.computeTimeMs / result.totalTimeMs);
    result.bubbleRatio = 1.0f - result.pipelineEfficiency;

    result.success = true;
    _lastResult = result;
    return result;
}

void PipelineScheduler::executeStageForward(int stageId, MicroBatch& batch) {
    auto& stage = _stages[stageId];

    batch.currentStage = stageId;
    stage.state = StageState::EXECUTING_FORWARD;
    stage.currentMicroBatch = batch.microBatchId;

    auto startTime = std::chrono::high_resolution_clock::now();

    // Receive activations from previous stage
    if (stageId > 0) {
        receiveActivations(stageId - 1, stageId, batch);
    }

    // Execute stage
    if (stage.executeFunc) {
        stage.executeFunc(batch, true);  // true = forward
    }

    // Checkpoint activations if enabled
    if (_checkpointingEnabled && !_checkpointLayers.empty()) {
        for (int layer = stage.startLayer; layer <= stage.endLayer; ++layer) {
            if (std::find(_checkpointLayers.begin(), _checkpointLayers.end(), layer) !=
                _checkpointLayers.end()) {
                checkpointActivations(batch.microBatchId, layer, batch.activations);
            }
        }
    }

    // Send activations to next stage
    if (stageId < static_cast<int>(_stages.size()) - 1) {
        sendActivations(stageId, stageId + 1, batch);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    batch.forwardEndTime = batch.forwardStartTime + duration;
    stage.totalForwardTime += duration;
    stage.forwardCount++;
    stage.state = StageState::IDLE;

    batch.forwardComplete = true;
}

void PipelineScheduler::executeStageBackward(int stageId, MicroBatch& batch) {
    auto& stage = _stages[stageId];

    batch.currentStage = stageId;
    stage.state = StageState::EXECUTING_BACKWARD;
    stage.currentMicroBatch = batch.microBatchId;

    auto startTime = std::chrono::high_resolution_clock::now();

    // Receive gradients from next stage
    if (stageId < static_cast<int>(_stages.size()) - 1) {
        receiveGradients(stageId + 1, stageId, batch);
    }

    // Recompute activations if using checkpointing
    if (_checkpointingEnabled) {
        auto checkpointedActivations = getCheckpointedActivations(batch.microBatchId, stage.startLayer);
        if (!checkpointedActivations.empty()) {
            batch.activations = recomputeActivations(batch.microBatchId, stage.startLayer, stage.endLayer);
        }
    }

    // Execute backward
    if (stage.executeFunc) {
        stage.executeFunc(batch, false);  // false = backward
    }

    // Send gradients to previous stage
    if (stageId > 0) {
        sendGradients(stageId, stageId - 1, batch);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    batch.backwardEndTime = batch.backwardStartTime + duration;
    stage.totalBackwardTime += duration;
    stage.backwardCount++;
    stage.state = StageState::IDLE;

    batch.backwardComplete = true;
}

void PipelineScheduler::sendActivations(int srcStage, int dstStage, MicroBatch& batch) {
    auto& transferMgr = DataTransferManager::getInstance();

    int srcDevice = _stages[srcStage].deviceId;
    int dstDevice = _stages[dstStage].deviceId;

    if (srcDevice != dstDevice) {
        // Transfer each output tensor to the next stage
        for (auto* tensor : batch.outputs) {
            // In practice, would need to allocate on destination device
            // For now, just synchronize
            transferMgr.synchronizeDevice(srcDevice);
        }
    }
}

void PipelineScheduler::receiveActivations(int srcStage, int dstStage, MicroBatch& batch) {
    auto& transferMgr = DataTransferManager::getInstance();

    int srcDevice = _stages[srcStage].deviceId;
    int dstDevice = _stages[dstStage].deviceId;

    if (srcDevice != dstDevice) {
        transferMgr.synchronizeDevice(dstDevice);
    }

    // Copy outputs from previous stage to inputs for this stage
    batch.inputs = batch.outputs;
}

void PipelineScheduler::sendGradients(int srcStage, int dstStage, MicroBatch& batch) {
    auto& transferMgr = DataTransferManager::getInstance();

    int srcDevice = _stages[srcStage].deviceId;
    int dstDevice = _stages[dstStage].deviceId;

    if (srcDevice != dstDevice) {
        transferMgr.synchronizeDevice(srcDevice);
    }
}

void PipelineScheduler::receiveGradients(int srcStage, int dstStage, MicroBatch& batch) {
    auto& transferMgr = DataTransferManager::getInstance();

    int srcDevice = _stages[srcStage].deviceId;
    int dstDevice = _stages[dstStage].deviceId;

    if (srcDevice != dstDevice) {
        transferMgr.synchronizeDevice(dstDevice);
    }

    batch.gradInputs = batch.gradOutputs;
}

void PipelineScheduler::synchronizeStage(int stageId) {
    if (stageId >= 0 && stageId < static_cast<int>(_stages.size())) {
        DataTransferManager::getInstance().synchronizeDevice(_stages[stageId].deviceId);
    }
}

void PipelineScheduler::synchronizeAll() {
    DataTransferManager::getInstance().synchronizeAll();
}

void PipelineScheduler::checkpointActivations(int microBatchId, int layerId, const std::vector<NDArray*>& activations) {
    CheckpointInfo info;
    info.layerId = layerId;
    info.activations = activations;
    info.saved = true;

    _checkpoints[microBatchId][layerId] = info;
}

std::vector<NDArray*> PipelineScheduler::getCheckpointedActivations(int microBatchId, int layerId) {
    auto it = _checkpoints.find(microBatchId);
    if (it != _checkpoints.end()) {
        auto layerIt = it->second.find(layerId);
        if (layerIt != it->second.end() && layerIt->second.saved) {
            return layerIt->second.activations;
        }
    }
    return {};
}

std::vector<NDArray*> PipelineScheduler::recomputeActivations(int microBatchId, int startLayer, int endLayer) {
    // Would need to re-run forward pass from checkpoint
    // For now, return empty (actual implementation would recompute)
    return {};
}

void PipelineScheduler::clearCheckpoints() {
    _checkpoints.clear();
}

size_t PipelineScheduler::getCheckpointMemory() const {
    size_t total = 0;
    for (const auto& [mbId, layers] : _checkpoints) {
        for (const auto& [layerId, info] : layers) {
            for (const auto* tensor : info.activations) {
                if (tensor) {
                    total += tensor->lengthOf() * DataTypeUtils::sizeOf(tensor->dataType());
                }
            }
        }
    }
    return total;
}

const PipelineStage& PipelineScheduler::getStage(int stageId) const {
    static PipelineStage empty;
    if (stageId >= 0 && stageId < static_cast<int>(_stages.size())) {
        return _stages[stageId];
    }
    return empty;
}

StageState PipelineScheduler::getStageState(int stageId) const {
    if (stageId >= 0 && stageId < static_cast<int>(_stages.size())) {
        return _stages[stageId].state;
    }
    return StageState::IDLE;
}

void PipelineScheduler::reset() {
    std::lock_guard<std::mutex> lock(_mutex);

    clearMicroBatches();
    clearCheckpoints();

    for (auto& stage : _stages) {
        stage.state = StageState::IDLE;
        stage.currentMicroBatch = -1;
        stage.totalForwardTime = 0;
        stage.totalBackwardTime = 0;
        stage.forwardCount = 0;
        stage.backwardCount = 0;
    }

    _lastResult = PipelineResult();
}

bool PipelineScheduler::isBusy() const {
    return _executing.load();
}

float PipelineScheduler::estimateBubbleRatio() const {
    int numStages = static_cast<int>(_stages.size());
    int warmupSteps = numStages - 1;
    int cooldownSteps = numStages - 1;
    int totalSteps = _numMicroBatches + warmupSteps + cooldownSteps;

    // Bubble ratio = (warmup + cooldown) / total
    return static_cast<float>(warmupSteps + cooldownSteps) / static_cast<float>(totalSteps);
}

float PipelineScheduler::estimateEfficiency() const {
    return 1.0f - estimateBubbleRatio();
}

int PipelineScheduler::findOptimalMicroBatches(size_t batchSize, size_t memoryBudget) {
    int numStages = static_cast<int>(_stages.size());

    // More micro-batches = less bubble but more activation memory
    // Optimal is typically numStages * 2 to numStages * 4
    int minMB = numStages;
    int maxMB = static_cast<int>(batchSize);

    // Estimate memory per micro-batch
    size_t estimatedActivationMemory = memoryBudget / maxMB;

    // Find largest micro-batch count that fits in memory
    for (int mb = maxMB; mb >= minMB; --mb) {
        size_t totalMemory = mb * estimatedActivationMemory;
        if (totalMemory <= memoryBudget) {
            return mb;
        }
    }

    return minMB;
}

double PipelineScheduler::getStageForwardTime(int stageId) const {
    if (stageId >= 0 && stageId < static_cast<int>(_stages.size())) {
        return _stages[stageId].totalForwardTime;
    }
    return 0.0;
}

double PipelineScheduler::getStageBackwardTime(int stageId) const {
    if (stageId >= 0 && stageId < static_cast<int>(_stages.size())) {
        return _stages[stageId].totalBackwardTime;
    }
    return 0.0;
}

double PipelineScheduler::getThroughput() const {
    if (_lastResult.totalTimeMs > 0) {
        return (_numMicroBatches / _lastResult.totalTimeMs) * 1000.0;
    }
    return 0.0;
}

MicroBatch& PipelineScheduler::getMicroBatch(int id) {
    if (id >= static_cast<int>(_microBatches.size())) {
        _microBatches.resize(id + 1);
    }
    return _microBatches[id];
}

void PipelineScheduler::initializeMicroBatches(const std::vector<std::vector<NDArray*>>& inputs) {
    _microBatches.resize(_numMicroBatches);

    for (int mb = 0; mb < _numMicroBatches; ++mb) {
        _microBatches[mb].microBatchId = mb;
        if (mb < static_cast<int>(inputs.size())) {
            _microBatches[mb].inputs = inputs[mb];
        }
        _microBatches[mb].forwardComplete = false;
        _microBatches[mb].backwardComplete = false;
        _microBatches[mb].currentStage = 0;
    }
}

void PipelineScheduler::clearMicroBatches() {
    _microBatches.clear();
}

void PipelineScheduler::printSchedule() const {
    std::cout << "=== Pipeline Schedule ===" << std::endl;
    std::cout << "Type: ";
    switch (_schedule) {
        case PipelineSchedule::GPIPE: std::cout << "GPipe"; break;
        case PipelineSchedule::ONE_F_ONE_B: std::cout << "1F1B"; break;
        case PipelineSchedule::INTERLEAVED_1F1B: std::cout << "Interleaved 1F1B"; break;
        case PipelineSchedule::ZERO_BUBBLE: std::cout << "Zero Bubble"; break;
        case PipelineSchedule::ASYNC: std::cout << "Async"; break;
    }
    std::cout << std::endl;
    std::cout << "Stages: " << _stages.size() << std::endl;
    std::cout << "Micro-batches: " << _numMicroBatches << std::endl;
    std::cout << "Estimated efficiency: " << std::fixed << std::setprecision(2)
              << (estimateEfficiency() * 100) << "%" << std::endl;

    std::cout << "\nSchedule entries: " << _scheduleEntries.size() << std::endl;
    for (const auto& entry : _scheduleEntries) {
        std::cout << "  t=" << entry.timeStep
                  << " stage=" << entry.stageId
                  << " mb=" << entry.microBatchId
                  << " " << (entry.isForward ? "FWD" : "BWD")
                  << std::endl;
    }
}

void PipelineScheduler::printStats() const {
    std::cout << "=== Pipeline Statistics ===" << std::endl;
    std::cout << "Total time: " << _lastResult.totalTimeMs << " ms" << std::endl;
    std::cout << "Compute time: " << _lastResult.computeTimeMs << " ms" << std::endl;
    std::cout << "Efficiency: " << std::fixed << std::setprecision(2)
              << (_lastResult.pipelineEfficiency * 100) << "%" << std::endl;
    std::cout << "Bubble ratio: " << std::fixed << std::setprecision(2)
              << (_lastResult.bubbleRatio * 100) << "%" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << getThroughput() << " micro-batches/sec" << std::endl;

    std::cout << "\nPer-stage times:" << std::endl;
    for (size_t i = 0; i < _stages.size(); ++i) {
        std::cout << "  Stage " << i << ": "
                  << "fwd=" << _stages[i].totalForwardTime << "ms "
                  << "bwd=" << _stages[i].totalBackwardTime << "ms"
                  << std::endl;
    }
}

// PipelineContext implementation
PipelineContext::PipelineContext(
    PipelineScheduler& scheduler,
    const std::vector<std::vector<NDArray*>>& inputs
)
    : _scheduler(scheduler)
    , _inputs(inputs) {
}

PipelineContext::~PipelineContext() {
}

std::vector<std::vector<NDArray*>> PipelineContext::forward() {
    _outputs = _scheduler.forward(_inputs);
    _forwardDone = true;
    return _outputs;
}

std::vector<std::vector<NDArray*>> PipelineContext::backward(
    const std::vector<std::vector<NDArray*>>& gradOutputs
) {
    if (!_forwardDone) {
        forward();
    }
    return _scheduler.backward(gradOutputs);
}

const PipelineResult& PipelineContext::getResult() const {
    return _scheduler.getLastResult();
}

}  // namespace modelparallel
}  // namespace sd
