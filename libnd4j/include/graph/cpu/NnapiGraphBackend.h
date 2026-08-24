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

#ifndef LIBND4J_NNAPI_GRAPH_BACKEND_H
#define LIBND4J_NNAPI_GRAPH_BACKEND_H

#include <graph/GraphBackend.h>
#include <graph/GraphBackendCommon.h>
#include <graph/NativeDynamicShapePlan.h>

#if HAVE_NNAPI

#include <android/NeuralNetworks.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sd {
namespace graph {

/**
 * Android NNAPI graph backend for the native plan executor.
 *
 * Routes DSP segments through Android's Neural Networks API (NNAPI) to
 * leverage hardware accelerators available on the device:
 *   - Qualcomm Hexagon DSP
 *   - ARM Mali / Qualcomm Adreno GPU
 *   - Dedicated NPU (Neural Processing Unit)
 *   - CPU fallback (via NNAPI's built-in CPU reference implementation)
 *
 * NNAPI translates the high-level operation graph into vendor-specific
 * accelerator code at model compilation time. The compiled model is then
 * executed repeatedly with near-zero dispatch overhead.
 *
 * Op mapping: nd4j ops are mapped to NNAPI operation codes
 * (ANEURALNETWORKS_*). Per-op capability is exposed to the shared resolver,
 * which partitions segments when the ordered capable-backend set changes.
 *
 * API level requirements:
 *   - API 27 (NNAPI 1.0): add, sub, mul, div, relu, sigmoid, tanh, softmax,
 *     conv2d, pooling, fully_connected, reshape, concatenation, lrn
 *   - API 29 (NNAPI 1.2): comparison, logical, reduce_*, argmax/argmin, cast,
 *     pow, select, resize, abs, exp, log, neg, sqrt, sin, squeeze, expand_dims,
 *     pad, tile, split, gather, transpose, space_to_batch, batch_to_space
 *   - API 30 (NNAPI 1.3): batch_matmul, quantized signed ops
 *
 * Resolution, lifecycle hints, and segment admission are exposed through the
 * shared GraphBackend contract.
 */
class NnapiGraphBackend : public GraphBackend {
 public:
  NnapiGraphBackend();
  ~NnapiGraphBackend() override;

  const char* name() const override { return "Android NNAPI"; }
  bool isAvailable() const override;
  bool isResolvable(const GraphBackendRequest& request) const override;
  int resolutionPriority(const GraphBackendRequest& request) const override;
  GraphBackendPlanningPolicy planningPolicy(
      const GraphBackendRequest& request) const override;
  bool canResolveSlot(const GraphBackendRequest& request, NativeSlot* slots,
                      int slotIndex) override;
  bool canResolveSegment(const GraphBackendRequest& request, NativeSlot* slots,
                         int start, int end) override;
  bool canFuseSegment(NativeSlot* slots, int start, int end) override;

  bool compileSegment(GraphSegment& seg, NativeSlot* slots,
                      NDArray** externalInputs, int numExternalInputs,
                      NDArray** outputSlots, int totalOutputSlots,
                      LongType shapeKey,
                      int totalSlots = 0,
                      int* requestedOutputSlotIndices = nullptr,
                      int numRequestedOutputs = 0) override;

  bool compileSegment(const GraphBackendRequest& request,
                      GraphSegment& seg, NativeSlot* slots,
                      NDArray** externalInputs, int numExternalInputs,
                      NDArray** outputSlots, int totalOutputSlots,
                      LongType shapeKey, int totalSlots,
                      int* requestedOutputSlotIndices,
                      int numRequestedOutputs) override;

  Status executeSegment(GraphSegment& seg, NativeSlot* slots,
                        NDArray** externalInputs, int numExternalInputs,
                        NDArray** outputSlots, int totalOutputSlots,
                        void* stream) override;

  void invalidateCache() override;

  std::vector<CompilationAuditEntry> getLastCompilationAudit() const override;

  static NnapiGraphBackend& getInstance();

  /// Set preferred execution preference (default: PREFER_SUSTAINED_SPEED)
  void setPreference(int preference) { preference_ = preference; }

 private:
  bool compileSegmentImpl(const GraphBackendRequest* request,
                          GraphSegment& seg, NativeSlot* slots,
                          NDArray** externalInputs, int numExternalInputs,
                          NDArray** outputSlots, int totalOutputSlots,
                          LongType shapeKey, int totalSlots,
                          int* requestedOutputSlotIndices,
                          int numRequestedOutputs);

  bool nnapiAvailable_ = false;
  int apiLevel_ = 0;
  int preference_ = ANEURALNETWORKS_PREFER_SUSTAINED_SPEED;
  const ANeuralNetworksDevice* requiredDevice_ = nullptr;
  std::string requiredDeviceName_;
  std::string selectedDeviceName_;
  std::string selectedDeviceVersion_;
  int32_t selectedDeviceType_ = ANEURALNETWORKS_DEVICE_UNKNOWN;
  int64_t selectedDeviceFeatureLevel_ = 0;

  // Resolve the accelerator-only device contract before the backend is admitted.
  // Tensor G3 must bind google-edgetpu; generic NNAPI selection is not proof of
  // accelerator placement because Android may otherwise partition onto CPU.
  bool resolveRequiredAcceleratorDevice();

  // NNAPI operand type from nd4j DataType. Returns -1 if unsupported.
  static int32_t toNnapiOperandType(DataType dt);

  // Check if a DataType can be represented in NNAPI
  static bool isNnapiSupportedType(DataType dt);

  // Map an nd4j op name to an NNAPI operation code.
  // Returns -1 if unmappable.
  static int getNnapiOpCode(const std::string& opName);

  // Minimum API level required for a given op. Returns 27 for basic ops.
  static int getMinApiLevel(const std::string& opName);

  // Validate the concrete wiring/parameter contract consumed by addImplicitParams.
  // Admission and model construction must share this guard so a mapped operation
  // can never reach vendor compilation with an ambiguous lowering.
  static bool validateSlotContract(const NativeSlot& slot, int nnapiOp,
                                   std::string& reason);
  bool isSlotResolvable(NativeSlot* slots, int slotIndex) const;

  // Compiled NNAPI model for a segment
  struct CompiledModel {
    static constexpr int kQ4KLoweringAbiVersion = 1;

    enum class BoundaryTransform : uint8_t {
      NONE = 0,
      INT64_TO_INT32 = 1,
      QUANTIZE_ASYMM_SIGNED = 2,
      DEQUANTIZE_ASYMM_SIGNED = 3,
    };

    struct QuantizedQ4KConstant {
      int slotIndex = -1;
      int sourceIndex = 0;
      LongType outputChannels = 0;
      LongType inputChannels = 0;
      float activationScale = 0.0f;
      float outputScale = 0.0f;
      std::vector<int8_t> filter;
      std::vector<float> perChannelScales;
      std::vector<int32_t> zeroBias;
      std::string packedWeightDigest;
      std::string loweringDigest;

      size_t ownedBytes() const {
        return filter.size() * sizeof(int8_t) +
               perChannelScales.size() * sizeof(float) +
               zeroBias.size() * sizeof(int32_t);
      }
    };

    ANeuralNetworksModel* model = nullptr;
    ANeuralNetworksCompilation* compilation = nullptr;
    int startSlot = -1;
    int endSlot = -1;
    LongType shapeKey = 0;
    bool valid = false;

    // Operand index mapping: maps external input/output source indices
    // to NNAPI operand indices within the model
    struct OperandMapping {
      int sourceIndex;   // nd4j source index (<0 = external, >=0 = outputSlot)
      uint32_t operand;  // NNAPI operand index
      bool isOutput;
      DataType sourceDataType;
      DataType bindingDataType;
      std::vector<LongType> dimensions;
      BoundaryTransform boundaryTransform = BoundaryTransform::NONE;
      float quantizationScale = 0.0f;
      int32_t quantizationZeroPoint = 0;
    };
    std::vector<OperandMapping> inputMappings;
    std::vector<OperandMapping> outputMappings;

    // Backend-owned lowering state. NNAPI retains pointers for constants larger
    // than its immediate-copy threshold, so these buffers must outlive both the
    // model and every execution created from its compilation.
    std::vector<QuantizedQ4KConstant> q4kConstants;
    std::string sourceWeightIdentity;
    std::string loweringCacheIdentity;

    // Compilation audit
    std::vector<CompilationAuditEntry> compilationAudit;
    std::mutex executionMutex;

    void invalidate() {
      std::lock_guard<std::mutex> lock(executionMutex);
      if (compilation) {
        ANeuralNetworksCompilation_free(compilation);
        compilation = nullptr;
      }
      if (model) {
        ANeuralNetworksModel_free(model);
        model = nullptr;
      }
      valid = false;
    }

    ~CompiledModel() { invalidate(); }

    // Non-copyable, moveable
    CompiledModel() = default;
    CompiledModel(const CompiledModel&) = delete;
    CompiledModel& operator=(const CompiledModel&) = delete;
    CompiledModel(CompiledModel&&) = delete;
    CompiledModel& operator=(CompiledModel&&) = delete;
  };

  // Non-owning registry used only by invalidateCache(). The plan's GraphSegment
  // owns each compiled artifact, so destroying a context releases its NNAPI
  // model/compilation instead of retaining it in this process-wide singleton.
  std::vector<std::weak_ptr<CompiledModel>> compiledArtifacts_;
  std::mutex cacheMtx_;
  std::vector<CompilationAuditEntry> lastCompilationAudit_;

  // Build the NNAPI model for a segment — adds operands and operations
  bool buildModel(ANeuralNetworksModel* model, CompiledModel& compiled,
                  NativeSlot* slots, int startSlot, int endSlot,
                  NDArray** externalInputs, int numExternalInputs,
                  NDArray** outputSlots, int totalOutputSlots,
                  int totalSlots,
                  const int* requestedOutputSlotIndices,
                  int numRequestedOutputs,
                  std::vector<int>& operationSourceSlots);

  // Add an NDArray as an NNAPI operand, returning its operand index in
  // *outIdx. If the array is non-contiguous, it will be dup()'d to a
  // contiguous copy stored in contiguousCopies for lifetime management.
  // Unsupported floating types (e.g. BFLOAT16) are promoted to FLOAT32.
  // Unsupported non-floating types (e.g. INT64) fail the model build so the
  // segment falls back instead of computing on silently converted values.
  bool addOperand(ANeuralNetworksModel* model, NDArray* arr, uint32_t& nextOperand,
                  std::vector<std::unique_ptr<NDArray>>& contiguousCopies,
                  uint32_t* outIdx);

  // Add a scalar constant operand (for fused activation codes, axis, etc.)
  uint32_t addScalarOperand(ANeuralNetworksModel* model, int32_t value, uint32_t& nextOperand);
  uint32_t addFloatOperand(ANeuralNetworksModel* model, float value, uint32_t& nextOperand);
  uint32_t addBoolOperand(ANeuralNetworksModel* model, bool value, uint32_t& nextOperand);

  // Add a 1D tensor constant operand from LongType* data (converted to int32)
  uint32_t addIntVectorOperand(ANeuralNetworksModel* model, const LongType* data, int count,
                               uint32_t& nextOperand,
                               std::vector<std::vector<int32_t>>& vectorStorage);

  // Add a shape tensor operand (1D INT32 tensor with the given shape values)
  uint32_t addShapeOperand(ANeuralNetworksModel* model, NDArray* arr, uint32_t& nextOperand,
                           std::vector<std::vector<int32_t>>& vectorStorage);

  // Add op-specific implicit parameters (padding, stride, axis, etc.)
  // Returns false if the op requires params we can't provide.
  bool addImplicitParams(ANeuralNetworksModel* model, NativeSlot& slot, int nnapiOp,
                         std::vector<uint32_t>& inputOperands, uint32_t& nextOperand,
                         NDArray** externalInputs, int numExternalInputs,
                         NDArray** outputSlots, int totalOutputSlots,
                         std::vector<std::vector<int32_t>>& vectorStorage);
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_NNAPI
#endif  // LIBND4J_NNAPI_GRAPH_BACKEND_H
