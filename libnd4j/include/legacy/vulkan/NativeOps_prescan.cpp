/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <execution/vulkan/VulkanExecutionStream.h>
#include <graph/vulkan/VulkanDeviceContext.h>
#include <graph/vulkan/VulkanMemoryPool.h>
#if defined(HAVE_MLIR) && HAVE_MLIR
#include <graph/vulkan/VulkanPipelineCache.h>
#endif
#include <legacy/NativeOps.h>

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>


namespace {

constexpr uint32_t kScanThreads = 128;
constexpr uint32_t kScanElementsPerBlock = 256;
constexpr uint32_t kPushConstantBytes = sizeof(uint32_t);
constexpr char kBlockScanPipelineKey[] = "native_prescan_i32_blocks_v1";
constexpr char kUniformAddPipelineKey[] = "native_prescan_i32_uniform_add_v1";

void setPrescanError(sd::Status status, const std::string& message) {
  safeSetErrorContext(static_cast<int>(status), message.c_str());
}

sd::graph::VulkanExecutionStream* resolvePrescanStream(
    void* opaque, int deviceId, std::string& error) {
  auto* stream =
      opaque != nullptr
          ? sd::graph::VulkanExecutionStream::fromOpaque(opaque, false)
          : sd::graph::VulkanExecutionStream::currentOrDefault(deviceId);
  if (stream == nullptr || !stream->isActive()) {
    error = "Vulkan prescan execution stream is unavailable";
    return nullptr;
  }
  if (stream->deviceId() != deviceId) {
    error = "Vulkan prescan stream does not belong to the allocation owner";
    return nullptr;
  }
  return stream;
}

#if defined(HAVE_MLIR) && HAVE_MLIR

const std::string& blockScanModule() {
  static const std::string module = R"mlir(
module {
  spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
    spirv.GlobalVariable @input bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
    spirv.GlobalVariable @output bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
    spirv.GlobalVariable @block_sums bind(0, 2) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
    spirv.GlobalVariable @params : !spirv.ptr<!spirv.struct<(i32 [0])>, PushConstant>
    spirv.GlobalVariable @local_id built_in("LocalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
    spirv.GlobalVariable @group_id built_in("WorkgroupId") : !spirv.ptr<vector<3xi32>, Input>
    spirv.GlobalVariable @scratch : !spirv.ptr<!spirv.array<256 x i32>, Workgroup>

    spirv.func @main() "None" {
      %zero = spirv.Constant 0 : i32
      %one = spirv.Constant 1 : i32
      %two = spirv.Constant 2 : i32
      %c128 = spirv.Constant 128 : i32
      %c255 = spirv.Constant 255 : i32
      %c256 = spirv.Constant 256 : i32

      %local_ptr = spirv.mlir.addressof @local_id : !spirv.ptr<vector<3xi32>, Input>
      %local_vec = spirv.Load "Input" %local_ptr : vector<3xi32>
      %tid = spirv.CompositeExtract %local_vec[0 : i32] : vector<3xi32>
      %group_ptr = spirv.mlir.addressof @group_id : !spirv.ptr<vector<3xi32>, Input>
      %group_vec = spirv.Load "Input" %group_ptr : vector<3xi32>
      %group = spirv.CompositeExtract %group_vec[0 : i32] : vector<3xi32>

      %params_ptr = spirv.mlir.addressof @params : !spirv.ptr<!spirv.struct<(i32 [0])>, PushConstant>
      %count_ptr = spirv.AccessChain %params_ptr[%zero] : !spirv.ptr<!spirv.struct<(i32 [0])>, PushConstant>, i32 -> !spirv.ptr<i32, PushConstant>
      %count = spirv.Load "PushConstant" %count_ptr : i32

      %input_ptr = spirv.mlir.addressof @input : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
      %output_ptr = spirv.mlir.addressof @output : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
      %sums_ptr = spirv.mlir.addressof @block_sums : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
      %scratch_ptr = spirv.mlir.addressof @scratch : !spirv.ptr<!spirv.array<256 x i32>, Workgroup>

      %base = spirv.IMul %group, %c256 : i32
      %index0 = spirv.IAdd %base, %tid : i32
      %index1 = spirv.IAdd %index0, %c128 : i32
      %lane1 = spirv.IAdd %tid, %c128 : i32
      %value0 = spirv.Variable init(%zero) : !spirv.ptr<i32, Function>
      %value1 = spirv.Variable init(%zero) : !spirv.ptr<i32, Function>

      %valid0 = spirv.SLessThan %index0, %count : i32
      spirv.mlir.selection {
        spirv.BranchConditional %valid0, ^then, ^merge
      ^then:
        %source0 = spirv.AccessChain %input_ptr[%zero, %index0] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
        %loaded0 = spirv.Load "StorageBuffer" %source0 : i32
        spirv.Store "Function" %value0, %loaded0 : i32
        spirv.Branch ^merge
      ^merge:
        spirv.mlir.merge
      }

      %valid1 = spirv.SLessThan %index1, %count : i32
      spirv.mlir.selection {
        spirv.BranchConditional %valid1, ^then, ^merge
      ^then:
        %source1 = spirv.AccessChain %input_ptr[%zero, %index1] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
        %loaded1 = spirv.Load "StorageBuffer" %source1 : i32
        spirv.Store "Function" %value1, %loaded1 : i32
        spirv.Branch ^merge
      ^merge:
        spirv.mlir.merge
      }

      %loaded_value0 = spirv.Load "Function" %value0 : i32
      %loaded_value1 = spirv.Load "Function" %value1 : i32
      %scratch0 = spirv.AccessChain %scratch_ptr[%tid] : !spirv.ptr<!spirv.array<256 x i32>, Workgroup>, i32 -> !spirv.ptr<i32, Workgroup>
      %scratch1 = spirv.AccessChain %scratch_ptr[%lane1] : !spirv.ptr<!spirv.array<256 x i32>, Workgroup>, i32 -> !spirv.ptr<i32, Workgroup>
      spirv.Store "Workgroup" %scratch0, %loaded_value0 : i32
      spirv.Store "Workgroup" %scratch1, %loaded_value1 : i32
      spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>

      spirv.mlir.loop {
        spirv.Branch ^header(%one : i32)
      ^header(%offset: i32):
        %continue = spirv.SLessThan %offset, %c256 : i32
        spirv.BranchConditional %continue, ^body, ^merge
      ^body:
        %step = spirv.IMul %offset, %two : i32
        %lane_plus_one = spirv.IAdd %tid, %one : i32
        %scaled = spirv.IMul %lane_plus_one, %step : i32
        %target = spirv.ISub %scaled, %one : i32
        %active = spirv.SLessThan %target, %c256 : i32
        spirv.mlir.selection {
          spirv.BranchConditional %active, ^then, ^merge
        ^then:
          %left_index = spirv.ISub %target, %offset : i32
          %left_ptr = spirv.AccessChain %scratch_ptr[%left_index] : !spirv.ptr<!spirv.array<256 x i32>, Workgroup>, i32 -> !spirv.ptr<i32, Workgroup>
          %target_ptr = spirv.AccessChain %scratch_ptr[%target] : !spirv.ptr<!spirv.array<256 x i32>, Workgroup>, i32 -> !spirv.ptr<i32, Workgroup>
          %left = spirv.Load "Workgroup" %left_ptr : i32
          %right = spirv.Load "Workgroup" %target_ptr : i32
          %sum = spirv.IAdd %left, %right : i32
          spirv.Store "Workgroup" %target_ptr, %sum : i32
          spirv.Branch ^merge
        ^merge:
          spirv.mlir.merge
        }
        spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
        %next = spirv.IMul %offset, %two : i32
        spirv.Branch ^header(%next : i32)
      ^merge:
        spirv.mlir.merge
      }

      %lane_zero = spirv.IEqual %tid, %zero : i32
      spirv.mlir.selection {
        spirv.BranchConditional %lane_zero, ^then, ^merge
      ^then:
        %last_ptr = spirv.AccessChain %scratch_ptr[%c255] : !spirv.ptr<!spirv.array<256 x i32>, Workgroup>, i32 -> !spirv.ptr<i32, Workgroup>
        %total = spirv.Load "Workgroup" %last_ptr : i32
        %sum_out = spirv.AccessChain %sums_ptr[%zero, %group] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
        spirv.Store "StorageBuffer" %sum_out, %total : i32
        spirv.Store "Workgroup" %last_ptr, %zero : i32
        spirv.Branch ^merge
      ^merge:
        spirv.mlir.merge
      }
      spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>

      spirv.mlir.loop {
        spirv.Branch ^header(%c128 : i32)
      ^header(%offset: i32):
        %continue = spirv.SLessThan %zero, %offset : i32
        spirv.BranchConditional %continue, ^body, ^merge
      ^body:
        %step = spirv.IMul %offset, %two : i32
        %lane_plus_one = spirv.IAdd %tid, %one : i32
        %scaled = spirv.IMul %lane_plus_one, %step : i32
        %target = spirv.ISub %scaled, %one : i32
        %active = spirv.SLessThan %target, %c256 : i32
        spirv.mlir.selection {
          spirv.BranchConditional %active, ^then, ^merge
        ^then:
          %left_index = spirv.ISub %target, %offset : i32
          %left_ptr = spirv.AccessChain %scratch_ptr[%left_index] : !spirv.ptr<!spirv.array<256 x i32>, Workgroup>, i32 -> !spirv.ptr<i32, Workgroup>
          %target_ptr = spirv.AccessChain %scratch_ptr[%target] : !spirv.ptr<!spirv.array<256 x i32>, Workgroup>, i32 -> !spirv.ptr<i32, Workgroup>
          %left = spirv.Load "Workgroup" %left_ptr : i32
          %right = spirv.Load "Workgroup" %target_ptr : i32
          spirv.Store "Workgroup" %left_ptr, %right : i32
          %sum = spirv.IAdd %left, %right : i32
          spirv.Store "Workgroup" %target_ptr, %sum : i32
          spirv.Branch ^merge
        ^merge:
          spirv.mlir.merge
        }
        spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
        %next = spirv.ShiftRightLogical %offset, %one : i32, i32
        spirv.Branch ^header(%next : i32)
      ^merge:
        spirv.mlir.merge
      }

      %scan0 = spirv.Load "Workgroup" %scratch0 : i32
      %scan1 = spirv.Load "Workgroup" %scratch1 : i32
      spirv.mlir.selection {
        spirv.BranchConditional %valid0, ^then, ^merge
      ^then:
        %dest0 = spirv.AccessChain %output_ptr[%zero, %index0] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
        spirv.Store "StorageBuffer" %dest0, %scan0 : i32
        spirv.Branch ^merge
      ^merge:
        spirv.mlir.merge
      }
      spirv.mlir.selection {
        spirv.BranchConditional %valid1, ^then, ^merge
      ^then:
        %dest1 = spirv.AccessChain %output_ptr[%zero, %index1] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
        spirv.Store "StorageBuffer" %dest1, %scan1 : i32
        spirv.Branch ^merge
      ^merge:
        spirv.mlir.merge
      }
      spirv.Return
    }
    spirv.EntryPoint "GLCompute" @main, @input, @output, @block_sums, @params, @local_id, @group_id, @scratch
    spirv.ExecutionMode @main "LocalSize", 128, 1, 1
  }
}
)mlir";
  return module;
}

const std::string& uniformAddModule() {
  static const std::string module = R"mlir(
module {
  spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
    spirv.GlobalVariable @data bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
    spirv.GlobalVariable @block_offsets bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
    spirv.GlobalVariable @params : !spirv.ptr<!spirv.struct<(i32 [0])>, PushConstant>
    spirv.GlobalVariable @global_id built_in("GlobalInvocationId") : !spirv.ptr<vector<3xi32>, Input>

    spirv.func @main() "None" {
      %zero = spirv.Constant 0 : i32
      %shift = spirv.Constant 8 : i32
      %global_ptr = spirv.mlir.addressof @global_id : !spirv.ptr<vector<3xi32>, Input>
      %global_vec = spirv.Load "Input" %global_ptr : vector<3xi32>
      %index = spirv.CompositeExtract %global_vec[0 : i32] : vector<3xi32>
      %params_ptr = spirv.mlir.addressof @params : !spirv.ptr<!spirv.struct<(i32 [0])>, PushConstant>
      %count_ptr = spirv.AccessChain %params_ptr[%zero] : !spirv.ptr<!spirv.struct<(i32 [0])>, PushConstant>, i32 -> !spirv.ptr<i32, PushConstant>
      %count = spirv.Load "PushConstant" %count_ptr : i32
      %valid = spirv.SLessThan %index, %count : i32

      spirv.mlir.selection {
        spirv.BranchConditional %valid, ^then, ^merge
      ^then:
        %data_ptr = spirv.mlir.addressof @data : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
        %offsets_ptr = spirv.mlir.addressof @block_offsets : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
        %block = spirv.ShiftRightLogical %index, %shift : i32, i32
        %value_ptr = spirv.AccessChain %data_ptr[%zero, %index] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
        %offset_ptr = spirv.AccessChain %offsets_ptr[%zero, %block] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
        %value = spirv.Load "StorageBuffer" %value_ptr : i32
        %offset = spirv.Load "StorageBuffer" %offset_ptr : i32
        %sum = spirv.IAdd %value, %offset : i32
        spirv.Store "StorageBuffer" %value_ptr, %sum : i32
        spirv.Branch ^merge
      ^merge:
        spirv.mlir.merge
      }
      spirv.Return
    }
    spirv.EntryPoint "GLCompute" @main, @data, @block_offsets, @params, @global_id
    spirv.ExecutionMode @main "LocalSize", 128, 1, 1
  }
}
)mlir";
  return module;
}

struct ScanDispatch {
  uint32_t count = 0;
  uint32_t groups = 0;
  sd::graph::VulkanAllocRecord input;
  sd::graph::VulkanAllocRecord output;
  sd::graph::VulkanAllocRecord blockSums;
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
};

struct UniformAddDispatch {
  uint32_t count = 0;
  uint32_t groups = 0;
  sd::graph::VulkanAllocRecord data;
  sd::graph::VulkanAllocRecord blockOffsets;
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
};

VkDeviceSize byteCount(uint32_t count) {
  return static_cast<VkDeviceSize>(count) * sizeof(int);
}

void releaseScratch(const std::vector<void*>& scratchPointers) {
  auto& memoryPool = sd::graph::VulkanMemoryPool::getInstance();
  for (void* pointer : scratchPointers) {
    if (pointer != nullptr) memoryPool.freeImmediate(pointer);
  }
}

void updateScanDescriptors(VkDevice device, const ScanDispatch& dispatch) {
  VkDescriptorBufferInfo buffers[3] = {
      {dispatch.input.buffer, 0, byteCount(dispatch.count)},
      {dispatch.output.buffer, 0, byteCount(dispatch.count)},
      {dispatch.blockSums.buffer, 0, byteCount(dispatch.groups)}};
  VkWriteDescriptorSet writes[3] = {};
  for (uint32_t i = 0; i < 3; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = dispatch.descriptorSet;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &buffers[i];
  }
  vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
}

void updateUniformDescriptors(VkDevice device,
                              const UniformAddDispatch& dispatch) {
  const uint32_t blockCount =
      (dispatch.count + kScanElementsPerBlock - 1) /
      kScanElementsPerBlock;
  VkDescriptorBufferInfo buffers[2] = {
      {dispatch.data.buffer, 0, byteCount(dispatch.count)},
      {dispatch.blockOffsets.buffer, 0, byteCount(blockCount)}};
  VkWriteDescriptorSet writes[2] = {};
  for (uint32_t i = 0; i < 2; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = dispatch.descriptorSet;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &buffers[i];
  }
  vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
}

void recordMemoryBarrier(VkCommandBuffer commandBuffer,
                         VkPipelineStageFlags sourceStage,
                         VkAccessFlags sourceAccess,
                         VkPipelineStageFlags destinationStage,
                         VkAccessFlags destinationAccess) {
  VkMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = sourceAccess;
  barrier.dstAccessMask = destinationAccess;
  vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 1,
                       &barrier, 0, nullptr, 0, nullptr);
}

#endif  // HAVE_MLIR

}  // namespace

void prescanArrayRecursive(sd::Pointer* extras, int* dZ, int* dX,
                           int numElements, int level) {
  (void)level;
  if (numElements < 0 ||
      (numElements > 0 && (dZ == nullptr || dX == nullptr))) {
    setPrescanError(
        sd::Status::BAD_INPUT,
        "Vulkan prescan received an invalid pointer or element count");
    return;
  }
  if (numElements == 0) return;

#if defined(HAVE_MLIR) && HAVE_MLIR
  auto& memoryPool = sd::graph::VulkanMemoryPool::getInstance();
  sd::graph::VulkanAllocRecord outputRecord;
  sd::graph::VulkanAllocRecord inputRecord;
  const VkDeviceSize bytes =
      static_cast<VkDeviceSize>(numElements) * sizeof(int);
  if (!memoryPool.queryRecord(dZ, outputRecord) ||
      !memoryPool.queryRecord(dX, inputRecord) ||
      outputRecord.deviceId != inputRecord.deviceId ||
      outputRecord.logicalDevice != inputRecord.logicalDevice ||
      outputRecord.logicalSize < bytes || inputRecord.logicalSize < bytes) {
    setPrescanError(
        sd::Status::BAD_INPUT,
        "Vulkan prescan requires same-device Vulkan int buffers of sufficient size");
    return;
  }

  std::string streamError;
  void* opaqueStream = extras == nullptr ? nullptr : extras[1];
  auto* stream =
      resolvePrescanStream(opaqueStream, outputRecord.deviceId, streamError);
  auto* deviceContext =
      sd::graph::VulkanDeviceContext::getContext(outputRecord.deviceId);
  if (stream == nullptr || deviceContext == nullptr ||
      deviceContext->descriptorPool() == VK_NULL_HANDLE ||
      deviceContext->shaderPipelineCache() == nullptr) {
    setPrescanError(
        sd::Status::KERNEL_FAILURE,
        streamError.empty() ? "Vulkan prescan device context is unavailable"
                            : streamError);
    return;
  }

  VkPhysicalDeviceProperties properties = {};
  vkGetPhysicalDeviceProperties(deviceContext->physicalDevice(), &properties);
  const auto& limits = properties.limits;
  if (limits.maxComputeWorkGroupInvocations < kScanThreads ||
      limits.maxComputeWorkGroupSize[0] < kScanThreads ||
      limits.maxComputeSharedMemorySize <
          kScanElementsPerBlock * sizeof(int) ||
      limits.maxPushConstantsSize < kPushConstantBytes ||
      bytes > limits.maxStorageBufferRange) {
    setPrescanError(
        sd::Status::KERNEL_FAILURE,
        "Vulkan device limits do not support the portable prescan kernel");
    return;
  }

  std::vector<void*> scratchPointers;
  std::vector<ScanDispatch> scans;
  sd::graph::VulkanAllocRecord stageInput = inputRecord;
  sd::graph::VulkanAllocRecord stageOutput = outputRecord;
  uint32_t stageCount = static_cast<uint32_t>(numElements);

  while (true) {
    const uint32_t groups = static_cast<uint32_t>(
        (static_cast<uint64_t>(stageCount) + kScanElementsPerBlock - 1) /
        kScanElementsPerBlock);
    if (groups == 0 || groups > limits.maxComputeWorkGroupCount[0]) {
      releaseScratch(scratchPointers);
      setPrescanError(sd::Status::KERNEL_FAILURE,
                      "Vulkan prescan dispatch exceeds the device grid limit");
      return;
    }

    const VkDeviceSize scratchBytes = byteCount(groups);
    if (scratchBytes > limits.maxStorageBufferRange) {
      releaseScratch(scratchPointers);
      setPrescanError(
          sd::Status::KERNEL_FAILURE,
          "Vulkan prescan hierarchy exceeds the storage-buffer range limit");
      return;
    }

    void* scratch =
        memoryPool.allocate(outputRecord.deviceId, scratchBytes);
    sd::graph::VulkanAllocRecord scratchRecord;
    if (scratch == nullptr ||
        !memoryPool.queryRecord(scratch, scratchRecord) ||
        scratchRecord.logicalDevice != outputRecord.logicalDevice ||
        scratchRecord.logicalSize < scratchBytes) {
      if (scratch != nullptr) memoryPool.freeImmediate(scratch);
      releaseScratch(scratchPointers);
      setPrescanError(sd::Status::KERNEL_FAILURE,
                      "Vulkan prescan scratch allocation failed");
      return;
    }
    scratchPointers.push_back(scratch);
    scans.push_back(
        {stageCount, groups, stageInput, stageOutput, scratchRecord,
         VK_NULL_HANDLE});

    if (groups == 1) break;
    stageCount = groups;
    stageInput = scratchRecord;
    stageOutput = scratchRecord;
  }

  std::vector<UniformAddDispatch> uniformAdds;
  for (size_t i = scans.size(); i > 1; --i) {
    const size_t target = i - 2;
    const uint32_t count = scans[target].count;
    const uint32_t groups = static_cast<uint32_t>(
        (static_cast<uint64_t>(count) + kScanThreads - 1) /
        kScanThreads);
    if (groups == 0 || groups > limits.maxComputeWorkGroupCount[0]) {
      releaseScratch(scratchPointers);
      setPrescanError(
          sd::Status::KERNEL_FAILURE,
          "Vulkan prescan uniform-add dispatch exceeds the device grid limit");
      return;
    }
    uniformAdds.push_back(
        {count, groups, scans[target].output, scans[target].blockSums,
         VK_NULL_HANDLE});
  }

  const std::string& scanModule = blockScanModule();
  const std::string& addModule = uniformAddModule();
  auto* pipelineCache = deviceContext->shaderPipelineCache();
  VkPipeline scanPipeline = pipelineCache->getOrCompile(
      kBlockScanPipelineKey, scanModule, deviceContext->device(),
      kPushConstantBytes);
  VkPipelineLayout scanPipelineLayout = pipelineCache->getPipelineLayout(
      kBlockScanPipelineKey, scanModule, kPushConstantBytes);
  VkDescriptorSetLayout scanDescriptorLayout =
      pipelineCache->getDescriptorSetLayout(
          kBlockScanPipelineKey, scanModule, kPushConstantBytes);
  const std::vector<uint32_t> scanBindings =
      pipelineCache->getDescriptorBindings(
          kBlockScanPipelineKey, scanModule, kPushConstantBytes);

  VkPipeline addPipeline = pipelineCache->getOrCompile(
      kUniformAddPipelineKey, addModule, deviceContext->device(),
      kPushConstantBytes);
  VkPipelineLayout addPipelineLayout = pipelineCache->getPipelineLayout(
      kUniformAddPipelineKey, addModule, kPushConstantBytes);
  VkDescriptorSetLayout addDescriptorLayout =
      pipelineCache->getDescriptorSetLayout(
          kUniformAddPipelineKey, addModule, kPushConstantBytes);
  const std::vector<uint32_t> addBindings =
      pipelineCache->getDescriptorBindings(
          kUniformAddPipelineKey, addModule, kPushConstantBytes);

  if (scanPipeline == VK_NULL_HANDLE ||
      scanPipelineLayout == VK_NULL_HANDLE ||
      scanDescriptorLayout == VK_NULL_HANDLE ||
      scanBindings != std::vector<uint32_t>({0, 1, 2}) ||
      addPipeline == VK_NULL_HANDLE ||
      addPipelineLayout == VK_NULL_HANDLE ||
      addDescriptorLayout == VK_NULL_HANDLE ||
      addBindings != std::vector<uint32_t>({0, 1})) {
    releaseScratch(scratchPointers);
    setPrescanError(sd::Status::KERNEL_FAILURE,
                    "Vulkan prescan SPIR-V pipeline compilation failed");
    return;
  }

  std::vector<VkDescriptorSetLayout> descriptorLayouts;
  descriptorLayouts.reserve(scans.size() + uniformAdds.size());
  descriptorLayouts.insert(descriptorLayouts.end(), scans.size(),
                           scanDescriptorLayout);
  descriptorLayouts.insert(descriptorLayouts.end(), uniformAdds.size(),
                           addDescriptorLayout);
  if (descriptorLayouts.size() >
      std::numeric_limits<uint32_t>::max()) {
    releaseScratch(scratchPointers);
    setPrescanError(sd::Status::KERNEL_FAILURE,
                    "Vulkan prescan descriptor hierarchy is too large");
    return;
  }

  std::vector<VkDescriptorSet> descriptorSets(
      descriptorLayouts.size(), VK_NULL_HANDLE);
  VkDescriptorSetAllocateInfo allocationInfo = {};
  allocationInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocationInfo.descriptorPool = deviceContext->descriptorPool();
  allocationInfo.descriptorSetCount =
      static_cast<uint32_t>(descriptorLayouts.size());
  allocationInfo.pSetLayouts = descriptorLayouts.data();
  if (deviceContext->allocateDescriptorSets(
          &allocationInfo, descriptorSets.data()) != VK_SUCCESS) {
    releaseScratch(scratchPointers);
    setPrescanError(sd::Status::KERNEL_FAILURE,
                    "Vulkan prescan descriptor allocation failed");
    return;
  }

  size_t descriptorIndex = 0;
  for (auto& scan : scans) {
    scan.descriptorSet = descriptorSets[descriptorIndex++];
    updateScanDescriptors(deviceContext->device(), scan);
  }
  for (auto& add : uniformAdds) {
    add.descriptorSet = descriptorSets[descriptorIndex++];
    updateUniformDescriptors(deviceContext->device(), add);
  }

  std::vector<std::function<void()>> cleanup;
  cleanup.emplace_back(
      [deviceContext, descriptorSets, scratchPointers]() {
        if (!descriptorSets.empty()) {
          deviceContext->freeDescriptorSets(
              static_cast<uint32_t>(descriptorSets.size()),
              descriptorSets.data());
        }
        releaseScratch(scratchPointers);
      });

  const uint64_t sequence = stream->enqueueCommands(
      [scanPipeline, scanPipelineLayout, addPipeline, addPipelineLayout,
       scans, uniformAdds](VkCommandBuffer commandBuffer) {
        recordMemoryBarrier(
            commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                VK_ACCESS_HOST_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

        for (const auto& scan : scans) {
          vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            scanPipeline);
          vkCmdBindDescriptorSets(
              commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
              scanPipelineLayout, 0, 1, &scan.descriptorSet, 0, nullptr);
          vkCmdPushConstants(commandBuffer, scanPipelineLayout,
                             VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             kPushConstantBytes, &scan.count);
          vkCmdDispatch(commandBuffer, scan.groups, 1, 1);
          recordMemoryBarrier(
              commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              VK_ACCESS_SHADER_WRITE_BIT,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        }

        for (size_t i = 0; i < uniformAdds.size(); ++i) {
          const auto& add = uniformAdds[i];
          vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            addPipeline);
          vkCmdBindDescriptorSets(
              commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
              addPipelineLayout, 0, 1, &add.descriptorSet, 0, nullptr);
          vkCmdPushConstants(commandBuffer, addPipelineLayout,
                             VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             kPushConstantBytes, &add.count);
          vkCmdDispatch(commandBuffer, add.groups, 1, 1);
          if (i + 1 < uniformAdds.size()) {
            recordMemoryBarrier(
                commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
          }
        }

        recordMemoryBarrier(
            commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                VK_ACCESS_HOST_READ_BIT);
        return true;
      },
      std::move(cleanup));

  if (sequence == 0) {
    setPrescanError(sd::Status::KERNEL_FAILURE,
                    "Vulkan prescan command submission failed");
    return;
  }
  sd::graph::VulkanPipelineCache::recordKernelLaunches(
      scans.size() + uniformAdds.size());
#else
  setPrescanError(
      sd::Status::KERNEL_FAILURE,
      "Vulkan prescan requires the universal MLIR/SPIR-V pipeline");
#endif
}


#endif  // SD_VULKAN && HAVE_VULKAN
