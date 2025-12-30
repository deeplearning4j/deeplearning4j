/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

// Created by Abdelrauf 2020

#ifndef DEV_TESTSARMCOMPUTEUTILS_H
#define DEV_TESTSARMCOMPUTEUTILS_H

#include <arm_compute/core/Helpers.h>
#include <arm_compute/core/ITensor.h>
#include <arm_compute/core/Strides.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/TensorShape.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/core/Validate.h>
#include <arm_compute/core/Window.h>
#include <arm_compute/runtime/NEON/NEFunctions.h>
#include <arm_compute/runtime/Tensor.h>
#include <arm_compute/runtime/TensorAllocator.h>
#include <array/NDArray.h>
#include <graph/Context.h>
#include <legacy/NativeOps.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include <iostream>

using namespace samediff;

#if 0
#define internal_printf(FORMAT, ...) sd_printf(FORMAT, __VA_ARGS__)
#define internal_print_arm_array(a, b) print_tensor(a, b)
#define internal_print_nd_array(a, b) ((a).printIndexedBuffer(b))
#define internal_print_nd_shape(a, b) ((a).printShapeInfo(b))
#else
#define internal_printf(FORMAT, ...)
#define internal_print_arm_array(a, b)
#define internal_print_nd_array(a, b)
#define internal_print_nd_shape(a, b)
#endif

namespace sd {
namespace ops {
namespace platforms {

using Arm_DataType = arm_compute::DataType;
using Arm_Tensor = arm_compute::Tensor;
using Arm_ITensor = arm_compute::ITensor;
using Arm_TensorInfo = arm_compute::TensorInfo;
using Arm_TensorShape = arm_compute::TensorShape;
using Arm_Strides = arm_compute::Strides;
using Arm_WeightsInfo = arm_compute::WeightsInfo;
using Arm_PermutationVector = arm_compute::PermutationVector;
using Arm_DataLayout = arm_compute::DataLayout;

/**
 * Here we actually declare our platform helpers
 */
DECLARE_PLATFORM(maxpool2d, ENGINE_CPU);

DECLARE_PLATFORM(avgpool2d, ENGINE_CPU);

DECLARE_PLATFORM(conv2d, ENGINE_CPU);

DECLARE_PLATFORM(deconv2d, ENGINE_CPU);

// Depthwise convolution
DECLARE_PLATFORM(depthwise_conv2d, ENGINE_CPU);

// Activation functions
DECLARE_PLATFORM(relu, ENGINE_CPU);
DECLARE_PLATFORM(sigmoid, ENGINE_CPU);
DECLARE_PLATFORM(tanh, ENGINE_CPU);
DECLARE_PLATFORM(softmax, ENGINE_CPU);
DECLARE_PLATFORM(elu, ENGINE_CPU);
DECLARE_PLATFORM(lrelu, ENGINE_CPU);
DECLARE_PLATFORM(relu6, ENGINE_CPU);
DECLARE_PLATFORM(abs, ENGINE_CPU);
DECLARE_PLATFORM(softplus, ENGINE_CPU);
DECLARE_PLATFORM(hardsigmoid, ENGINE_CPU);
DECLARE_PLATFORM(prelu, ENGINE_CPU);
DECLARE_PLATFORM(log_softmax, ENGINE_CPU);

// Normalization
DECLARE_PLATFORM(batchnorm, ENGINE_CPU);
DECLARE_PLATFORM(lrn, ENGINE_CPU);

// Fully connected / Linear
DECLARE_PLATFORM(xw_plus_b, ENGINE_CPU);

// Matrix operations
DECLARE_PLATFORM(matmul, ENGINE_CPU);

// Concatenation
DECLARE_PLATFORM(concat, ENGINE_CPU);

// Reduce operations
DECLARE_PLATFORM(reduce_mean, ENGINE_CPU);
DECLARE_PLATFORM(reduce_sum, ENGINE_CPU);

// Transpose/Permute
DECLARE_PLATFORM(transpose, ENGINE_CPU);

// Elementwise operations
DECLARE_PLATFORM(add, ENGINE_CPU);
DECLARE_PLATFORM(subtract, ENGINE_CPU);
DECLARE_PLATFORM(multiply, ENGINE_CPU);
DECLARE_PLATFORM(divide, ENGINE_CPU);
DECLARE_PLATFORM(maximum, ENGINE_CPU);
DECLARE_PLATFORM(minimum, ENGINE_CPU);

// Math operations
DECLARE_PLATFORM(exp, ENGINE_CPU);
DECLARE_PLATFORM(log, ENGINE_CPU);
DECLARE_PLATFORM(sqrt, ENGINE_CPU);
DECLARE_PLATFORM(neg, ENGINE_CPU);
DECLARE_PLATFORM(floor, ENGINE_CPU);
DECLARE_PLATFORM(ceil, ENGINE_CPU);
DECLARE_PLATFORM(round, ENGINE_CPU);
DECLARE_PLATFORM(rsqrt, ENGINE_CPU);

// Argmax/Argmin
DECLARE_PLATFORM(argmax, ENGINE_CPU);
DECLARE_PLATFORM(argmin, ENGINE_CPU);

// Pad
DECLARE_PLATFORM(pad, ENGINE_CPU);

// Split
DECLARE_PLATFORM(split, ENGINE_CPU);

// Tile
DECLARE_PLATFORM(tile, ENGINE_CPU);

// Gather
DECLARE_PLATFORM(gather, ENGINE_CPU);

// Sign
DECLARE_PLATFORM(sign, ENGINE_CPU);

// Squeeze/Reshape
DECLARE_PLATFORM(squeeze, ENGINE_CPU);

// Stack
DECLARE_PLATFORM(stack, ENGINE_CPU);

// Reverse
DECLARE_PLATFORM(reverse, ENGINE_CPU);

// Additional reduction operations
DECLARE_PLATFORM(reduce_max, ENGINE_CPU);
DECLARE_PLATFORM(reduce_min, ENGINE_CPU);
DECLARE_PLATFORM(reduce_prod, ENGINE_CPU);
DECLARE_PLATFORM(mean, ENGINE_CPU);

// Power operations
DECLARE_PLATFORM(pow, ENGINE_CPU);
DECLARE_PLATFORM(square, ENGINE_CPU);
DECLARE_PLATFORM(reciprocal, ENGINE_CPU);

// Comparison operations
DECLARE_PLATFORM(greater, ENGINE_CPU);
DECLARE_PLATFORM(less, ENGINE_CPU);
DECLARE_PLATFORM(equals, ENGINE_CPU);
DECLARE_PLATFORM(greater_equal, ENGINE_CPU);
DECLARE_PLATFORM(less_equal, ENGINE_CPU);
DECLARE_PLATFORM(not_equals, ENGINE_CPU);

// Shape operations
DECLARE_PLATFORM(reshape, ENGINE_CPU);
DECLARE_PLATFORM(flatten, ENGINE_CPU);
DECLARE_PLATFORM(slice, ENGINE_CPU);
DECLARE_PLATFORM(expand_dims, ENGINE_CPU);

// Clip operation
DECLARE_PLATFORM(clip_by_value, ENGINE_CPU);

// Trigonometric operations
DECLARE_PLATFORM(sin, ENGINE_CPU);
DECLARE_PLATFORM(cos, ENGINE_CPU);

// Normalization
DECLARE_PLATFORM(l2_normalize, ENGINE_CPU);

// Copy operation
DECLARE_PLATFORM(copy, ENGINE_CPU);

// Modern activations
DECLARE_PLATFORM(silu, ENGINE_CPU);
DECLARE_PLATFORM(gelu, ENGINE_CPU);
DECLARE_PLATFORM(hardswish, ENGINE_CPU);

// Resize/Scale
DECLARE_PLATFORM(resize_bilinear, ENGINE_CPU);
DECLARE_PLATFORM(resize_nearest_neighbor, ENGINE_CPU);
DECLARE_PLATFORM(crop_and_resize, ENGINE_CPU);

// Logical operations
DECLARE_PLATFORM(boolean_and, ENGINE_CPU);
DECLARE_PLATFORM(boolean_or, ENGINE_CPU);
DECLARE_PLATFORM(boolean_not, ENGINE_CPU);

// Depth/Space operations
DECLARE_PLATFORM(depth_to_space, ENGINE_CPU);
DECLARE_PLATFORM(space_to_depth, ENGINE_CPU);
DECLARE_PLATFORM(batch_to_space, ENGINE_CPU);
DECLARE_PLATFORM(space_to_batch, ENGINE_CPU);

// Shape operations
DECLARE_PLATFORM(unstack, ENGINE_CPU);
DECLARE_PLATFORM(strided_slice, ENGINE_CPU);

// Utility operations
DECLARE_PLATFORM(fill, ENGINE_CPU);

// Statistical operations
DECLARE_PLATFORM(moments, ENGINE_CPU);
DECLARE_PLATFORM(top_k, ENGINE_CPU);

// Normalization
DECLARE_PLATFORM(instance_normalization, ENGINE_CPU);

// More activations
DECLARE_PLATFORM(mish, ENGINE_CPU);
DECLARE_PLATFORM(leaky_relu, ENGINE_CPU);

// Linear/GEMM operations
DECLARE_PLATFORM(linear, ENGINE_CPU);
DECLARE_PLATFORM(gemm, ENGINE_CPU);

// utils
Arm_DataType getArmType(const sd::DataType& dType);

Arm_TensorInfo getArmTensorInfo(int rank, sd::LongType* bases, sd::DataType ndArrayType,
                                Arm_DataLayout layout = Arm_DataLayout::UNKNOWN);

Arm_TensorInfo getArmTensorInfo(NDArray& arr, Arm_DataLayout layout = Arm_DataLayout::UNKNOWN);

Arm_Tensor getArmTensor(NDArray& arr, Arm_DataLayout layout = Arm_DataLayout::UNKNOWN);

void copyFromTensor(const Arm_Tensor& inTensor, NDArray& output);
void copyToTensor(NDArray& input, Arm_Tensor& outTensor);
void print_tensor(Arm_ITensor& tensor, const char* msg);
bool isArmcomputeFriendly(NDArray& arr);

template <typename F>
class ArmFunction {
 public:
  template <typename... Args>
  void configure(NDArray* input, NDArray* output, Arm_DataLayout layout, Args&&... args) {
    bool inputHasPaddedBuffer = input->hasPaddedBuffer();
    bool outputHasPaddedBuffer = output->hasPaddedBuffer();
    if (inputHasPaddedBuffer) {
      in = getArmTensor(*input, layout);
      internal_printf("input is a padded buffer %d\n", 0);
    } else {
      auto inInfo = getArmTensorInfo(*input, layout);
      in.allocator()->init(inInfo);
    }
    if (outputHasPaddedBuffer) {
      out = getArmTensor(*output, layout);
      internal_printf("output is a padded buffer %d\n", 0);
    } else {
      auto outInfo = getArmTensorInfo(*output, layout);
      out.allocator()->init(outInfo);
    }
    armFunction.configure(&in, &out, std::forward<Args>(args)...);
    if (!inputHasPaddedBuffer) {
      if (in.info()->has_padding()) {
        // allocate and copy
        in.allocator()->allocate();
        inputNd = input;
      } else {
        // import only for ews()==1
        in.allocator()->import_memory(input->buffer());
        internal_printf("input import %d\n", 0);
      }
    }
    if (!outputHasPaddedBuffer) {
      if (out.info()->has_padding()) {
        // store pointer to our array to copy after run
        out.allocator()->allocate();
        outNd = output;
      } else {
        // import only for ews()==1
        out.allocator()->import_memory(output->buffer());
      }
    }
  }
  void run() {
    if (inputNd) {
      // copy
      copyToTensor(*inputNd, in);
    }
    armFunction.run();
    if (outNd) {
      copyFromTensor(out, *outNd);
      internal_printf("output copy %d\n", 0);
      internal_print_arm_array(out, "out");
    }
  }

 private:
  Arm_Tensor in;
  Arm_Tensor out;
  NDArray* inputNd = nullptr;
  NDArray* outNd = nullptr;
  F armFunction{};
};

template <typename F>
class ArmFunctionWeighted {
 public:
  template <typename... Args>
  void configure(NDArray* input, NDArray* weights, NDArray* biases, NDArray* output, Arm_DataLayout layout,
                 arm_compute::PermutationVector permuteVector, Args&&... args) {
    bool inputHasPaddedBuffer = input->hasPaddedBuffer();
    bool weightsHasPaddedBuffer = weights->hasPaddedBuffer();
    bool outputHasPaddedBuffer = output->hasPaddedBuffer();
    bool biasesHasPaddedBuffer = false;
    if (inputHasPaddedBuffer) {
      in = getArmTensor(*input, layout);
    } else {
      in.allocator()->init(getArmTensorInfo(*input, layout));
    }
    if (weightsHasPaddedBuffer) {
      w = getArmTensor(*weights, layout);
    } else {
      w.allocator()->init(getArmTensorInfo(*weights, layout));
    }
    if (outputHasPaddedBuffer) {
      out = getArmTensor(*output, layout);
    } else {
      out.allocator()->init(getArmTensorInfo(*output, layout));
    }
    Arm_Tensor* bias_ptr = nullptr;
    if (biases) {
      biasesHasPaddedBuffer = biases->hasPaddedBuffer();
      if (biasesHasPaddedBuffer) {
        b = getArmTensor(*biases, layout);
      } else {
        b.allocator()->init(getArmTensorInfo(*biases, layout));
      }
      bias_ptr = &b;
    }
    if (permuteVector.num_dimensions() == 0) {
      armFunction.configure(&in, &w, bias_ptr, &out, std::forward<Args>(args)...);
    } else {
      // configure with permute kernel
      Arm_TensorShape shape;
      int rank = permuteVector.num_dimensions();
      shape.set_num_dimensions(rank);
      auto wInfoPtr = w.info();
      for (int i = 0; i < rank; i++) {
        shape[i] = wInfoPtr->dimension(permuteVector[i]);
      }
      for (int i = rank; i < arm_compute::MAX_DIMS; i++) {
        shape[i] = 1;
      }
      Arm_TensorInfo wPermInfo(shape, 1, wInfoPtr->data_type(), layout);
      wPerm.allocator()->init(wPermInfo);
      permuter.configure(&w, &wPerm, permuteVector);
      armFunction.configure(&in, &wPerm, bias_ptr, &out, std::forward<Args>(args)...);
      wPerm.allocator()->allocate();
      runPerm = true;
    }
    // import buffer
    if (!inputHasPaddedBuffer) {
      if (in.info()->has_padding()) {
        // allocate and copy
        in.allocator()->allocate();
        inputNd = input;
      } else {
        // import buffer
        in.allocator()->import_memory(input->buffer());
      }
    }
    if (!weightsHasPaddedBuffer) {
      if (w.info()->has_padding()) {
        // store pointer to our array to copy after run
        w.allocator()->allocate();
        wNd = weights;
      } else {
        // import
        w.allocator()->import_memory(weights->buffer());
      }
    }
    if (biases && !biasesHasPaddedBuffer) {
      if (b.info()->has_padding()) {
        // store pointer to our array to copy after run
        b.allocator()->allocate();
        bNd = biases;
      } else {
        // import
        b.allocator()->import_memory(biases->buffer());
      }
    }
    if (!outputHasPaddedBuffer) {
      if (out.info()->has_padding()) {
        // store pointer to our array to copy after run
        out.allocator()->allocate();
        outNd = output;
      } else {
        // import
        out.allocator()->import_memory(output->buffer());
      }
    }
  }
  void run() {
    if (inputNd) {
      // copy
      copyToTensor(*inputNd, in);
    }
    if (bNd) {
      // copy
      copyToTensor(*bNd, b);
    }
    if (wNd) {
      // copy
      copyToTensor(*wNd, w);
    }
    if (runPerm) {
      permuter.run();
    }
    armFunction.run();
    if (outNd) {
      copyFromTensor(out, *outNd);
    }
  }

 private:
  bool runPerm = false;
  Arm_Tensor in;
  Arm_Tensor b;
  Arm_Tensor w;
  Arm_Tensor wPerm;
  Arm_Tensor out;
  NDArray* inputNd = nullptr;
  NDArray* wNd = nullptr;
  NDArray* bNd = nullptr;
  NDArray* outNd = nullptr;
  arm_compute::NEPermute permuter;
  F armFunction{};
};

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // DEV_TESTSARMCOMPUTEUTILS_H
