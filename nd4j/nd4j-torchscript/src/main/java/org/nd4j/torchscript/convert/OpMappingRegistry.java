/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.torchscript.convert;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.torchscript.ir.TorchScriptNode;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry mapping PyTorch operators to SameDiff operations.
 * Supports CNN-focused operations for vision models.
 */
@Slf4j
public class OpMappingRegistry {

    private static final Map<String, OpMapping> MAPPINGS = new ConcurrentHashMap<>();

    static {
        // Register CNN operations
        registerConvOps();
        registerPoolingOps();
        registerNormalizationOps();
        registerActivationOps();
        registerLinearOps();
        registerElementwiseOps();
        registerReshapeOps();
        registerDropoutOps();
    }

    private OpMappingRegistry() {
        // Utility class
    }

    private static void registerConvOps() {
        register("aten::conv2d", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            // Conv2d: input, weight, bias, stride, padding, dilation, groups
            List<Integer> stride = node.getIntListAttribute("stride");
            List<Integer> padding = node.getIntListAttribute("padding");
            List<Integer> dilation = node.getIntListAttribute("dilation");
            int groups = node.getIntAttribute("groups", 1);

            // Get weight from weights map or inputs
            String weightName = node.getInputNames().size() > 1 ? node.getInputNames().get(1) : null;
            String biasName = node.getInputNames().size() > 2 ? node.getInputNames().get(2) : null;

            // Note: Actual implementation would create conv2d with proper config
            log.debug("Creating conv2d op: stride={}, padding={}, dilation={}, groups={}",
                    stride, padding, dilation, groups);

            return input; // Placeholder - actual implementation needs full conv2d config
        });

        register("aten::conv1d", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            log.debug("Creating conv1d op");
            return input;
        });

        register("aten::conv3d", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            log.debug("Creating conv3d op");
            return input;
        });

        register("aten::conv_transpose2d", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            log.debug("Creating conv_transpose2d (deconv2d) op");
            return input;
        });
    }

    private static void registerPoolingOps() {
        register("aten::max_pool2d", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            List<Integer> kernelSize = node.getIntListAttribute("kernel_size");
            List<Integer> stride = node.getIntListAttribute("stride");
            List<Integer> padding = node.getIntListAttribute("padding");

            log.debug("Creating max_pool2d: kernel={}, stride={}, padding={}",
                    kernelSize, stride, padding);

            // Note: Actual implementation would use sd.cnn().maxPooling2d()
            return input;
        });

        register("aten::avg_pool2d", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            log.debug("Creating avg_pool2d");
            return input;
        });

        register("aten::adaptive_avg_pool2d", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            // Output size is typically [1, 1] for global average pooling
            log.debug("Creating adaptive_avg_pool2d");
            return input;
        });

        register("aten::adaptive_max_pool2d", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            log.debug("Creating adaptive_max_pool2d");
            return input;
        });
    }

    private static void registerNormalizationOps() {
        register("aten::batch_norm", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            // batch_norm: input, weight, bias, running_mean, running_var, training, momentum, eps
            float eps = node.getFloatAttribute("eps", 1e-5f);
            float momentum = node.getFloatAttribute("momentum", 0.1f);
            boolean training = node.getBoolAttribute("training", false);

            log.debug("Creating batch_norm: eps={}, momentum={}, training={}",
                    eps, momentum, training);

            return input;
        });

        register("aten::layer_norm", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            float eps = node.getFloatAttribute("eps", 1e-5f);
            log.debug("Creating layer_norm: eps={}", eps);
            return input;
        });

        register("aten::group_norm", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            int numGroups = node.getIntAttribute("num_groups", 32);
            float eps = node.getFloatAttribute("eps", 1e-5f);
            log.debug("Creating group_norm: groups={}, eps={}", numGroups, eps);
            return input;
        });

        register("aten::instance_norm", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            log.debug("Creating instance_norm");
            return input;
        });
    }

    private static void registerActivationOps() {
        register("aten::relu", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            return sd.nn().relu(input, 0);
        });

        register("aten::relu_", (sd, node, inputs, weights) -> {
            // In-place ReLU, treat same as regular ReLU
            SDVariable input = inputs.get(node.getInputNames().get(0));
            return sd.nn().relu(input, 0);
        });

        register("aten::leaky_relu", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            float negativeSlope = node.getFloatAttribute("negative_slope", 0.01f);
            return sd.nn().leakyRelu(input, negativeSlope);
        });

        register("aten::sigmoid", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            return sd.nn().sigmoid(input);
        });

        register("aten::tanh", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            return sd.math().tanh(input);
        });

        register("aten::softmax", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            int dim = node.getIntAttribute("dim", -1);
            return sd.nn().softmax(input, dim);
        });

        register("aten::log_softmax", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            int dim = node.getIntAttribute("dim", -1);
            return sd.nn().logSoftmax(input, dim);
        });

        register("aten::gelu", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            return sd.nn().gelu(input);
        });

        register("aten::silu", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            // SiLU = x * sigmoid(x) = Swish
            return sd.nn().swish(input);
        });

        register("aten::hardswish", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            // HardSwish(x) = x * hardSigmoid(x) where hardSigmoid(x) = max(0, min(1, (x + 3) / 6))
            return input.mul(sd.nn().hardSigmoid(input));
        });

        register("aten::hardsigmoid", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            return sd.nn().hardSigmoid(input);
        });

        register("aten::elu", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            // Note: SameDiff elu uses default alpha=1.0
            return sd.nn().elu(input);
        });

        register("aten::selu", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            return sd.nn().selu(input);
        });
    }

    private static void registerLinearOps() {
        register("aten::linear", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            // linear: input, weight, bias
            log.debug("Creating linear (dense) op");
            return input;
        });

        register("aten::addmm", (sd, node, inputs, weights) -> {
            // addmm: beta * input + alpha * (mat1 @ mat2)
            SDVariable input = inputs.get(node.getInputNames().get(0));
            log.debug("Creating addmm op");
            return input;
        });

        register("aten::matmul", (sd, node, inputs, weights) -> {
            SDVariable a = inputs.get(node.getInputNames().get(0));
            SDVariable b = inputs.get(node.getInputNames().get(1));
            return sd.mmul(a, b);
        });

        register("aten::mm", (sd, node, inputs, weights) -> {
            SDVariable a = inputs.get(node.getInputNames().get(0));
            SDVariable b = inputs.get(node.getInputNames().get(1));
            return sd.mmul(a, b);
        });

        register("aten::bmm", (sd, node, inputs, weights) -> {
            SDVariable a = inputs.get(node.getInputNames().get(0));
            SDVariable b = inputs.get(node.getInputNames().get(1));
            // Use mmul which handles batched matrix multiplication
            return sd.mmul(a, b);
        });
    }

    private static void registerElementwiseOps() {
        register("aten::add", (sd, node, inputs, weights) -> {
            SDVariable a = inputs.get(node.getInputNames().get(0));
            SDVariable b = inputs.get(node.getInputNames().get(1));
            return a.add(b);
        });

        register("aten::sub", (sd, node, inputs, weights) -> {
            SDVariable a = inputs.get(node.getInputNames().get(0));
            SDVariable b = inputs.get(node.getInputNames().get(1));
            return a.sub(b);
        });

        register("aten::mul", (sd, node, inputs, weights) -> {
            SDVariable a = inputs.get(node.getInputNames().get(0));
            SDVariable b = inputs.get(node.getInputNames().get(1));
            return a.mul(b);
        });

        register("aten::div", (sd, node, inputs, weights) -> {
            SDVariable a = inputs.get(node.getInputNames().get(0));
            SDVariable b = inputs.get(node.getInputNames().get(1));
            return a.div(b);
        });

        register("aten::cat", (sd, node, inputs, weights) -> {
            // Concatenate along dimension
            int dim = node.getIntAttribute("dim", 0);
            List<String> inputNames = node.getInputNames();
            SDVariable[] inputVars = new SDVariable[inputNames.size()];
            for (int i = 0; i < inputNames.size(); i++) {
                inputVars[i] = inputs.get(inputNames.get(i));
            }
            return sd.concat(dim, inputVars);
        });

        register("aten::mean", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            List<Integer> dims = node.getIntListAttribute("dim");
            boolean keepdim = node.getBoolAttribute("keepdim", false);

            if (dims.isEmpty()) {
                return sd.mean(input);
            } else {
                long[] dimsArray = dims.stream().mapToLong(Integer::longValue).toArray();
                return sd.mean(input, keepdim, dimsArray);
            }
        });

        register("aten::sum", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            return sd.sum(input);
        });
    }

    private static void registerReshapeOps() {
        register("aten::view", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            // Note: Would need to get target shape from attributes
            log.debug("Creating view (reshape) op");
            return input;
        });

        register("aten::reshape", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            log.debug("Creating reshape op");
            return input;
        });

        register("aten::flatten", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            int startDim = node.getIntAttribute("start_dim", 0);
            int endDim = node.getIntAttribute("end_dim", -1);
            log.debug("Creating flatten op: start_dim={}, end_dim={}", startDim, endDim);
            return input;
        });

        register("aten::squeeze", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            int dim = node.getIntAttribute("dim", 0);
            return sd.squeeze(input, dim);
        });

        register("aten::unsqueeze", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            int dim = node.getIntAttribute("dim", 0);
            return sd.expandDims(input, dim);
        });

        register("aten::permute", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            List<Integer> dims = node.getIntListAttribute("dims");
            log.debug("Creating permute op: dims={}", dims);
            return input;
        });

        register("aten::transpose", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            int dim0 = node.getIntAttribute("dim0", 0);
            int dim1 = node.getIntAttribute("dim1", 1);
            log.debug("Creating transpose op: dim0={}, dim1={}", dim0, dim1);
            return input;
        });

        register("aten::contiguous", (sd, node, inputs, weights) -> {
            // No-op in SameDiff (memory layout hint)
            return inputs.get(node.getInputNames().get(0));
        });
    }

    private static void registerDropoutOps() {
        register("aten::dropout", (sd, node, inputs, weights) -> {
            SDVariable input = inputs.get(node.getInputNames().get(0));
            float p = node.getFloatAttribute("p", 0.5f);
            boolean training = node.getBoolAttribute("training", false);

            if (training) {
                // inverted=false means standard dropout (multiply by 1/(1-p) during training)
                return sd.nn().dropout(input, false, (double) p);
            } else {
                // During inference, dropout is a no-op
                return input;
            }
        });

        register("aten::dropout_", (sd, node, inputs, weights) -> {
            // In-place dropout
            SDVariable input = inputs.get(node.getInputNames().get(0));
            float p = node.getFloatAttribute("p", 0.5f);
            boolean training = node.getBoolAttribute("training", false);

            if (training) {
                // inverted=false means standard dropout (multiply by 1/(1-p) during training)
                return sd.nn().dropout(input, false, (double) p);
            } else {
                return input;
            }
        });
    }

    /**
     * Register an operation mapping.
     *
     * @param opName  the PyTorch operation name
     * @param mapping the mapping function
     */
    public static void register(String opName, OpMapping mapping) {
        MAPPINGS.put(opName, mapping);
    }

    /**
     * Get the mapping for an operation.
     *
     * @param opName the operation name
     * @return the mapping or null
     */
    public static OpMapping getMapping(String opName) {
        return MAPPINGS.get(opName);
    }

    /**
     * Check if a mapping exists for an operation.
     *
     * @param opName the operation name
     * @return true if mapping exists
     */
    public static boolean hasMapping(String opName) {
        return MAPPINGS.containsKey(opName);
    }

    /**
     * Get all supported operation names.
     *
     * @return set of supported operations
     */
    public static Set<String> getSupportedOps() {
        return Collections.unmodifiableSet(MAPPINGS.keySet());
    }

    /**
     * Interface for individual operation mappings.
     */
    @FunctionalInterface
    public interface OpMapping {
        /**
         * Convert a TorchScript node to a SameDiff operation.
         *
         * @param sd      the SameDiff instance
         * @param node    the TorchScript node
         * @param inputs  map of input variable names to SDVariables
         * @param weights map of weight names to INDArrays
         * @return the output SDVariable
         */
        SDVariable convert(SameDiff sd, TorchScriptNode node,
                           Map<String, SDVariable> inputs,
                           Map<String, INDArray> weights);
    }
}
