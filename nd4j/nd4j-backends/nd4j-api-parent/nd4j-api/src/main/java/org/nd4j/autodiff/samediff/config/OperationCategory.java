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

package org.nd4j.autodiff.samediff.config;

/**
 * Categories of operations for bulk kernel configuration.
 *
 * Each category owns a set of glob-style name patterns that are matched against
 * registered op names inside {@link KernelConfiguration}'s category-based methods
 * (e.g. {@code forConvolutions()}, {@code forMatMul()}).
 *
 * @see KernelConfiguration
 * @see CategoryConfiguration
 */
public enum OperationCategory {

    CONVOLUTIONS("conv*", "depthwise*", "separable*"),
    POOLING("*pool*", "max_pool*", "avg_pool*"),
    NORMALIZATION("*norm*", "batch_norm*", "layer_norm*"),
    ACTIVATIONS("relu*", "sigmoid", "tanh", "softmax*", "gelu*"),
    LINEAR_ALGEBRA("matmul", "gemm", "dot", "tensordot"),
    ELEMENT_WISE("add", "subtract", "multiply", "divide", "pow"),
    REDUCTION("reduce_*", "sum", "mean", "max", "min"),
    ATTENTION("attention*", "dot_product_attention*", "multi_head_attention*"),
    RECURRENT("lstm*", "gru*", "rnn*");

    private final String[] patterns;

    OperationCategory(String... patterns) {
        this.patterns = patterns;
    }

    /**
     * Returns the glob-style op-name patterns that belong to this category.
     */
    public String[] getPatterns() {
        return patterns;
    }
}
