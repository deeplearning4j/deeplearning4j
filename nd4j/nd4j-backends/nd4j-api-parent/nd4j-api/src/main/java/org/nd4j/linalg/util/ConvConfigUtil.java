/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.util;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.nd4j.common.base.Preconditions;

/**
 * Class with utility methods for validating convolution op configurations like {@link org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig}
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class ConvConfigUtil {

    /**
     * Validate a 2D convolution's Kernel, Stride, Padding, and Dilation
     */
    public static void validate2D(long kH, long kW, long sH, long sW, long pH, long pW, long dH, long dW){
        Preconditions.checkArgument(kH != 0, "Kernel height can not be 0");
        Preconditions.checkArgument(kW != 0, "Kernel width can not be 0");

        Preconditions.checkArgument(sH > 0, "Stride height can not be negative or 0, got: %s", sH);
        Preconditions.checkArgument(sW > 0, "Stride width can not be negative or 0, got: %s", sW);

        Preconditions.checkArgument(pH >= 0, "Padding height can not be negative, got: %s", pH);
        Preconditions.checkArgument(pW >= 0, "Padding width can not be negative, got: %s", pW);

        Preconditions.checkArgument(dH > 0, "Dilation height can not be negative or 0, got: %s", dH);
        Preconditions.checkArgument(dW > 0, "Dilation width can not be negative or 0, got: %s", dW);
    }

    /**
     * Validate a 3D convolution's Kernel, Stride, Padding, and Dilation
     */
    public static void validate3D(long kH, long kW, long kD, long sH, long sW, long sD, long pH, long pW, long pD, long dH, long dW, long dD){
        Preconditions.checkArgument(kH != 0, "Kernel height can not be 0");
        Preconditions.checkArgument(kW != 0, "Kernel width can not be 0");
        Preconditions.checkArgument(kD != 0, "Kernel depth can not be 0");

        Preconditions.checkArgument(sH > 0, "Stride height can not be negative or 0, got: %s", sH);
        Preconditions.checkArgument(sW > 0, "Stride width can not be negative or 0, got: %s", sW);
        Preconditions.checkArgument(sD > 0, "Stride depth can not be negative or 0, got: %s", sD);

        Preconditions.checkArgument(pH >= 0, "Padding height can not be negative, got: %s", pH);
        Preconditions.checkArgument(pW >= 0, "Padding width can not be negative, got: %s", pW);
        Preconditions.checkArgument(pD >= 0, "Padding depth can not be negative, got: %s", pD);

        Preconditions.checkArgument(dH > 0, "Dilation height can not be negative or 0, got: %s", dH);
        Preconditions.checkArgument(dW > 0, "Dilation width can not be negative or 0, got: %s", dW);
        Preconditions.checkArgument(dD > 0, "Dilation depth can not be negative or 0, got: %s", dD);
    }

    /**
     * Validate a 3D convolution's Output Padding
     */
    public static void validateExtra3D(long aH, long aW, long aD){
        Preconditions.checkArgument(aH >= 0, "Output padding height can not be negative, got: %s", aH);
        Preconditions.checkArgument(aW >= 0, "Output padding width can not be negative, got: %s", aW);
        Preconditions.checkArgument(aD >= 0, "Output padding depth can not be negative, got: %s", aD);
    }

    /**
     * Validate a 1D convolution's Kernel, Stride, and Padding
     */
    public static void validate1D(long k, long s, long p, long d){
        Preconditions.checkArgument(k != 0, "Kernel can not be 0");

        Preconditions.checkArgument(s > 0, "Stride can not be negative or 0, got: %s", s);

        Preconditions.checkArgument(d > 0, "Dilation can not be negative or 0, got: %s", s);

        Preconditions.checkArgument(p >= 0, "Padding can not be negative, got: %s", p);
    }

    /**
     * Validate a LocalResponseNormalizationConfig
     */
    public static void validateLRN(double alpha, double beta, double bias, int depth) {
        Preconditions.checkArgument(depth > 0, "Depth can not be 0 or negative, got: %s", depth);
    }
}
