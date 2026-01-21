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

package org.nd4j.linalg.api.ops.impl.layers.convolution.config;

import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Configuration for Deformable Convolution 2D
 *
 * @author Eclipse Deeplearning4j
 */
@Data
@Builder
@NoArgsConstructor
public class DeformableConv2DConfig extends BaseConvolutionConfig {

    @Builder.Default
    private int kH = 1;  // kernel height
    @Builder.Default
    private int kW = 1;  // kernel width
    @Builder.Default
    private int sH = 1;  // stride height
    @Builder.Default
    private int sW = 1;  // stride width
    @Builder.Default
    private int pH = 0;  // padding height
    @Builder.Default
    private int pW = 0;  // padding width
    @Builder.Default
    private int dH = 1;  // dilation height
    @Builder.Default
    private int dW = 1;  // dilation width
    @Builder.Default
    private int groups = 1;  // convolution groups
    @Builder.Default
    private int deformableGroups = 1;  // deformable/offset groups
    @Builder.Default
    private boolean isNCHW = true;  // data format

    public DeformableConv2DConfig(int kH, int kW, int sH, int sW, int pH, int pW,
                                  int dH, int dW, int groups, int deformableGroups,
                                  boolean isNCHW) {
        this.kH = kH;
        this.kW = kW;
        this.sH = sH;
        this.sW = sW;
        this.pH = pH;
        this.pW = pW;
        this.dH = dH;
        this.dW = dW;
        this.groups = groups;
        this.deformableGroups = deformableGroups;
        this.isNCHW = isNCHW;
    }

    @Override
    public Map<String, Object> toProperties() {
        Map<String, Object> ret = new LinkedHashMap<>();
        ret.put("kH", kH);
        ret.put("kW", kW);
        ret.put("sH", sH);
        ret.put("sW", sW);
        ret.put("pH", pH);
        ret.put("pW", pW);
        ret.put("dH", dH);
        ret.put("dW", dW);
        ret.put("groups", groups);
        ret.put("deformableGroups", deformableGroups);
        ret.put("isNCHW", isNCHW);
        return ret;
    }

    @Override
    protected void validate() {
        // Validate kernel dimensions
        if (kH <= 0 || kW <= 0) {
            throw new IllegalStateException("Kernel dimensions must be positive: kH=" + kH + ", kW=" + kW);
        }
        // Validate stride dimensions
        if (sH <= 0 || sW <= 0) {
            throw new IllegalStateException("Stride dimensions must be positive: sH=" + sH + ", sW=" + sW);
        }
        // Validate dilation dimensions
        if (dH <= 0 || dW <= 0) {
            throw new IllegalStateException("Dilation dimensions must be positive: dH=" + dH + ", dW=" + dW);
        }
        // Validate padding (can be zero)
        if (pH < 0 || pW < 0) {
            throw new IllegalStateException("Padding cannot be negative: pH=" + pH + ", pW=" + pW);
        }
        // Validate groups
        if (groups <= 0) {
            throw new IllegalStateException("Groups must be positive: " + groups);
        }
        // Validate deformable groups
        if (deformableGroups <= 0) {
            throw new IllegalStateException("Deformable groups must be positive: " + deformableGroups);
        }
    }
}
