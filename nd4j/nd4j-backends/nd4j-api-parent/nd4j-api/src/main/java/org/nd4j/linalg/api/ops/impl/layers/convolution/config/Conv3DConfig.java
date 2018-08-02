/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.api.ops.impl.layers.convolution.config;


import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.LinkedHashMap;
import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Conv3DConfig extends BaseConvolutionConfig {
    //kernel
    @Builder.Default
    private long kD = 1;
    @Builder.Default
    private long kW = 1;
    @Builder.Default
    private long kH = 1;

    //strides
    @Builder.Default
    private long sD = 1;
    @Builder.Default
    private long sW = 1;
    @Builder.Default
    private long sH = 1;

    //padding
    @Builder.Default
    private long pD = 0;
    @Builder.Default
    private long pW = 0;
    @Builder.Default
    private long pH = 0;

    //dilations
    @Builder.Default
    private long dD = 1;
    @Builder.Default
    private long dW = 1;
    @Builder.Default
    private long dH = 1;

    //output padding
    @Builder.Default
    private long aD = 0;
    @Builder.Default
    private long aW = 0;
    @Builder.Default
    private long aH = 0;

    @Builder.Default
    private boolean biasUsed = false;
    private boolean isValidMode;
    @Builder.Default
    private boolean isNCDHW = true;

    @Builder.Default
    private String dataFormat = "NDHWC";

    public Map<String, Object> toProperties() {
        Map<String, Object> ret = new LinkedHashMap<>();
        ret.put("kD", kD);
        ret.put("kW", kW);
        ret.put("kH", kH);
        ret.put("sD", sD);
        ret.put("sW", sW);
        ret.put("sH", sH);
        ret.put("pD", pD);
        ret.put("pW", pW);
        ret.put("pH", pH);
        ret.put("dD", dD);
        ret.put("dW", dW);
        ret.put("dH", dH);
        ret.put("aD", aD);
        ret.put("aW", aW);
        ret.put("aH", aH);
        ret.put("biasUsed", biasUsed);
        ret.put("dataFormat", dataFormat);
        ret.put("isValidMode", isValidMode);

        return ret;
    }


}
