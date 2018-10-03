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
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling3D;

import java.util.LinkedHashMap;
import java.util.Map;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class Pooling3DConfig extends BaseConvolutionConfig {
    private long kD, kW, kH; // kernel
    private long sD, sW, sH; // strides
    private long pD, pW, pH; // padding
    // dilation
    @Builder.Default
    private long dD = 1;
    @Builder.Default
    private long dW = 1;
    @Builder.Default
    private long dH = 1;
    private Pooling3D.Pooling3DType type;
    private boolean ceilingMode;
    @Builder.Default private boolean isNCDHW = true;

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
        ret.put("type", type.toString());
        ret.put("ceilingMode", ceilingMode);
        return ret;

    }
}
