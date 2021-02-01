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

import java.util.LinkedHashMap;
import java.util.Map;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling3D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling3D.Pooling3DType;
import org.nd4j.linalg.util.ConvConfigUtil;

@Data
@Builder
@NoArgsConstructor
public class Pooling3DConfig extends BaseConvolutionConfig {
    @Builder.Default private long kD = -1, kW = -1, kH = -1; // kernel
    @Builder.Default private long sD = 1, sW = 1, sH = 1; // strides
    @Builder.Default private long pD = 0, pW = 0, pH = 0; // padding
    // dilation
    @Builder.Default
    private long dD = 1;
    @Builder.Default
    private long dW = 1;
    @Builder.Default
    private long dH = 1;
    @Builder.Default
    private Pooling3D.Pooling3DType type = Pooling3DType.MAX;
    private boolean isSameMode;
    @Builder.Default private boolean isNCDHW = true;

    public Pooling3DConfig(long kD, long kW, long kH, long sD, long sW, long sH, long pD, long pW, long pH, long dD,
            long dW, long dH, Pooling3DType type, boolean isSameMode, boolean isNCDHW) {
        this.kD = kD;
        this.kW = kW;
        this.kH = kH;
        this.sD = sD;
        this.sW = sW;
        this.sH = sH;
        this.pD = pD;
        this.pW = pW;
        this.pH = pH;
        this.dD = dD;
        this.dW = dW;
        this.dH = dH;
        this.type = type;
        this.isSameMode = isSameMode;
        this.isNCDHW = isNCDHW;

        validate();
    }

    @Override
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
        ret.put("isSameMode", isSameMode);
        return ret;

    }

    @Override
    protected void validate() {
        ConvConfigUtil.validate3D(kH, kW, kD, sH, sW, sD, pH, pW, pD, dH, dW, dD);

        //TODO check other args
    }
}
