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

import java.util.LinkedHashMap;
import java.util.Map;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.base.Preconditions;
import org.nd4j.enums.WeightsFormat;
import org.nd4j.linalg.util.ConvConfigUtil;

@Data
@Builder
@NoArgsConstructor
public class Conv2DConfig extends BaseConvolutionConfig {
    public static final String NCHW = "NCHW";
    public static final String NHWC = "NHWC";

    @Builder.Default
    private long kH = -1L;
    @Builder.Default
    private long kW = -1L;
    @Builder.Default
    private long sH = 1;
    @Builder.Default
    private long sW = 1; // strides >= 1
    @Builder.Default
    private long pH = 0; // padding >= 0
    @Builder.Default
    private long pW = 0;
    @Builder.Default
    private long dH = 1;
    @Builder.Default
    private long dW = 1;  // dilations >= 1
    private boolean isSameMode;
    @Builder.Default
    private String dataFormat = NCHW;
    @Builder.Default
    private WeightsFormat weightsFormat = WeightsFormat.YXIO;

    public Conv2DConfig(long kH, long kW, long sH, long sW, long pH, long pW, long dH, long dW, boolean isSameMode,
            String dataFormat, WeightsFormat weightsFormat) {

        this.kH = kH;
        this.kW = kW;
        this.sH = sH;
        this.sW = sW;
        this.pH = pH;
        this.pW = pW;
        this.dH = dH;
        this.dW = dW;
        this.isSameMode = isSameMode;
        this.dataFormat = dataFormat;
        this.weightsFormat = weightsFormat;

        validate();
    }

    public boolean isNHWC(){
        Preconditions.checkState(dataFormat.equalsIgnoreCase(NCHW) || dataFormat.equalsIgnoreCase(NHWC),
                "Data format must be one of %s or %s, got %s", NCHW, NHWC, dataFormat);
        return dataFormat.equalsIgnoreCase(NHWC);
    }

    public void isNHWC(boolean isNHWC){
        if(isNHWC){
            dataFormat = NHWC;
        } else {
            dataFormat = NCHW;
        }
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
        ret.put("isSameMode", isSameMode);
        ret.put("dataFormat", dataFormat);
        return ret;
    }

    @Override
    protected void validate() {
        ConvConfigUtil.validate2D(kH, kW, sH, sW, pH, pW, dH, dW);
        Preconditions.checkArgument(dataFormat != null, "Data format can't be null");
    }


}
