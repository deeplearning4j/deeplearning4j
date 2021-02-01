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

package org.nd4j.linalg.api.ops.impl.layers.convolution.config;


import java.util.LinkedHashMap;
import java.util.Map;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.util.ConvConfigUtil;

@Data
@Builder
@NoArgsConstructor
public class Conv3DConfig extends BaseConvolutionConfig {
    public static final String NDHWC = "NDHWC";
    public static final String NCDHW = "NCDHW";

    //kernel
    @Builder.Default
    private long kD = -1;
    @Builder.Default
    private long kW = -1;
    @Builder.Default
    private long kH = -1;

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

    @Builder.Default
    private boolean biasUsed = false;
    private boolean isSameMode;

    @Builder.Default
    private String dataFormat = NDHWC;

    public Conv3DConfig(long kD, long kW, long kH, long sD, long sW, long sH, long pD, long pW, long pH, long dD,
            long dW, long dH, boolean biasUsed, boolean isSameMode, String dataFormat) {
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
        this.biasUsed = biasUsed;
        this.isSameMode = isSameMode;
        this.dataFormat = dataFormat;

        validate();
    }

    public boolean isNCDHW(){
        Preconditions.checkState(dataFormat.equalsIgnoreCase(NCDHW) || dataFormat.equalsIgnoreCase(NDHWC),
                "Data format must be one of %s or %s, got %s", NCDHW, NDHWC, dataFormat);
        return dataFormat.equalsIgnoreCase(NCDHW);
    }

    public void isNCDHW(boolean isNCDHW){
        if(isNCDHW){
            dataFormat = NCDHW;
        } else {
            dataFormat = NDHWC;
        }
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
        ret.put("biasUsed", biasUsed);
        ret.put("dataFormat", dataFormat);
        ret.put("isSameMode", isSameMode);

        return ret;
    }

    @Override
    protected void validate() {
        ConvConfigUtil.validate3D(kH, kW, kD, sH, sW, sD, pH, pW, pD, dH, dW, dD);
        Preconditions.checkArgument(dataFormat != null, "Data format can't be null");
    }


}
