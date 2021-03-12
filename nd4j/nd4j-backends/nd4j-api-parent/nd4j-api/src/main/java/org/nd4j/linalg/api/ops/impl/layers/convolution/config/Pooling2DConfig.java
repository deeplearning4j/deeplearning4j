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
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling2D.Divisor;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling2D.Pooling2DType;
import org.nd4j.linalg.util.ConvConfigUtil;

@Data
@Builder
public class Pooling2DConfig extends BaseConvolutionConfig {

    @Builder.Default private long kH = -1, kW = -1;
    @Builder.Default private long sH = 1, sW = 1;
    @Builder.Default private long pH = 0, pW = 0;
    /**
     * Extra is an optional parameter mainly for use with pnorm right now.
     * All pooling implementations take 9 parameters save pnorm.
     * Pnorm takes 10 and is cast to an int.
     */
    private double extra;
    @Builder.Default
    private Pooling2D.Pooling2DType type = Pooling2DType.MAX;
    @Builder.Default
    private Pooling2D.Divisor divisor = Pooling2D.Divisor.EXCLUDE_PADDING;
    private boolean isSameMode;
    @Builder.Default
    private long dH = 1;
    @Builder.Default
    private long dW = 1;
    @Builder.Default
    private boolean isNHWC = false;

    public Pooling2DConfig() {
    }

    public Pooling2DConfig(long kH, long kW, long sH, long sW, long pH, long pW, double extra, Pooling2DType type,
                           Divisor divisor, boolean isSameMode, long dH, long dW, boolean isNHWC) {
        this.kH = kH;
        this.kW = kW;
        this.sH = sH;
        this.sW = sW;
        this.pH = pH;
        this.pW = pW;
        this.extra = extra;
        this.type = type;
        this.divisor = divisor;
        this.isSameMode = isSameMode;
        this.dH = dH;
        this.dW = dW;
        this.isNHWC = isNHWC;

        validate();
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
        ret.put("extra", extra);
        ret.put("type", type.toString());
        ret.put("isSameMode", isSameMode);
        ret.put("dH", dH);
        ret.put("dW", dW);
        ret.put("isNHWC", isNHWC);
        return ret;
    }

    @Override
    protected void validate() {
        ConvConfigUtil.validate2D(kH, kW, sH, sW, pH, pW, dH, dW);

        //TODO check other args?
    }

}
