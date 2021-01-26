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
import org.nd4j.linalg.util.ConvConfigUtil;

@Data
@Builder
@NoArgsConstructor
public class LocalResponseNormalizationConfig extends BaseConvolutionConfig {

    private double alpha, beta, bias;
    private int depth;

    public LocalResponseNormalizationConfig(double alpha, double beta, double bias, int depth) {
        this.alpha = alpha;
        this.beta = beta;
        this.bias = bias;
        this.depth = depth;

        validate();
    }

    @Override
    public Map<String, Object> toProperties() {
        Map<String, Object> ret = new LinkedHashMap<>();
        ret.put("alpha", alpha);
        ret.put("beta", beta);
        ret.put("bias", bias);
        ret.put("depth", depth);
        return ret;
    }

    @Override
    protected void validate() {
        ConvConfigUtil.validateLRN(alpha, beta, bias, depth);
    }

}
