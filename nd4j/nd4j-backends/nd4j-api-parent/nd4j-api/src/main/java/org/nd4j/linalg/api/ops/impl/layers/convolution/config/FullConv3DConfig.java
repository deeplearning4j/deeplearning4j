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

import lombok.Builder;
import lombok.Data;

import java.util.LinkedHashMap;
import java.util.Map;

@Builder
@Data
public class FullConv3DConfig extends BaseConvolutionConfig {
    private long dT,dW,dH,pT,pW,pH,dilationT,dilationW,dilationH,aT,aW,aH;
    private boolean biasUsed;
    private String dataFormat;




    public Map<String,Object> toProperties() {
        Map<String,Object> ret = new LinkedHashMap<>();
        ret.put("dT",dT);
        ret.put("dW",dW);
        ret.put("dH",dH);
        ret.put("pT",pT);
        ret.put("pW",pW);
        ret.put("pH",pH);
        ret.put("dD",dilationT);
        ret.put("dW",dilationW);
        ret.put("dH",dilationH);
        ret.put("aT",aT);
        ret.put("aW",aW);
        ret.put("aH",aH);
        ret.put("biasUsed",biasUsed);
        ret.put("dataFormat",dataFormat);
        return ret;
    }
}
