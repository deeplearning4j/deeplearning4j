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
import org.nd4j.base.Preconditions;

import java.util.LinkedHashMap;
import java.util.Map;

@Builder
@Data
@AllArgsConstructor
@NoArgsConstructor
public class Conv1DConfig extends BaseConvolutionConfig {
    public static final String NCW = "NCW";
    public static final String NWC = "NWC";

    @Builder.Default private long k = -1L;
    @Builder.Default
    private long s = 1;
    @Builder.Default
    private long p = 0;
    @Builder.Default
    private String dataFormat = NCW;
    private boolean isSameMode;

    public boolean isNWC(){
        Preconditions.checkState(dataFormat.equalsIgnoreCase(NCW) || dataFormat.equalsIgnoreCase(NWC),
                "Data format must be one of %s or %s, got %s", NCW, NWC, dataFormat);
        return dataFormat.equalsIgnoreCase(NWC);
    }

    public void isNWC(boolean isNWC){
        if(isNWC){
            dataFormat = NWC;
        } else {
            dataFormat = NCW;
        }
    }

    public Map<String, Object> toProperties() {
        Map<String, Object> ret = new LinkedHashMap<>();
        ret.put("k", k);
        ret.put("s", s);
        ret.put("p", p);
        ret.put("isSameMode", isSameMode);
        ret.put("dataFormat", dataFormat);
        return ret;
    }


}
