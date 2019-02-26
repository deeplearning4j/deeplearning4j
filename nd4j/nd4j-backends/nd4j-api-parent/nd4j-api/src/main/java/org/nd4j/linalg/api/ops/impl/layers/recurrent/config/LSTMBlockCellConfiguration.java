/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.linalg.api.ops.impl.layers.recurrent.config;

import lombok.Builder;
import lombok.Data;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.LinkedHashMap;
import java.util.Map;

@Builder
@Data
public class LSTMBlockCellConfiguration {

    private boolean peepHole;           //IArg(0)
    private double forgetBias;          //TArg(0)
    private double clippingCellValue;   //TArg(1)

    private SDVariable xt, cLast, yLast, W, Wci, Wcf, Wco, b;

    public Map<String,Object> toProperties()  {
        Map<String,Object> ret = new LinkedHashMap<>();
        ret.put("peepHole",peepHole);
        ret.put("clippingCellValue",clippingCellValue);
        ret.put("forgetBias",forgetBias);
        return ret;
    }

    public SDVariable[] args()  {
        return new SDVariable[] {xt,cLast, yLast, W, Wci, Wcf, Wco, b};
    }


    public int[] iArgs() {
        return new int[] {ArrayUtil.fromBoolean(peepHole)};
    }

    public double[] tArgs() {
        return new double[] {forgetBias,clippingCellValue};
    }
}
