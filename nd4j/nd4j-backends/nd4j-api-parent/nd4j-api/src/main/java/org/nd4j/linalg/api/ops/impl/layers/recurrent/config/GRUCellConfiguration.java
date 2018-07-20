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

package org.nd4j.linalg.api.ops.impl.layers.recurrent.config;

import lombok.Builder;
import lombok.Data;
import org.nd4j.autodiff.samediff.SDVariable;

import java.util.LinkedHashMap;
import java.util.Map;

@Data
@Builder
public class GRUCellConfiguration {
    /**
     *     NDArray<T>* xt   = INPUT_VARIABLE(0);                   // input [batchSize x inSize]
     NDArray<T>* ht_1 = INPUT_VARIABLE(1);                   // previous cell output [batchSize x numUnits],  that is at previous time step t-1

     NDArray<T>* Wx   = INPUT_VARIABLE(2);                   // input-to-hidden  weights, [inSize   x 3*numUnits]
     NDArray<T>* Wh   = INPUT_VARIABLE(3);                   // hidden-to-hidden weights, [numUnits x 3*numUnits]
     NDArray<T>* b    = INPUT_VARIABLE(4);                   // biases, [1 x 3*numUnits]

     NDArray<T>* ht   =  OUTPUT_VARIABLE(0);                  // current cell output [batchSize x numUnits], that is at current time step t

     const int batchSize   = (INPUT_VARIABLE(0))->sizeAt(0);
     const int inSize      = (INPUT_VARIABLE(0))->sizeAt(1);
     const int numUnits    = (INPUT_VARIABLE(1))->sizeAt(1);

     */

    private int batchSize,intSize,numUnits;
    private SDVariable xt,ht_1,Wx,Wh,b;

    public Map<String,Object> toProperties() {
        Map<String,Object> ret = new LinkedHashMap<>();
        ret.put("batchSize",batchSize);
        ret.put("intSize",intSize);
        ret.put("numUnits",numUnits);
        return ret;
    }

    public int[] iArgs() {
        return new int[] {batchSize,intSize,numUnits};
    }

    public SDVariable[] args() {
        return new SDVariable[] {xt,ht_1,Wx,Wh,b};
    }

}
