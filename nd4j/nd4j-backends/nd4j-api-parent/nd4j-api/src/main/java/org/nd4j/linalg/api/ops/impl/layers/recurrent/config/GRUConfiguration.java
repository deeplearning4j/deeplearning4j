/*
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

@Data
@Builder
/**
 *
 * @author Max Pumperla
 */
public class GRUConfiguration {
    /**
     NDArray<T>* x    = INPUT_VARIABLE(0);    // input [time x bS x inSize]
     NDArray<T>* h0   = INPUT_VARIABLE(1);     // initial cell output (at time step = 0) [bS x numUnits]

     NDArray<T>* Wx   = INPUT_VARIABLE(2);    // input-to-hidden  weights, [inSize   x 3*numUnits]
     NDArray<T>* Wh   = INPUT_VARIABLE(3);    // hidden-to-hidden weights, [numUnits x 3*numUnits]
     NDArray<T>* b    = INPUT_VARIABLE(4);    // biases, [1 x 3*numUnits]

     NDArray<T>* h    =  OUTPUT_VARIABLE(0);  // cell outputs [time x bS x numUnits], that is per each time step
     */

    private SDVariable x, h0, Wx, Wh, b;


    public SDVariable[] args() {
        return new SDVariable[] {x, h0, Wx, Wh, b};
    }

}
