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

package org.nd4j.linalg.api.ops.impl.layers.recurrent.config;

import lombok.Builder;
import lombok.Data;
import org.nd4j.autodiff.samediff.SDVariable;

@Data
@Builder
public class GRUCellConfiguration {
    /*
    Inputs:
    x        input [bS x inSize]
    hLast    previous cell output [bS x numUnits],  that is at previous time step t-1
    Wru      RU weights - [bS, 2*numUnits] - reset and update gates
    Wc       C weights - [bS, numUnits] - cell gate
    bru      r and u biases, [2*numUnits] - reset and update gates
    bc       c biases, [numUnits] - cell gate
     */

    private SDVariable xt, hLast, Wru, Wc, bru, bc;

    public SDVariable[] args() {
        return new SDVariable[] {xt, hLast, Wru, Wc, bru, bc};
    }

}
