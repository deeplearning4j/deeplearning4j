/*
 *  ******************************************************************************
 *  *
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
package org.nd4j.linalg.api.ops.impl.layers.recurrent.config;

/**
 * direction <br>
 *  FWD: 0 = fwd
 *  BWD: 1 = bwd
 *  BIDIR_SUM: 2 = bidirectional sum
 *  BIDIR_CONCAT: 3 = bidirectional concat
 *  BIDIR_EXTRA_DIM: 4 = bidirectional extra output dim (in conjunction with format dataFormat = 3) */

//    const auto directionMode = INT_ARG(1);    // direction:

public enum LSTMDirectionMode {
    //Note: ordinal (order) here matters for C++ level. Any new formats hsould be added at end


    FWD,
    BWD,
    BIDIR_SUM,
    BIDIR_CONCAT,
    BIDIR_EXTRA_DIM

}
