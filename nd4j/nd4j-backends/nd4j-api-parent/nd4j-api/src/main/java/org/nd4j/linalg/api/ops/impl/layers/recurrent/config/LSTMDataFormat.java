/* ******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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

    /**
     * notations <br>
     * bS - batch size
     * sL - sequence length, number of time steps
     * nIn - input size
     * nOut - output size (hidden size) <br<
     *
     * for unidirectional:
     * SBN: 0 = [sL, bS, nIn],
     * BSN: 1 = [bS, sL ,nIn],
     * BNS: 2 = [bS, nIn, sL],
     * for bidirectional:
     * S2BN: 3 = [sL, 2, bS, nOut] (for ONNX)
     */

    public enum LSTMDataFormat {
        //Note: ordinal (order) here matters for C++ level. Any new formats hsould be added at end


        SBN,
        BSN,
        BNS,
        S2BN

    }




