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
package org.nd4j.linalg.api.ops.impl.layers.recurrent.config;

    /**
     * notations <br>
     * for unidirectional:
     * TNS: shape [timeLength, numExamples, inOutSize] - sometimes referred to as "time major"<br>
     * NST: shape [numExamples, inOutSize, timeLength]<br>
     * NTS: shape [numExamples, timeLength, inOutSize]<br>
     * for bidirectional:
     * T2NS: 3 = [timeLength, 2, numExamples, inOutSize] (for ONNX)
     */

    public enum LSTMDataFormat {
        //Note: ordinal (order) here matters for C++ level. Any new formats hsould be added at end


        TNS,
        NTS,
        NST,
        T2NS

    }




