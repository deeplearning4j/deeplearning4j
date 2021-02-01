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

    /**
     * integer numbers corresponding to activations:
     * 0=tanh,
     * 1=relu,
     * 2=sigmoid,
     * 3=affine,
     * 4=leaky relu,
     * 5= thresholded relu,
     * 6=scaled tanh,
     * 7=hard sigmoid,
     * 8=ELU,
     * 9=softsign,
     * 10=softplus
     */
    public enum LSTMActivations {
        //Note: ordinal (order) here matters for C++ level. Any new formats hsould be added at end

        TANH,
        RELU,
        SIGMOID,
        AFFINE,
        LEAKY_RELU,
        THRESHHOLD_RELU,
        SCALED_TAHN,
        HARD_SIGMOID,
        ELU,
        SOFTSIGN,
        SOFTPLUS


}
