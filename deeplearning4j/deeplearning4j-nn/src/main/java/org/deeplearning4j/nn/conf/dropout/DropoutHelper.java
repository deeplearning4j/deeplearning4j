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

package org.deeplearning4j.nn.conf.dropout;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A helper interface for native dropout implementations
 *
 * @author Alex Black
 */
public interface DropoutHelper {

    /**
     * @return Check if this dropout helper is supported in the current environment
     */
    boolean checkSupported();

    /**
     * Apply the dropout during forward pass
     * @param inputActivations       Input activations (pre dropout)
     * @param resultArray            Output activations (post dropout). May be same as (or different to) input array
     * @param dropoutInputRetainProb Probability of retaining an activation
     */
    void applyDropout(INDArray inputActivations, INDArray resultArray, double dropoutInputRetainProb);

    /**
     * Perform backpropagation. Note that the same dropout mask should be used for backprop as was used during the last
     * call to {@link #applyDropout(INDArray, INDArray, double)}
     * @param gradAtOutput Gradient at output (from perspective of forward pass)
     * @param gradAtInput  Result array - gradient at input. May be same as (or different to) gradient at input
     */
    void backprop(INDArray gradAtOutput, INDArray gradAtInput);


}

