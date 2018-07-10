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

package org.deeplearning4j.nn.workspace;

/**
 * Array type enumeration for use with {@link LayerWorkspaceMgr}<br>
 * <br>
 * Array types:<br>
 * INPUT: The array set to the input field of a layer (i.e., input activations)<br>
 * ACTIVATIONS: The output activations for a layer's feed-forward method<br>
 * ACTIVATION_GRAD: Activation gradient arrays - aka "epsilons" - output from a layer's backprop method<br>
 * FF_WORKING_MEM: Working memory during feed-forward. Arrays allocated here will be invalidated once a layer's
 *                 feed-forward method returns.<br>
 * BP_WORKING_MEM: Working memory during backprop. Arrays allocated here will be invalidated once a layer's backprop
 *                 method returns<br>
 * RNN_FF_LOOP_WORKING_MEM: Working memory during a single time step of RNN forward pass. Opened/closed once per timestep
 *                 for RNN layers only.<br>
 * RNN_BP_LOOP_WORKING_MEM Working memory during a single time step of RNN backward pass. Opened/closed once per timestep
 *                 for RNN layers only.<br>
 * UPDATER_WORKING_MEM: Working memory for updaters (like {@link org.nd4j.linalg.learning.config.Adam},
 *                 {@link org.nd4j.linalg.learning.config.Nesterovs} etc.<br>
 * FF_CACHE: Only used in some layers, and when {@link org.deeplearning4j.nn.conf.CacheMode} is not set to NONE. Used
 *                 to increase performance at the expense of memory, by storing partial activations from forward
 *                 pass, so they don't need to be recalculated during backprop
 *
 * @author Alex Black
 */
public enum ArrayType {
    INPUT,
    ACTIVATIONS,
    ACTIVATION_GRAD,
    FF_WORKING_MEM,
    BP_WORKING_MEM,
    RNN_FF_LOOP_WORKING_MEM,
    RNN_BP_LOOP_WORKING_MEM,
    UPDATER_WORKING_MEM,
    FF_CACHE
}
