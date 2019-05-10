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

package org.deeplearning4j.nn.api;

/**
 * MaskState: specifies whether a mask should be applied or not.
 *
 * Masks should not be applied in all cases, depending on the network configuration - for example input Dense -> RNN
 * -> Dense -> RnnOutputLayer<br>
 * The first dense layer should be masked (using the input mask) whereas the second shouldn't be, as it has valid data
 * coming from the RNN layer below. For variable length situations like that, the masking can be implemented using the
 * label mask, which will backpropagate 0s for those time steps.<br>
 * In other cases, the *should* be applied - for example, input -> BidirectionalRnn -> Dense -> Output. In such a case,
 * the dense layer should be masked using the input mask.<br>
 * <p>
 * Essentially: Active = apply mask to activations and errors.<br>
 * Passthrough = feed forward the input mask (if/when necessary) but don't actually apply it.<br>
 *
 * @author Alex Black
 */
public enum MaskState {
    Active, Passthrough
}
