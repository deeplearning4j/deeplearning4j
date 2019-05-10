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

package org.deeplearning4j.nn.conf;

/** Defines the type of backpropagation. 'Standard' setting (default) is used
 * for training most networks (MLP, CNN, etc)
 * In the context of recurrent neural networks, Standard means 
 * @author Alex
 *
 */
public enum BackpropType {
    /** Default option. Used for training most networks, including MLP, DBNs, CNNs etc.*/
    Standard,
    /** Truncated BackPropagation Through Time. Only applicable in context of
     * training networks with recurrent neural network layers such as GravesLSTM
     */
    TruncatedBPTT
}
