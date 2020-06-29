/*******************************************************************************
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
package org.deeplearning4j.rl4j.network;

import org.nd4j.linalg.dataset.api.DataSet;

/**
 * An interface defining the <i>trainable</i> aspect of a {@link NeuralNet}.
 */
public interface ITrainableNeuralNet<NET_TYPE extends ITrainableNeuralNet> {
    /**
     * Train the neural net using the supplied <i>feature-labels</i>
     * @param featuresLabels The feature-labels
     */
    void fit(DataSet featuresLabels);

    /**
     * Changes this instance to be a copy of the <i>from</i> network.
     * @param from The network that will be the source of the copy.
     */
    void copy(NET_TYPE from);

    /**
     * Creates a clone of the network instance.
     */
    NET_TYPE clone();
}
