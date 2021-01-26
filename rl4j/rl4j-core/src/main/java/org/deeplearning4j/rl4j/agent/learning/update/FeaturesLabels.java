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
package org.deeplearning4j.rl4j.agent.learning.update;

import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;

/**
 * A container that holds the features and the associated labels.
 */
public class FeaturesLabels {

    @Getter
    private final Features features;

    private final HashMap<String, INDArray> labels = new HashMap<String, INDArray>();

    /**
     * @param features
     */
    public FeaturesLabels(Features features) {
        this.features = features;
    }

    /**
     * @return The number of examples in features and each labels.
     */
    public long getBatchSize() {
        return features.getBatchSize();
    }

    /**
     * Add labels by name
     * @param name
     * @param labels
     */
    public void putLabels(String name, INDArray labels) {
        this.labels.put(name, labels);
    }

    /**
     * Get the labels associated to the name.
     * @param name
     * @return
     */
    public INDArray getLabels(String name) {
        return this.labels.get(name);
    }
}
