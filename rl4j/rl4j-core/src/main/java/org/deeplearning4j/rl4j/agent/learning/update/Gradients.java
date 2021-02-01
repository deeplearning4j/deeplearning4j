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
package org.deeplearning4j.rl4j.agent.learning.update;

import lombok.Getter;
import org.deeplearning4j.nn.gradient.Gradient;

import java.util.HashMap;

/**
 * A {@link Gradient} container used to update neural networks.
 */
public class Gradients {

    @Getter
    private final long batchSize;

    private final HashMap<String, Gradient> gradients = new HashMap<String, Gradient>();

    /**
     * @param batchSize The size of the training batch used to create this instance
     */
    public Gradients(long batchSize) {
        this.batchSize = batchSize;
    }

    /**
     * Add a {@link Gradient} by name.
     * @param name
     * @param gradient
     */
    public void putGradient(String name, Gradient gradient) {
        gradients.put(name, gradient);
    }

    /**
     * Get a {@link Gradient} by name
     * @param name
     * @return
     */
    public Gradient getGradient(String name) {
        return gradients.get(name);
    }

}
