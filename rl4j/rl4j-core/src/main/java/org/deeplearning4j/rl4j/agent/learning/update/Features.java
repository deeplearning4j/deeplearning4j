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
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A container that holds the observations of a batch
 */
public class Features {

    private final INDArray[] features;

    /**
     * The size of the batch
     */
    @Getter
    private final long batchSize;

    public Features(INDArray[] features) {
        this.features = features;
        batchSize = features[0].shape()[0];
    }

    /**
     * @param channelIdx The channel to get
     * @return A {@link INDArray} associated to the channel index
     */
    public INDArray get(int channelIdx) {
        return features[channelIdx];
    }
}