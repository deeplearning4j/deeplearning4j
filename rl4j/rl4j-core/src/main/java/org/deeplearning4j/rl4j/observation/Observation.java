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

package org.deeplearning4j.rl4j.observation;

import lombok.Getter;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Represent an observation from the environment
 *
 * @author Alexandre Boulanger
 */
public class Observation {

    /**
     * A singleton representing a skipped observation
     */
    public static Observation SkippedObservation = new Observation(null);

    /**
     * @return A INDArray containing the data of the observation
     */
    @Getter
    private final INDArray data;

    public boolean isSkipped() {
        return data == null;
    }

    public Observation(INDArray data) {
        this.data = data;
    }

    /**
     * Creates a duplicate instance of the current observation
     * @return
     */
    public Observation dup() {
        if(data == null) {
            return SkippedObservation;
        }

        return new Observation(data.dup());
    }
}
