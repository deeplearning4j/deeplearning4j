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

package org.deeplearning4j.rl4j.observation;

import lombok.Getter;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Represent an observation from the environment
 *
 * @author Alexandre Boulanger
 */
// TODO: Remove Encodable
public class Observation implements Encodable {

    /**
     * A singleton representing a skipped observation
     */
    public static Observation SkippedObservation = new Observation();

    /**
     * @return A INDArray containing the data of the observation
     */
    @Getter
    private final INDArray[] channelsData;

    public INDArray getChannelData(int channelIdx) {
        return channelsData[channelIdx];
    }

    // TODO: Remove once Encodable is removed
    @Override
    public double[] toArray() {
        return channelsData[0].data().asDouble();
    }

    public boolean isSkipped() {
        return channelsData == null;
    }

    private Observation() {
        this.channelsData = null;
    }

    // TODO: Remove when legacy code is gone
    public Observation(INDArray data) {
        this.channelsData = new INDArray[] { data };
    }

    public Observation(INDArray[] channelsData) {
        this.channelsData = channelsData;
    }

    // TODO: Remove when legacy code is gone
    public INDArray getData() {
        return channelsData[0];
    }

    public int numChannels() {
        return channelsData.length;
    }

    /**
     * Creates a duplicate instance of the current observation
     * @return
     */
    public Observation dup() {
        if(channelsData == null) {
            return SkippedObservation;
        }

        INDArray[] duplicated = new INDArray[channelsData.length];
        for(int i = 0; i < channelsData.length; ++i) {
            duplicated[i] = channelsData[i].dup();
        }
        return new Observation(duplicated);
    }
}
