/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.deeplearning4j.rl4j.observation.transforms;

import lombok.Setter;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.VoidObservation;

/**
 * The PassthroughTransform is the base of all transfroms that takes an Observation as input, process it and output the processed Observation. <br>
 * These transforms are normally used with a PipelineTransform.
 *
 * @author Alexandre Boulanger
 */
public abstract class PassthroughTransform implements ObservationTransform {

    @Setter
    private ObservationTransform previous;

    /**
     * Bring the transform to an initial state and call reset() on the previous transform.
     */
    public void reset() {
        performReset();

        if(previous != null) {
            previous.reset();
        }
    }

    /**
     * A class that extends PassthroughTransform should override this method if it needs to do something to reset itself.
     */
    protected void performReset() {
        // Do Nothing
    }

    /**
     * Will pass the input to the chain (if any) and process the result by calling handle()
     * If input is a VoidObservation, it is returned immediately
     */
    public Observation transform(Observation input) {
        Observation observation = previous == null ? input : previous.transform(input);
        if(observation instanceof VoidObservation) {
            return observation;
        }
        return handle(observation);
    }

    /**
     * In order to be ready, all transforms in the chain should be ready along with the current instance.
     */
    @Override
    public boolean isReady() {
        return (previous == null || previous.isReady()) && getIsReady();
    }

    /**
     * A class that extends PassthroughTransform should override this method to process the input.
     */
    protected abstract Observation handle(Observation input);

    /**
     * A class that extends PassthroughTransform should override this method if it is not always ready. <br>
     * For example, a stacking transform that returns the last X observations stacked together will
     * return false until it has X elements in its pool.
     */
    protected boolean getIsReady() {
        return true;
    }
}
