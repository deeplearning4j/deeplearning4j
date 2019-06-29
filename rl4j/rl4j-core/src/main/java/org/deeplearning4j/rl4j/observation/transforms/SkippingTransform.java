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

import lombok.Builder;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.VoidObservation;
import org.nd4j.base.Preconditions;

/**
 * The SkippingTransform will output either its input observation or VoidObservation when the observation is skipped.
 *
 * @author Alexandre Boulanger
 */
public class SkippingTransform extends PassthroughTransform {

    private int skipFrame = 1;
    private int currentIdx = 0;

    /**
     * @param skipFrame For example, a skipFrame of 4 will skip 3 out of 4 observations.
     */
    public SkippingTransform(int skipFrame) {
        Preconditions.checkArgument(skipFrame > 0, "skipFrame must be greater than 0, got %s", skipFrame);
        this.skipFrame = skipFrame;
    }

    public SkippingTransform(Builder builder) {
        this.skipFrame = builder.skipFrame;
    }

    @Override
    public void reset() {
        super.reset();
        currentIdx = 0;
    }

    @Override
    public boolean getIsReady() {
        return true;
    }

    @Override
    protected Observation handle(Observation input) {
        if(currentIdx++ % skipFrame == 0) {
            return input;
        }

        return VoidObservation.getInstance();
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private int skipFrame = 1;

        /**
         * @param skipFrame For example, a skipFrame of 4 will skip 3 out of 4 observations.
         */
        public Builder skipFrame(int skipFrame) {
            Preconditions.checkArgument(skipFrame > 0, "skipFrame must be greater than 0, got %s", skipFrame);
            this.skipFrame = skipFrame;

            return this;
        }

        public SkippingTransform build() {
            return new SkippingTransform(this);
        }
    }
}
