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

import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.VoidObservation;

import java.util.ArrayList;
import java.util.List;

/**
 * The PipelineTransform is used to chain together several PassthroughTransforms. <br>
 *
 * @author Alexandre Boulanger
 */
public class PipelineTransform implements ObservationTransform {

    private final ObservationTransform outputObservationTransform;

    private PipelineTransform(Builder builder) {
        ObservationTransform previous = builder.previous;

        for (PassthroughTransform transform : builder.transforms) {
            transform.setPrevious(previous);
            previous = transform;
        }

        outputObservationTransform = previous;
    }

    /**
     * Call reset on all transforms in the pipeline.
     */
    public void reset() {
        if(outputObservationTransform != null) {
            outputObservationTransform.reset();
        }
    }

    /**
     * Pass the input through all the pipeline transforms and return the result.
     */
    public Observation transform(Observation input) {
        if(input instanceof VoidObservation) {
            return input;
        }

        return outputObservationTransform == null  ? input : outputObservationTransform.transform(input);
    }

    /**
     * A PipelineTransform will be ready when all transforms within are ready.
     */
    @Override
    public boolean isReady() {
        return outputObservationTransform == null || outputObservationTransform.isReady();
    }

    /**
     * Return a PipelineTransform builder that will be chained after the transform 'previous'.
     */
    public static Builder builder(ObservationTransform previous) {
        return new Builder(previous);
    }

    /**
     * Return a PipelineTransform builder.
     */
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private final ObservationTransform previous;
        private List<PassthroughTransform> transforms = new ArrayList<>();

        public Builder() {
            previous = null;
        }

        /**
         * The builder will chain the PipelineTransform after the transform 'previous'
         */
        public Builder(ObservationTransform previous) {
            this.previous = previous;
        }

        /**
         * Append a transform at the end of the pipeline.
         */
        public Builder flowTo(PassthroughTransform transform) {
            transforms.add(transform);
            return this;
        }

        public PipelineTransform build() {
            return new PipelineTransform(this);
        }

    }
}
