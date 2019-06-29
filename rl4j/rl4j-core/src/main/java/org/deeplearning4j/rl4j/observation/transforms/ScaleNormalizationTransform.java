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
import org.deeplearning4j.rl4j.observation.SimpleObservation;
import org.deeplearning4j.rl4j.observation.VoidObservation;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * The ScaleNormalizationTransform is a basic normalization transform that divides the input by a known scale.
 *
 * @author Alexandre Boulanger
 */
public class ScaleNormalizationTransform extends PassthroughTransform {

    private final double scale;

    private ScaleNormalizationTransform(Builder builder) {
        this(builder.scale);
    }

    public ScaleNormalizationTransform(double scale) {
        this.scale = scale;
    }

    @Override
    protected Observation handle(Observation input) {
        INDArray ndArray = input.toNDArray();
        ndArray.muli(1.0 / scale);

        return new SimpleObservation(ndArray);
    }

    @Override
    protected boolean getIsReady() {
        return true;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private double scale = 1.0;

        /**
         *
         * @param scale should be greater than 0. The inputs of the transforms will multiplied by (1.0 / scale).
         * @return
         */
        public Builder scale(double scale) {
            Preconditions.checkArgument(scale > 0.0, "The scale must be greater than 0, got %s", scale);

            this.scale = scale;
            return this;
        }

        public ScaleNormalizationTransform build() {
            return new ScaleNormalizationTransform(this);
        }
    }
}