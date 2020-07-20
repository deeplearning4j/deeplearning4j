/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.deeplearning4j.optimize.solvers.accumulation.encoding.threshold;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.ThresholdAlgorithm;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.ThresholdAlgorithmReducer;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A simple fixed threshold algorithm, not adaptive in any way.
 * An adaptive threshold algorithm such as {@link AdaptiveThresholdAlgorithm} should be preferred for better performance
 * in most cases.
 *
 * @author Alex Black
 */
@AllArgsConstructor
@Data
public class FixedThresholdAlgorithm implements ThresholdAlgorithm {

    private final double threshold;

    @Override
    public double calculateThreshold(int iteration, int epoch, Double lastThreshold, Boolean lastWasDense, Double lastSparsityRatio, INDArray updatesPlusResidual) {
        return threshold;
    }

    @Override
    public ThresholdAlgorithmReducer newReducer() {
        return new FixedAlgorithmThresholdReducer();
    }

    @Override
    public FixedThresholdAlgorithm clone() {
        return new FixedThresholdAlgorithm(threshold);
    }


    public static class FixedAlgorithmThresholdReducer implements ThresholdAlgorithmReducer {

        private FixedThresholdAlgorithm instance;

        @Override
        public void add(ThresholdAlgorithm instance) {
            Preconditions.checkState(instance instanceof FixedThresholdAlgorithm, "Invalid threshold: cannot be reduced using this class, %s", instance.getClass().getSimpleName());
            this.instance = (FixedThresholdAlgorithm) instance;
        }

        @Override
        public ThresholdAlgorithmReducer merge(ThresholdAlgorithmReducer other) {
            if(this.instance != null || other == null)
                return this;
            this.instance = ((FixedAlgorithmThresholdReducer)other).instance;
            return this;
        }

        @Override
        public ThresholdAlgorithm getFinalResult() {
            return instance;
        }
    }
}
