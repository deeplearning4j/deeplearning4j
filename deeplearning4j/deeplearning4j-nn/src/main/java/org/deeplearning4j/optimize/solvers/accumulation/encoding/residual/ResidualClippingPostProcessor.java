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

package org.deeplearning4j.optimize.solvers.accumulation.encoding.residual;

import org.deeplearning4j.optimize.solvers.accumulation.encoding.ResidualPostProcessor;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

public class ResidualClippingPostProcessor implements ResidualPostProcessor {

    private final double thresholdMultipleClipValue;
    private final int frequency;

    public ResidualClippingPostProcessor(double thresholdMultipleClipValue, int frequency) {
        Preconditions.checkState(thresholdMultipleClipValue >= 1.0, "Threshold multiple must be a positive value and " +
                "greater than 1.0 (1.0 means ");
        this.thresholdMultipleClipValue = thresholdMultipleClipValue;
        this.frequency = frequency;
    }

    @Override
    public void processResidual(int iteration, int epoch, double lastThreshold, INDArray residualVector) {
        if(iteration > 0 && iteration % frequency == 0) {


            double currClip = lastThreshold * thresholdMultipleClipValue;
            //TODO replace with single op once we have GPU op version
            BooleanIndexing.replaceWhere(residualVector, currClip, Conditions.greaterThan(currClip));
            BooleanIndexing.replaceWhere(residualVector, -currClip, Conditions.lessThan(-currClip));
        }
    }

    @Override
    public ResidualClippingPostProcessor clone() {
        return new ResidualClippingPostProcessor(thresholdMultipleClipValue, frequency);
    }
}
