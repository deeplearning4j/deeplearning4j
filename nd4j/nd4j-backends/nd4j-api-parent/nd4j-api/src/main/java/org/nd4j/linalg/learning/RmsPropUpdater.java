/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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

package org.nd4j.linalg.learning;

import lombok.Data;
import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;

import java.util.Collections;
import java.util.Map;

/**
 * RMS Prop updates:
 * <p>
 * http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
 * http://cs231n.github.io/neural-networks-3/#ada
 *
 * @author Adam Gibson
 */
@Data
public class RmsPropUpdater implements GradientUpdater<RmsProp> {
    public static final String G_STATE = "G";

    private final RmsProp config;

    private INDArray lastGradient;
    private char gradientReshapeOrder;

    public RmsPropUpdater(RmsProp config) {
        this.config = config;
    }

    @Override
    public void setState(@NonNull Map<String, INDArray> stateMap, boolean initialize) {
        if(!stateMap.containsKey(G_STATE) || stateMap.size() != 1){
            throw new IllegalStateException("State map should contain only key [" + G_STATE + "] but has keys " + stateMap.keySet());
        }
        this.lastGradient = stateMap.get(G_STATE);
    }

    @Override
    public Map<String, INDArray> getState() {
        return Collections.singletonMap(G_STATE, this.lastGradient);
    }

    @Override
    public void setStateViewArray(INDArray viewArray, long[] gradientShape, char gradientOrder, boolean initialize) {
        if (!viewArray.isRowVectorOrScalar())
            throw new IllegalArgumentException("Invalid input: expect row vector input");
        if (initialize)
            viewArray.assign(config.getEpsilon());
        this.lastGradient = viewArray;

        //Reshape to match the expected shape of the input gradient arrays
        this.lastGradient = Shape.newShapeNoCopy(this.lastGradient, gradientShape, gradientOrder == 'f');
        if (lastGradient == null)
            throw new IllegalStateException("Could not correctly reshape gradient view array");

        gradientReshapeOrder = gradientOrder;
    }

    @Override
    public void applyUpdater(INDArray gradient, int iteration, int epoch) {
        if (lastGradient == null)
            throw new IllegalStateException("Updater has not been initialized with view state");

        double learningRate = config.getLearningRate(iteration, epoch);
        double rmsDecay = config.getRmsDecay();
        double epsilon = config.getEpsilon();

        // lr * gradient / (sqrt(cache) + 1e-8)
        Nd4j.exec(new org.nd4j.linalg.api.ops.impl.updaters.RmsPropUpdater(gradient, lastGradient, learningRate, rmsDecay, epsilon));
    }
}
