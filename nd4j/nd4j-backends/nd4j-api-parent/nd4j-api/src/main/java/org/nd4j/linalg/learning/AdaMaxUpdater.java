/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.nd4j.linalg.learning;

import lombok.Data;
import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.AdaMax;

import java.util.HashMap;
import java.util.Map;

/**
 * The AdaMax updater, a variant of Adam.
 * https://arxiv.org/abs/1412.6980
 *
 * @author Justin Long
 */
@Data
public class AdaMaxUpdater implements GradientUpdater<AdaMax> {
    public static final String M_STATE = "M";
    public static final String U_STATE = "V";

    private final AdaMax config;

    private INDArray m, u; // moving avg & exponentially weighted infinity norm
    private char gradientReshapeOrder;

    public AdaMaxUpdater(AdaMax config) {
        this.config = config;
    }

    @Override
    public void setState(@NonNull Map<String, INDArray> stateMap, boolean initialize) {
        if(!stateMap.containsKey(M_STATE) || !stateMap.containsKey(U_STATE) || stateMap.size() != 2){
            throw new IllegalStateException("State map should contain only keys [" + M_STATE + "," + U_STATE + "] but has keys " + stateMap.keySet());
        }
        this.m = stateMap.get(M_STATE);
        this.u = stateMap.get(U_STATE);
    }

    @Override
    public Map<String, INDArray> getState() {
        Map<String,INDArray> r = new HashMap<>();
        r.put(M_STATE, m);
        r.put(U_STATE, u);
        return r;
    }

    @Override
    public void setStateViewArray(INDArray viewArray, long[] gradientShape, char gradientOrder, boolean initialize) {
        if (!viewArray.isRowVector())
            throw new IllegalArgumentException("Invalid input: expect row vector input");
        if (initialize)
            viewArray.assign(0);
        long length = viewArray.length();
        this.m = viewArray.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, length / 2));
        this.u = viewArray.get(NDArrayIndex.point(0), NDArrayIndex.interval(length / 2, length));

        //Reshape to match the expected shape of the input gradient arrays
        this.m = Shape.newShapeNoCopy(this.m, gradientShape, gradientOrder == 'f');
        this.u = Shape.newShapeNoCopy(this.u, gradientShape, gradientOrder == 'f');
        if (m == null || u == null)
            throw new IllegalStateException("Could not correctly reshape gradient view arrays");

        this.gradientReshapeOrder = gradientOrder;
    }

    /**
     * Calculate the update based on the given gradient
     *
     * @param gradient  the gradient to get the update for
     * @param iteration
     * @return the gradient
     */
    @Override
    public void applyUpdater(INDArray gradient, int iteration, int epoch) {
        if (m == null || u == null)
            throw new IllegalStateException("Updater has not been initialized with view state");

        //m = B_1 * m + (1-B_1)*grad
        //u = max(B_2 * u, |grad|)

        double lr = config.getLearningRate(iteration, epoch);
        double b1 = config.getBeta1();
        double b2 = config.getBeta2();
        double eps = config.getEpsilon();

        Nd4j.exec(new org.nd4j.linalg.api.ops.impl.updaters.AdaMaxUpdater(gradient, u, m, lr, b1, b2, eps, iteration));
    }
}
