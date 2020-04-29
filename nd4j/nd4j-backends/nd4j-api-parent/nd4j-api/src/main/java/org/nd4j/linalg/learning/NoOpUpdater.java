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

package org.nd4j.linalg.learning;

import lombok.Data;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.NoOp;

import java.util.Collections;
import java.util.Map;

/**
 * NoOp updater: gradient updater that makes no changes to the gradient
 *
 * @author Alex Black
 */
@Data
public class NoOpUpdater implements GradientUpdater<NoOp> {

    private final NoOp config;

    public NoOpUpdater(NoOp config) {
        this.config = config;
    }

    @Override
    public void setState(Map<String, INDArray> stateMap, boolean initialize) {
        Preconditions.checkState(stateMap == null || stateMap.isEmpty(), "No-op updater does not have any updater state," +
                " attempting to set with %s values", (stateMap == null ? 0 : stateMap.size()));
    }

    @Override
    public Map<String, INDArray> getState() {
        return Collections.emptyMap();
    }

    @Override
    public void setStateViewArray(INDArray viewArray, long[] shape, char order, boolean initialize) {
        //No op
    }

    @Override
    public void applyUpdater(INDArray gradient, int iteration, int epoch) {
        //No op
    }
}
