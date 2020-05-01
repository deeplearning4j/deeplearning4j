/*******************************************************************************
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
package org.deeplearning4j.rl4j.observation.transform.operation;

import org.datavec.api.transform.Operation;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;

public class SimpleNormalizationTransform implements Operation<INDArray, INDArray> {

    private final double offset;
    private final double divisor;

    public SimpleNormalizationTransform(double min, double max) {
        Preconditions.checkArgument(min < max, "Min must be smaller than max.");

        this.offset = min;
        this.divisor = (max - min);
    }

    @Override
    public INDArray transform(INDArray input) {
        if(offset != 0.0) {
            input.subi(offset);
        }

        input.divi(divisor);

        return input;
    }
}
