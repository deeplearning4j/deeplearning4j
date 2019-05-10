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

package org.deeplearning4j.optimize.stepfunctions;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.optimize.api.StepFunction;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Subtract the line
 *
 * @author Adam Gibson
 */
@Slf4j
public class NegativeGradientStepFunction implements StepFunction {
    @Override
    public void step(INDArray x, INDArray line, double step) {
        step(x, line);
    }

    @Override
    public void step(INDArray x, INDArray line) {
        x.subi(line);
    }

    @Override
    public void step() {

    }
}
