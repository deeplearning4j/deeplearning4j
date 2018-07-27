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

package org.nd4j.linalg.api.ops.random.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Collections;
import java.util.List;

/**
 * Random normal distribution
 *
 * @author Alex Black
 */
public class RandomNormal extends DynamicCustomOp {

    private double mean;
    private double stdev;

    public RandomNormal() {

    }

    public RandomNormal(SameDiff sameDiff, SDVariable shape, double mean, double stdev) {
        super(null, sameDiff, new SDVariable[]{shape});
        this.mean = mean;
        this.stdev = stdev;

        addTArgument(mean, stdev);
    }

    @Override
    public String opName() {
        return "randomnormal";
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("Not TF op name set for " + getClass().getSimpleName());
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }
}
