/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.conf.preprocessor;

import lombok.AccessLevel;
import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Unit variance operation
 *
 * @author Adma Gibson
 */
@Data
public class UnitVarianceProcessor extends BaseInputPreProcessor {

    @Getter(AccessLevel.NONE)
    @Setter(AccessLevel.NONE)
	INDArray columnStds;

    @Override
    public INDArray preProcess(INDArray input) {
        columnStds = input.std(0);
        columnStds.addi(Nd4j.EPS_THRESHOLD);
        input.diviRowVector(columnStds);
        return input;
    }

    @Override
    public INDArray backprop(INDArray output) {
        return output;	//no-op
    }

    @Override
    public UnitVarianceProcessor clone() {
        UnitVarianceProcessor clone = (UnitVarianceProcessor) super.clone();
        if(clone.columnStds != null) clone.columnStds = clone.columnStds.dup();
        return clone;
    }
}
