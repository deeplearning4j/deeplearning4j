/*-
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


import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;

/**
 * Binomial sampling pre processor
 * @author Adam Gibson
 */
@Data
@EqualsAndHashCode(callSuper = false)
public class BinomialSamplingPreProcessor extends BaseInputPreProcessor {

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        try(MemoryWorkspace ws = workspaceMgr.notifyScopeEntered(ArrayType.ACTIVATIONS)) {
            return Nd4j.getDistributions().createBinomial(1, input).sample(input.shape());
        }
    }


    @Override
    public INDArray backprop(INDArray output, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        return output; //No op?
    }

    @Override
    public InputType getOutputType(InputType inputType) {
        if (inputType == null)
            throw new IllegalStateException("Invalid input type: cannot be null");
        return inputType;
    }
}
