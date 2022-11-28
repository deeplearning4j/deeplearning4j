/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.nlp;

import lombok.Builder;
import lombok.NonNull;
import lombok.val;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.shade.guava.primitives.Ints;

public class SkipGramInference extends DynamicCustomOp {




    @Override
    public String opName() {
        return "skipgram_inference";
    }

    public SkipGramInference(){ }



    /**
     * full constructor
     */
    @Builder
    public SkipGramInference(@NonNull int target,
                             @NonNull int iteration,
                             @NonNull int ngStarter,
                             @NonNull INDArray syn0,
                             @NonNull INDArray syn1,
                             @NonNull INDArray syn1Neg,
                             @NonNull INDArray expTable,
                             @NonNull INDArray negTable,
                             int nsRounds,
                             @NonNull int[] indices,
                             @NonNull byte  [] codes,
                             @NonNull double[] alpha,
                             @NonNull int randomValue,
                             INDArray inferenceVector,
                             boolean preciseMode,
                             int numWorkers) {

        inputArguments.add(syn0);
        inputArguments.add(syn1);
        inputArguments.add(syn1Neg);
        inputArguments.add(expTable);
        inputArguments.add(negTable);
        inputArguments.add(inferenceVector);


        iArguments.add((long) codes.length);
        iArguments.add((long) indices.length);
        iArguments.add((long) iteration);

        int codeIdx = 0;
        int indicesIdx = 0;
        for(int i = 0; i < codes.length + indices.length; i++) {
            if(i < codes.length)
                iArguments.add((long) codes[codeIdx++]);
            else
                iArguments.add((long) indices[indicesIdx++]);

        }


        // couple of options
        iArguments.add((long) target);
        iArguments.add((long) ngStarter);
        iArguments.add((long) randomValue);
        iArguments.add((long) numWorkers);
        iArguments.add((long) nsRounds);
        for(double tArg : alpha)
            tArguments.add(tArg);

        bArguments.add(!inferenceVector.isEmpty());
        bArguments.add(preciseMode);

        // this op is always inplace
        setInPlace(true);
        setInplaceCall(true);

        for (val in:inputArguments)
            outputArguments.add(in);
    }
}
