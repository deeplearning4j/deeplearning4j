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

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

public class SkipGramRound extends DynamicCustomOp {




    @Override
    public String opName() {
        return "skipgram";
    }

    public SkipGramRound(){ }



    /**
     * full constructor
     */
    @Builder
    public SkipGramRound(@NonNull INDArray target,
                         @NonNull INDArray ngStarter,
                         @NonNull INDArray syn0,
                         @NonNull INDArray syn1,
                         @NonNull INDArray syn1Neg,
                         @NonNull INDArray expTable,
                         @NonNull INDArray negTable,
                         int nsRounds,
                         @NonNull INDArray indices,
                         @NonNull INDArray codes,
                         @NonNull INDArray alpha,
                         @NonNull INDArray randomValue,
                         INDArray inferenceVector,
                         boolean preciseMode,
                         int numWorkers,
                         int iterations) {
        inputArguments.add(target);
        inputArguments.add(ngStarter);
        inputArguments.add(indices);
        inputArguments.add(codes);
        inputArguments.add(syn0);
        inputArguments.add(syn1);
        inputArguments.add(syn1Neg);
        inputArguments.add(expTable);
        inputArguments.add(negTable);
        inputArguments.add(alpha);
        inputArguments.add(randomValue);
        inputArguments.add(inferenceVector);


        // couple of options
        iArguments.add((long) numWorkers);
        iArguments.add((long) nsRounds);
        iArguments.add((long) iterations);
        bArguments.add(!inferenceVector.isEmpty());
        bArguments.add(preciseMode);

        // this op is always inplace
        setInPlace(true);
        setInplaceCall(true);

        for (var in:inputArguments)
            outputArguments.add(in);
    }
}
