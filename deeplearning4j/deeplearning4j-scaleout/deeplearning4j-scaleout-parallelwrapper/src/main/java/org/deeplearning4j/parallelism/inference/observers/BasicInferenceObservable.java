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

package org.deeplearning4j.parallelism.inference.observers;

import com.google.common.base.Preconditions;
import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.parallelism.inference.InferenceObservable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.Collections;
import java.util.List;
import java.util.Observable;

/**
 * This class holds reference input, and implements basic use case: SEQUENTIAL inference
 */
@Slf4j
public class BasicInferenceObservable extends Observable implements InferenceObservable {
    private INDArray[] input;
    private INDArray[] inputMasks;
    @Getter
    private long id;
    private INDArray[] output;
    protected Exception exception;


    public BasicInferenceObservable(INDArray... inputs) {
        this(inputs, null);
    }

    public BasicInferenceObservable(INDArray[] inputs, INDArray[] inputMasks){
        super();
        this.input = inputs;
        this.inputMasks = inputMasks;
    }

    @Override
    public void addInput(@NonNull INDArray... input){
        addInput(input, null);
    }

    @Override
    public void addInput(@NonNull INDArray[] input, INDArray[] inputMasks) {
        this.input = input;
        this.inputMasks = inputMasks;
    }

    @Override
    public void setOutputBatches(@NonNull List<INDArray[]> output) {
        Preconditions.checkArgument(output.size() == 1, "Expected size 1 output: got size " + output.size());
        this.output = output.get(0);
        this.setChanged();
        notifyObservers();
    }

    @Override
    public List<Pair<INDArray[],INDArray[]>> getInputBatches(){
        return Collections.singletonList(new Pair<>(input, inputMasks));
    }

    @Override
    public void setOutputException(Exception exception){
        this.exception = exception;
        this.setChanged();
        notifyObservers();
    }

    @Override
    public INDArray[] getOutput(){
        checkOutputException();
        return output;
    }

    protected void checkOutputException(){
        if(exception != null){
            if(exception instanceof RuntimeException){
                throw (RuntimeException)exception;
            } else {
                throw new RuntimeException("Exception encountered while getting output: " + exception.getMessage(), exception);
            }
        }
    }
}
