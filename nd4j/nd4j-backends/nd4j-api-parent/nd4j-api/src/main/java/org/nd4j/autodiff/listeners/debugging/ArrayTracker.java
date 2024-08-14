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
package org.nd4j.autodiff.listeners.debugging;

import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.util.*;

public class ArrayTracker extends BaseListener {
    private final Set<String> variablesToTrack;
    private final Map<String, String> lastKnownStates;

    public ArrayTracker(String... variableNames) {
        this.variablesToTrack = new HashSet<>(Arrays.asList(variableNames));
        this.lastKnownStates = new HashMap<>();
    }
    public ArrayTracker(List<String> variableNames) {
        this.variablesToTrack = new HashSet<>(variableNames);
        this.lastKnownStates = new HashMap<>();
    }



    @Override
    public void preOpExecution(SameDiff sd, At at, SameDiffOp op, OpContext opContext) {
        if (opContext != null) {
            checkAndUpdateStates(opContext, op);
        }

    }

    @Override
    public void opExecution(SameDiff sd, At at, MultiDataSet batch, SameDiffOp op, OpContext opContext, INDArray[] outputs) {
        if (opContext != null) {
             checkAndUpdateStates(opContext, op);
        }
    }



    private void checkAndUpdateStates(OpContext context, SameDiffOp op) {
        for (String varName : variablesToTrack) {
            // Check if the variable is an input to the operation
            if (op.getInputsToOp() != null && op.getInputsToOp().contains(varName)) {
                int inputIdx = op.getInputsToOp().indexOf(varName);
                if (inputIdx >= 0 && inputIdx < context.numInputArguments()) {
                    INDArray array = context.getInputArray(inputIdx);
                    updateArrayState(varName, array);
                }
            }

            // Check if the variable is an output of the operation
            if (op.getOutputsOfOp() != null && op.getOutputsOfOp().contains(varName)) {
                int outputIdx = op.getOutputsOfOp().indexOf(varName);
                if (outputIdx >= 0 && outputIdx < context.numOutputArguments()) {
                    INDArray array = context.getOutputArray(outputIdx);
                    updateArrayState(varName, array);
                }
            }
        }
    }

    private void updateArrayState(String varName, INDArray currentArray) {
        if (currentArray != null && (!lastKnownStates.containsKey(varName) || !lastKnownStates.get(varName).equals(currentArray.toString()))) {
            lastKnownStates.put(varName, currentArray.toString());
        }
    }

    @Override
    public boolean isActive(Operation operation) {
        return true; // Activate listener for all operations
    }
}
