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
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.common.primitives.Counter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.*;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.util.ArrayList;
import java.util.List;

public class ControlflowListener extends BaseListener {

    private Counter<String> entersExecuted = new Counter<>();
    private Counter<String> exitsExecuted = new Counter<>();
    private Counter<String> mergesExecuted = new Counter<>();
    private Counter<String> nextIterationExecuted = new Counter<>();

    private Counter<String> switchesExecuted = new Counter<>();

    private Counter<String> loopCondExecuted = new Counter<>();

    @Override
    public boolean isActive(Operation operation) {
        return true;
    }

    @Override
    public void operationStart(SameDiff sd, Operation op) {
        super.operationStart(sd, op);
    }

    @Override
    public void operationEnd(SameDiff sd, Operation op) {
        super.operationEnd(sd, op);
    }

    @Override
    public void preOpExecution(SameDiff sd, At at, SameDiffOp op, OpContext opContext) {
        super.preOpExecution(sd, at, op, opContext);


    }

    @Override
    public void opExecution(SameDiff sd, At at, MultiDataSet batch, SameDiffOp op, OpContext opContext, INDArray[] outputs) {
        super.opExecution(sd, at, batch, op, opContext, outputs);
        if(op.getOp() instanceof Enter) {
            entersExecuted.incrementCount(op.getName(),1.0);
        } else if(op.getOp() instanceof Exit) {
            exitsExecuted.incrementCount(op.getName(),1.0);
        } else if(op.getOp() instanceof NextIteration) {
            nextIterationExecuted.incrementCount(op.getName(),1.0);
        } else if(op.getOp() instanceof Switch) {
            switchesExecuted.incrementCount(op.getName(),1.0);
        } else if(op.getOp() instanceof Merge) {
            mergesExecuted.incrementCount(op.getName(),1.0);
        } else if(op.getOp() instanceof LoopCond) {
            loopCondExecuted.incrementCount(op.getName(),1.0);
        }
    }
}
