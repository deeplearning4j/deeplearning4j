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

package org.nd4j.autodiff.listeners;

import org.nd4j.autodiff.listeners.records.LossCurve;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.dataset.api.MultiDataSet;

public abstract class BaseListener implements Listener {


    @Override
    public ListenerVariables requiredVariables(SameDiff sd){
        return ListenerVariables.empty();
    }

    @Override
    public void epochStart(SameDiff sd, At at) {
        //No op
    }

    @Override
    public ListenerResponse epochEnd(SameDiff sd, At at, LossCurve lossCurve, long epochTimeMillis) {
        return ListenerResponse.CONTINUE;
    }

    @Override
    public ListenerResponse validationDone(SameDiff sd, At at, long validationTimeMillis) {
        //No op
        return ListenerResponse.CONTINUE;
    }

    @Override
    public void iterationStart(SameDiff sd, At at, MultiDataSet data, long etlMs) {
        //No op
    }

    @Override
    public void iterationDone(SameDiff sd, At at, MultiDataSet dataSet, Loss loss) {
        //No op
    }

    @Override
    public void operationStart(SameDiff sd, Operation op) {
        //No op
    }

    @Override
    public void operationEnd(SameDiff sd, Operation op) {
        //No op
    }

    @Override
    public void preOpExecution(SameDiff sd, At at, SameDiffOp op, OpContext opContext) {
        //No op
    }

    @Override
    public void opExecution(SameDiff sd, At at, MultiDataSet batch, SameDiffOp op, OpContext opContext, INDArray[] outputs) {
        //No op
    }

    @Override
    public void activationAvailable(SameDiff sd, At at, MultiDataSet batch, SameDiffOp op, String varName,
            INDArray activation) {
        //No op
    }

    @Override
    public void preUpdate(SameDiff sd, At at, Variable v, INDArray update) {
        //No op
    }
}
