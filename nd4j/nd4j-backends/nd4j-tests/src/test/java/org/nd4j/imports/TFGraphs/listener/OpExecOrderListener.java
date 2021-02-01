/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.imports.tfgraphs.listener;

import lombok.Getter;
import lombok.Setter;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;

import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.dataset.api.MultiDataSet;

public class OpExecOrderListener extends BaseListener {

    @Getter @Setter
    protected List<String> opNamesList;
    protected Set<String> opSet;

    public OpExecOrderListener(){
        this.opNamesList = new ArrayList<>();
        this.opSet = new HashSet<>();
    }

    @Override
    public void opExecution(SameDiff sd, At at, MultiDataSet batch, SameDiffOp op, OpContext opContext, INDArray[] outputs) {
        String opName = op.getName();
        if(!opSet.contains(opName)){
            opNamesList.add(opName);
            opSet.add(opName);
        }
    }

    @Override
    public boolean isActive(Operation operation) {
        return true;
    }
}
