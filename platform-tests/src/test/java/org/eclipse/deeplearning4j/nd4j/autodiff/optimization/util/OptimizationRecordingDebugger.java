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
package org.eclipse.deeplearning4j.nd4j.autodiff.optimization.util;

import lombok.Getter;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.autodiff.samediff.optimize.debug.OptimizationDebugger;

import java.util.HashMap;
import java.util.Map;

public class OptimizationRecordingDebugger implements OptimizationDebugger {

    @Getter
    private Map<String,Optimizer> applied = new HashMap<>();

    @Override
    public void beforeOptimizationCheck(SameDiff sd, SameDiffOp op, Optimizer o) {
        //No op
    }

    @Override
    public void afterOptimizationsCheck(SameDiff sd, SameDiffOp op, Optimizer o, boolean wasApplied) {
        if(wasApplied){
            applied.put(op.getName(), o);
        }
    }
}