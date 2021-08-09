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

package org.nd4j.autodiff.samediff.optimize.optimizations;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class OptimizationUtils {

    private OptimizationUtils(){ }

    public static void replaceOpInputsWith(SameDiff sd, @NonNull String replaceInput, @NonNull String newInput){
        if(replaceInput.equals(newInput))
            return;

        //Update op input structure: Replace all instances replaceInput->X with newInput->X
        Collection<SameDiffOp> ops = sd.getOps().values();
        for(SameDiffOp o : ops){
            List<String> l = o.getInputsToOp();
            while(l != null && l.contains(replaceInput)){
                int idx = l.indexOf(replaceInput);
                l.set(idx, newInput);
            }
        }

        //Update variable structure
        Variable v = sd.getVariables().get(replaceInput);
        Variable v2 = sd.getVariables().get(newInput);
        //NOTE: this only works if we carefully control the order in which replaceOpInputsWith is called!
        v2.setInputsForOp(v.getInputsForOp());
        v.setInputsForOp(new ArrayList<String>());
    }

    public static void removeOp(@NonNull SameDiff sd, @NonNull String opToRemove){
        SameDiffOp op = sd.getOps().remove(opToRemove);
        for(String s : op.getInputsToOp()){
            Variable v = sd.getVariables().get(s);
            v.getInputsForOp().remove(op.getName());
        }
    }

    public static void removeVariable(@NonNull SameDiff sd, @NonNull String varToRemove){
        sd.getVariables().remove(varToRemove);
    }

}
