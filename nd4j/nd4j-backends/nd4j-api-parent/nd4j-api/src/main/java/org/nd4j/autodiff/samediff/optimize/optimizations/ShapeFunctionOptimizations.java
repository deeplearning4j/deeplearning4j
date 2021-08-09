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

import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.optimize.OptimizationHelper;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.linalg.api.ops.impl.shape.Permute;

import java.util.ArrayList;
import java.util.List;

public class ShapeFunctionOptimizations extends BaseOptimizerSet {

    /**
     * Fuse [permute1 -> permute2 -> ... -> permuteN] into a single permute op,
     * as long as the intermediate permute outputs aren't needed for another op
     */
    public static class FuseChainedPermutes implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op, ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if(!(op.getOp() instanceof Permute))
                return false;

            List<String> inputs = op.getInputsToOp();
            String input = inputs.get(0);

            List<String> toFuse = new ArrayList<>();
            toFuse.add(op.getName());
            String currInput = input;
            while(currInput != null){
                Variable v = sd.getVariables().get(currInput);
                //In order to fuse permute operations, we require:
                // (a) the intermediate variable is ONLY needed by the next permute
                // (b) the permute dimensions are constant,

                if(v.getInputsForOp().size() > 1)
                    break;
            }

            if(toFuse.size() > 1){
                //Fuse the permute ops

//                return true;
                return false;
            }


            return false;
        }
    }

    /**
     * Fuse [reshape1 -> reshape2 -> ... -> reshapeN] into a single reshape op,
     * as long as the intermediate reshape ops aren't needed for another op
     */
    public static class FuseChainedReshapes implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op, ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            return false;
        }
    }

    /**
     * Fuse [concat(concat(concat(x,y,dim=D), z, dim=D), a, dim=D)] into a single concat op, concat(x,y,z,a, dim=D)
     * As long as the intermediate outputs aren't needed elsewhere
     */
    public static class FuseChainedConcatOps implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op, ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            return false;
        }
    }

}
