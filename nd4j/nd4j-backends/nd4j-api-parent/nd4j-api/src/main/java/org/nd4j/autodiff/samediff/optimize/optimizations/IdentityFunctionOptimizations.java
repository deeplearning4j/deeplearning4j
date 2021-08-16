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
import org.nd4j.autodiff.samediff.optimize.OptimizationHelper;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.linalg.api.ops.impl.transforms.same.Identity;

import java.util.Properties;

public class IdentityFunctionOptimizations extends BaseOptimizerSet {

    /**
     * Remove permute(0,1,2,...,rank-1) as this is a no-op
     */
    public static class RemoveIdentityPermute implements Optimizer {

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op, ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            return false;
        }
    }

    /**
     * Remove identity(x)
     */
    public static class RemoveIdentityOps implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op, ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if(op.getOp() instanceof Identity){
                String inName = op.getInputsToOp().get(0);
                String outputName = op.getOutputsOfOp().get(0);
                OptimizationUtils.removeOp(sd, op.getName());
                OptimizationUtils.replaceOpInputsWith(sd, outputName, inName);
                OptimizationUtils.removeVariable(sd, outputName);
                return true;
            }

            return false;
        }
    }
}
