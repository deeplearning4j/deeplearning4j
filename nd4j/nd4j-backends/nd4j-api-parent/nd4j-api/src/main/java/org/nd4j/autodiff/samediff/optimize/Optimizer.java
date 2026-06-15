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

package org.nd4j.autodiff.samediff.optimize;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;

import java.util.Set;

/**
 * @author Alex Black
 */
public interface Optimizer {

    /**
     * @param sd              Current SameDiff instance to optimize
     * @param helper          Helper class for optimization
     * @param op              Operation to check for optimization
     * @param constantArrays  Array holder for constant arrays
     * @param variablesArrays Array holder for variable arrays
     * @return True if the optimization was applied
     */
    boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op, ArrayHolder constantArrays, ArrayHolder variablesArrays);

    /**
     * Returns the set of op types this optimizer is interested in.
     * If null or empty, the optimizer will be called for all ops.
     * This allows for significant performance improvement by filtering ops early.
     *
     * @return Set of DifferentialFunction classes this optimizer handles, or null for all ops
     */
    default Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
        return null;  // Default: check all ops
    }

}
