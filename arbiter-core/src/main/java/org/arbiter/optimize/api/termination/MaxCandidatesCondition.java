/*
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
package org.arbiter.optimize.api.termination;

import lombok.AllArgsConstructor;
import org.arbiter.optimize.runner.IOptimizationRunner;

@AllArgsConstructor
public class MaxCandidatesCondition implements TerminationCondition {

    private final int maxCandidates;

    @Override
    public void initialize(IOptimizationRunner optimizationRunner) {
        //No op
    }

    @Override
    public boolean terminate(IOptimizationRunner optimizationRunner) {
        return optimizationRunner.numCandidatesTotal() >= maxCandidates;
    }

    @Override
    public String toString(){
        return "MaxCandidatesCondition("+maxCandidates+")";
    }
}
