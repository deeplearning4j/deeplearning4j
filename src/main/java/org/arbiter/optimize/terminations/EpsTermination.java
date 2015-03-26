/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.arbiter.optimize.terminations;

import org.arbiter.optimize.api.TerminationCondition;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Epsilon termination (absolute change based on tolerance)
 *
 * @author Adam Gibson
 */
public class EpsTermination implements TerminationCondition {
    private double eps = 1e-4;
    private double tolerance = Nd4j.EPS_THRESHOLD;

    public EpsTermination(double eps, double tolerance) {
        this.eps = eps;
        this.tolerance = tolerance;
    }

    public EpsTermination() {
    }

    @Override
    public boolean terminate(double cost,double old, Object[] otherParams) {
       //special case for initial termination, ignore
        if(cost == 0 && old == 0)
           return false;

        if(otherParams.length >= 2) {
            double eps = (double) otherParams[0];
            double tolerance = (double) otherParams[1];
            return 2.0 * Math.abs(old-cost) <= tolerance*(Math.abs(old) + Math.abs(cost) + eps);
        }

        else
            return 2.0 * Math.abs(old  - cost) <= tolerance * (Math.abs(old) + Math.abs(cost) + eps);




    }
}
