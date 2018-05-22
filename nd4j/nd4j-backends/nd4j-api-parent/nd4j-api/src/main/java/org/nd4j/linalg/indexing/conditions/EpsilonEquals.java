/*-
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 */

package org.nd4j.linalg.indexing.conditions;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by agibsonccc on 10/8/14.
 */
public class EpsilonEquals extends BaseCondition {
    private double eps = Nd4j.EPS_THRESHOLD;
    /**
     * Special constructor for pairwise boolean operations.
     */
    public EpsilonEquals() {
        this(0.0);
    }

    public EpsilonEquals(Number value) {
        this(value, Nd4j.EPS_THRESHOLD);
    }

    public EpsilonEquals(Number value, Number eps) {
        super(value);
        this.eps = eps.doubleValue();
    }

    public EpsilonEquals(IComplexNumber complexNumber) {
        super(complexNumber);
    }

    @Override
    public double epsThreshold() {
        return this.eps;
    }

    /**
     * Returns condition ID for native side
     *
     * @return
     */
    @Override
    public int condtionNum() {
        return 0;
    }

    @Override
    public Boolean apply(Number input) {
        return Math.abs(input.floatValue() - value.floatValue()) < Nd4j.EPS_THRESHOLD;
    }

    @Override
    public Boolean apply(IComplexNumber input) {
        return Math.abs(input.absoluteValue().floatValue() - input.absoluteValue().floatValue()) < Nd4j.EPS_THRESHOLD;

    }
}
