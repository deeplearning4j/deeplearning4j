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

/**
 * Created by agibsonccc on 10/9/14.
 */
public class Not implements Condition {

    private Condition opposite;

    /**
     * Returns condition ID for native side
     *
     * @return
     */
    @Override
    public int condtionNum() {
        return -1;
    }

    @Override
    public double getValue() {
        return -1;
    }

    public Not(Condition condition) {
        this.opposite = condition;
    }

    @Override
    public Boolean apply(Number input) {
        return !opposite.apply(input);
    }

    @Override
    public Boolean apply(IComplexNumber input) {
        return !opposite.apply(input);
    }

    @Override
    public double epsThreshold() {
        return 0;
    }
}
