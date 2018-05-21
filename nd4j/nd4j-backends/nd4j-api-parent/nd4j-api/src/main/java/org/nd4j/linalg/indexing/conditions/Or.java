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
 * An or between 2 conditions.
 *
 * @author Adam Gibson
 */
public class Or implements Condition {

    private Condition[] conditions;

    public Or(Condition... conditions) {
        this.conditions = conditions;
    }

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

    @Override
    public Boolean apply(Number input) {
        boolean ret = conditions[0].apply(input);
        for (int i = 1; i < conditions.length; i++) {
            ret = ret || conditions[i].apply(input);
        }
        return ret;
    }

    @Override
    public Boolean apply(IComplexNumber input) {
        boolean ret = conditions[0].apply(input);
        //short circuit: no need to check anything else
        if (!ret)
            return false;
        for (int i = 1; i < conditions.length; i++) {
            ret = conditions[i].apply(input);
        }
        return ret;
    }

    @Override
    public double epsThreshold() {
        return 0;
    }
}
