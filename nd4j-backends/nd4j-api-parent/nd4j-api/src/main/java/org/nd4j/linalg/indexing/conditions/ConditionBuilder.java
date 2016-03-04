/*
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

import org.nd4j.linalg.util.ArrayUtil;

/**
 * Mini dsl for building conditions
 *
 * @author Adam Gibson
 */
public class ConditionBuilder {

    private Condition soFar;


    public ConditionBuilder or(Condition... conditions) {
        if (soFar == null)
            soFar = new Or(conditions);
        else {
            soFar = new Or(ArrayUtil.combine(conditions, new Condition[]{soFar}));
        }
        return this;
    }

    public ConditionBuilder and(Condition... conditions) {
        if (soFar == null)
            soFar = new And(conditions);
        else {
            soFar = new And(ArrayUtil.combine(conditions, new Condition[]{soFar}));
        }
        return this;
    }

    public ConditionBuilder eq(Condition... conditions) {
        if (soFar == null)
            soFar = new ConditionEquals(conditions);
        else {
            soFar = new ConditionEquals(ArrayUtil.combine(conditions, new Condition[]{soFar}));
        }
        return this;
    }

    public ConditionBuilder not() {
        if (soFar == null)
            throw new IllegalStateException("No condition to take the opposite of");
        soFar = new Not(soFar);
        return this;
    }

    public Condition build() {
        return soFar;
    }


}
