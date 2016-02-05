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
package org.arbiter.optimize.parameter;

import org.arbiter.optimize.api.ParameterSpace;

import java.util.Collections;
import java.util.List;

public class FixedValue<T> implements ParameterSpace<T> {
    private T value;
    private int index;

    public FixedValue(T value) {
        this.value = value;
    }

    @Override
    public String toString(){
        return "FixedValue("+value+")";
    }

    @Override
    public T getValue(double[] input) {
        return value;
    }

    @Override
    public int numParameters() {
        return 0;
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return Collections.emptyList();
    }

    @Override
    public boolean isLeaf() {
        return true;
    }

    @Override
    public void setIndices(int... indices) {
        if(indices != null && indices.length != 0) throw new IllegalArgumentException("Invaild: FixedValue ParameterSpace "
            + "should not be given an index");
    }
}
