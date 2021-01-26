/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.arbiter.optimize.parameter;

import lombok.EqualsAndHashCode;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * BooleanParameterSpace is a {@code ParameterSpace<Boolean>}; Defines {True, False} as a parameter space
 * If argument to setValue is less than or equal to 0.5 it will return True else False
 *
 * @author susaneraly
 */
@EqualsAndHashCode
public class BooleanSpace implements ParameterSpace<Boolean> {
    private int index = -1;

    @Override
    public Boolean getValue(double[] input) {
        if (index == -1) {
            throw new IllegalStateException("Cannot get value: ParameterSpace index has not been set");
        }
        if (input[index] <= 0.5) return Boolean.TRUE;
        else return Boolean.FALSE;
    }

    @Override
    public int numParameters() {
        return 1;
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return Collections.singletonList((ParameterSpace) this);
    }

    @Override
    public Map<String, ParameterSpace> getNestedSpaces() {
        return Collections.emptyMap();
    }

    @Override
    public boolean isLeaf() {
        return true;
    }

    @Override
    public void setIndices(int... indices) {
        if (indices == null || indices.length != 1)
            throw new IllegalArgumentException("Invalid index");
        this.index = indices[0];
    }

    @Override
    public String toString() {
        return "BooleanSpace()";
    }
}
