/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.arbiter.optimize.api.adapter;

import lombok.AllArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;

import java.util.List;
import java.util.Map;

/**
 * An abstract class used for adapting one type into another. Subclasses of this need to merely implement 2 simple methods
 *
 * @param <F> Type to convert from
 * @param <T> Type to convert to
 * @author Alex Black
 */
@AllArgsConstructor
public abstract class ParameterSpaceAdapter<F, T> implements ParameterSpace<T> {


    protected abstract T convertValue(F from);

    protected abstract ParameterSpace<F> underlying();


    @Override
    public T getValue(double[] parameterValues) {
        return convertValue(underlying().getValue(parameterValues));
    }

    @Override
    public int numParameters() {
        return underlying().numParameters();
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return underlying().collectLeaves();
    }

    @Override
    public Map<String, ParameterSpace> getNestedSpaces() {
        return underlying().getNestedSpaces();
    }

    @Override
    public boolean isLeaf() {
        return underlying().isLeaf();
    }

    @Override
    public void setIndices(int... indices) {
        underlying().setIndices(indices);
    }

    @Override
    public String toString() {
        return underlying().toString();
    }
}
