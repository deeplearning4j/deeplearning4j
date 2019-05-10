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

package org.deeplearning4j.arbiter.dropout;

import lombok.AllArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.nn.conf.dropout.Dropout;
import org.deeplearning4j.nn.conf.dropout.GaussianDropout;
import org.deeplearning4j.nn.conf.dropout.IDropout;

import java.util.Collections;
import java.util.List;
import java.util.Map;

@AllArgsConstructor
public class GaussianDropoutSpace implements ParameterSpace<IDropout> {

    private ParameterSpace<Double> rate;

    public GaussianDropoutSpace(double rate){
        this(new FixedValue<>(rate));
    }

    @Override
    public IDropout getValue(double[] parameterValues) {
        return new GaussianDropout(rate.getValue(parameterValues));
    }

    @Override
    public int numParameters() {
        return rate.numParameters();
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return Collections.<ParameterSpace>singletonList(rate);
    }

    @Override
    public Map<String, ParameterSpace> getNestedSpaces() {
        return Collections.<String,ParameterSpace>singletonMap("rate", rate);
    }

    @Override
    public boolean isLeaf() {
        return false;
    }

    @Override
    public void setIndices(int... indices) {
        rate.setIndices(indices);
    }
}
