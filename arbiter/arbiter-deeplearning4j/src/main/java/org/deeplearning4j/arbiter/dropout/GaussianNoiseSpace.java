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
import org.deeplearning4j.nn.conf.dropout.GaussianNoise;
import org.deeplearning4j.nn.conf.dropout.IDropout;

import java.util.Collections;
import java.util.List;
import java.util.Map;

@AllArgsConstructor
public class GaussianNoiseSpace implements ParameterSpace<IDropout> {

    private ParameterSpace<Double> stddev;

    public GaussianNoiseSpace(double stddev){
        this(new FixedValue<>(stddev));
    }

    @Override
    public IDropout getValue(double[] parameterValues) {
        return new GaussianNoise(stddev.getValue(parameterValues));
    }

    @Override
    public int numParameters() {
        return stddev.numParameters();
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return Collections.<ParameterSpace>singletonList(stddev);
    }

    @Override
    public Map<String, ParameterSpace> getNestedSpaces() {
        return Collections.<String,ParameterSpace>singletonMap("stddev", stddev);
    }

    @Override
    public boolean isLeaf() {
        return false;
    }

    @Override
    public void setIndices(int... indices) {
        stddev.setIndices(indices);
    }
}
