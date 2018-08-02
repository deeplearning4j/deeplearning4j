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

package org.deeplearning4j.arbiter.conf.dropout;

import lombok.AllArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.AbstractParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.nn.conf.dropout.Dropout;
import org.deeplearning4j.nn.conf.dropout.IDropout;

import java.util.List;

@AllArgsConstructor
public class DropoutSpace extends AbstractParameterSpace<IDropout> {

    private ParameterSpace<Double> dropout;

    @Override
    public Dropout getValue(double[] parameterValues) {
        double p = dropout.getValue(parameterValues);
        if(p == 0){
            //Special case: 0 dropout = "disabled" in DL4J. But Dropout class doesn't support this
            return null;
        }
        return new Dropout(p);
    }

    @Override
    public int numParameters() {
        return dropout.numParameters();
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return dropout.collectLeaves();
    }

    @Override
    public boolean isLeaf() {
        return false;
    }

    @Override
    public void setIndices(int... indices) {
        dropout.setIndices(indices);
    }
}
