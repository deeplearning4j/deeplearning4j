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

package org.deeplearning4j.arbiter.layers;

import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.arbiter.util.LeafUtils;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.LSTM;

/**
 * Layer space for LSTM layers
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PRIVATE) //For Jackson JSON/YAML deserialization
public class LSTMLayerSpace extends AbstractLSTMLayerSpace<LSTM> {

    private LSTMLayerSpace(Builder builder) {
        super(builder);

        this.numParameters = LeafUtils.countUniqueParameters(collectLeaves());
    }


    @Override
    public LSTM getValue(double[] values) {
        LSTM.Builder b = new LSTM.Builder();
        setLayerOptionsBuilder(b, values);
        return b.build();
    }

    protected void setLayerOptionsBuilder(LSTM.Builder builder, double[] values) {
        super.setLayerOptionsBuilder(builder, values);
    }

    @Override
    public String toString() {
        return toString(", ");
    }

    @Override
    public String toString(String delim) {
        StringBuilder sb = new StringBuilder("LSTMLayerSpace(");
        sb.append(super.toString(delim)).append(")");
        return sb.toString();
    }

    public static class Builder extends AbstractLSTMLayerSpace.Builder<Builder> {

        @Override
        @SuppressWarnings("unchecked")
        public LSTMLayerSpace build() {
            return new LSTMLayerSpace(this);
        }
    }
}
