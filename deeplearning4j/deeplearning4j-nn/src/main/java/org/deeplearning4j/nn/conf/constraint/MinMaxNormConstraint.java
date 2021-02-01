/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.conf.constraint;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Broadcast;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.util.Collections;
import java.util.Set;

@Data
@EqualsAndHashCode(callSuper = true)
public class MinMaxNormConstraint extends BaseConstraint {
    public static final double DEFAULT_RATE = 1.0;

    private double min;
    private double max;
    private double rate;

    private MinMaxNormConstraint(){
        //No arg for json ser/de
    }

    /**
     * Apply to weights but not biases by default
     *
     * @param max            Maximum L2 value
     * @param min            Minimum L2 value
     * @param dimensions     Dimensions to apply to. For DenseLayer, OutputLayer, RnnOutputLayer, LSTM, etc: this should
     *                       be dimension 1. For CNNs, this should be dimensions [1,2,3] corresponding to last 3 of
     *                       parameters which have order [depthOut, depthIn, kH, kW]
     */
    public MinMaxNormConstraint(double min, double max, int... dimensions){
        this(min, max, DEFAULT_RATE, null, dimensions);
    }

    /**
     * Apply to weights but not biases by default
     *
     * @param max            Maximum L2 value
     * @param min            Minimum L2 value
     * @param rate           Constraint rate
     * @param dimensions     Dimensions to apply to. For DenseLayer, OutputLayer, RnnOutputLayer, LSTM, etc: this should
     *                       be dimension 1. For CNNs, this should be dimensions [1,2,3] corresponding to last 3 of
     *                       parameters which have order [depthOut, depthIn, kH, kW]
     */
    public MinMaxNormConstraint(double min, double max, double rate, int... dimensions){
        this(min, max, rate, Collections.<String>emptySet(), dimensions);
    }

    /**
     *
     * @param max            Maximum L2 value
     * @param min            Minimum L2 value
     * @param rate           Constraint rate
     * @param paramNames     Which parameter names to apply constraint to
     * @param dimensions     Dimensions to apply to. For DenseLayer, OutputLayer, RnnOutputLayer, LSTM, etc: this should
     *                       be dimension 1. For CNNs, this should be dimensions [1,2,3] corresponding to last 3 of
     *                       parameters which have order [depthOut, depthIn, kH, kW]
     */
    public MinMaxNormConstraint(double min, double max, double rate, Set<String> paramNames, int... dimensions){
        super(paramNames, dimensions);
        if(rate <= 0 || rate > 1.0){
            throw new IllegalStateException("Invalid rate: must be in interval (0,1]: got " + rate);
        }
        this.min = min;
        this.max = max;
        this.rate = rate;
    }

    @Override
    public void apply(INDArray param) {
        INDArray norm = param.norm2(dimensions);
        INDArray clipped = norm.unsafeDuplication();
        CustomOp op = DynamicCustomOp.builder("clipbyvalue")
                .addInputs(clipped)
                .callInplace(true)
                .addFloatingPointArguments(min, max)
                .build();
        Nd4j.getExecutioner().exec(op);

        norm.addi(epsilon);
        clipped.divi(norm);

        if(rate != 1.0){
            clipped.muli(rate).addi(norm.muli(1.0-rate));
        }

        Broadcast.mul(param, clipped, param, getBroadcastDims(dimensions, param.rank()) );
    }

    @Override
    public MinMaxNormConstraint clone() {
        return new MinMaxNormConstraint(min, max, rate, params, dimensions);
    }
}
