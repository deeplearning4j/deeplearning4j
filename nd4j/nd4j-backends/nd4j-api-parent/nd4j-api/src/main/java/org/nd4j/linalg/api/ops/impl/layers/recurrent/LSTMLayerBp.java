/*
 *  ******************************************************************************
 *  *
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
package org.nd4j.linalg.api.ops.impl.layers.recurrent;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMLayerConfig;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMLayerWeights;
import org.nd4j.shade.guava.primitives.Booleans;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;


/**
 * LSTM layer backpropagation
 */
@NoArgsConstructor
public class LSTMLayerBp extends DynamicCustomOp {

    @Getter
    private LSTMLayerConfig configuration;

    @Getter
    private LSTMLayerWeights weights;

    private SDVariable cLast;
    private SDVariable yLast;
    private SDVariable maxTSLength;


    public LSTMLayerBp(@NonNull SameDiff sameDiff, @NonNull SDVariable x, SDVariable cLast, SDVariable yLast, SDVariable maxTSLength, @NonNull LSTMLayerWeights weights, @NonNull LSTMLayerConfig configuration,
                       SDVariable dLdh, SDVariable dLdhL, SDVariable dLdcL) {
        super("lstmLayer_bp", sameDiff, wrapFilterNull(x, weights.getWeights(), weights.getRWeights(), weights.getBias(),
                maxTSLength, yLast, cLast, weights.getPeepholeWeights(), dLdh, dLdhL, dLdcL));
        this.configuration = configuration;
        this.weights = weights;
        this.cLast = cLast;
        this.yLast = yLast;
        this.maxTSLength = maxTSLength;
        addIArgument(iArgs());
        addTArgument(tArgs());
        addBArgument(bArgs(weights, maxTSLength, yLast, cLast));


        Preconditions.checkState(this.configuration.isRetLastH() || this.configuration.isRetLastC() || this.configuration.isRetFullSequence(),
                "You have to specify at least one output you want to return. Use isRetLastC, isRetLast and isRetFullSequence  methods  in LSTMLayerConfig builder to specify them");


    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {

        DataType dt = inputDataTypes.get(1);
        Preconditions.checkState(dt.isFPType(), "Input type 1 must be a floating point type, got %s", dt);
        ArrayList<DataType> list = new ArrayList<>();
        list.add(dt); // dLdx
        list.add(dt); // dLdWx
        list.add(dt); // dLdWr

        if (this.weights.hasBias()) {
            list.add(dt);
        } // dLdb

        if (this.maxTSLength != null) {
            list.add(dt);
        } // dLdSl
        if (this.yLast != null) {
            list.add(dt);
        } //dLdhI
        if (this.cLast != null) {
            list.add(dt);
        } // dLdcI
        if (this.weights.hasPH()) {
            list.add(dt);
        } // dLdWp

        return list;
    }


    @Override
    public String opName() {
        return "lstmLayer_bp";
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        return configuration.toProperties(true, true);
    }


    public long[] iArgs() {
        return new long[]{
                configuration.getLstmdataformat().ordinal(),// INT_ARG(0)
                configuration.getDirectionMode().ordinal(), // INT_ARG(1)
                configuration.getGateAct().ordinal(),  // INT_ARG(2)
                configuration.getOutAct().ordinal(), // INT_ARG(3)
                configuration.getCellAct().ordinal()  // INT_ARG(4)

        };
    }

    public double[] tArgs() {
        return new double[]{this.configuration.getCellClip()}; // T_ARG(0)
    }


    protected <T> boolean[] bArgs(LSTMLayerWeights weights, T maxTSLength, T yLast, T cLast) {
        return new boolean[]{
                weights.hasBias(),         // hasBiases: B_ARG(0)
                maxTSLength != null,         // hasSeqLen: B_ARG(1)
                yLast != null,               // hasInitH: B_ARG(2)
                cLast != null,              // hasInitC: B_ARG(3)
                weights.hasPH(),          // hasPH: B_ARG(4)
                configuration.isRetFullSequence(), //retFullSequence: B_ARG(5)
                configuration.isRetLastH(),  //  retLastH: B_ARG(6)
                configuration.isRetLastC()   // retLastC: B_ARG(7)
        };

    }

    @Override
    public boolean isConfigProperties() {
        return true;
    }

    @Override
    public String configFieldName() {
        return "configuration";
    }


    @Override
    public int getNumOutputs() {

        return Booleans.countTrue(
                true,
                true,
                true,
                weights.hasBias(),
                this.maxTSLength != null,
                this.yLast != null,
                this.cLast != null,
                weights.hasPH()
        );
    }


}


