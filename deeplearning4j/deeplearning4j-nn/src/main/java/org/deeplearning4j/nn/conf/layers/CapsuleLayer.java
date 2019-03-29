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

package org.deeplearning4j.nn.conf.layers;

import java.util.Map;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InputType.InputTypeRecurrent;
import org.deeplearning4j.nn.conf.inputs.InputType.Type;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer;
import org.deeplearning4j.util.CapsuleUtils;
import org.deeplearning4j.util.ValidationUtils;
import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

//TODO short description
/**
 * An implementation of the DigiCaps layer from Dynamic Routing Between Capsules
 *
 * From <a href="http://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf">Dynamic Routing Between Capsules</a>
 *
 * @author Ryan Nett
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class CapsuleLayer extends SameDiffLayer {

    private static final String WEIGHT_PARAM = "weight";
    private static final String BIAS_PARAM = "bias";

    private boolean hasBias = false;
    private long inputCapsules = 0;
    private long inputCapsuleDimensions = 0;
    private int capsules;
    private int capsuleDimensions;
    private int routings;

    public CapsuleLayer(Builder builder){
        super(builder);
        this.hasBias = builder.hasBias;
        this.inputCapsules = builder.inputCapsules;
        this.inputCapsuleDimensions = builder.inputCapsuleDimensions;
        this.capsules = builder.capsules;
        this.capsuleDimensions = builder.capsuleDimensions;
        this.routings = builder.routings;

        if(capsules <= 0 || capsuleDimensions <= 0 || routings <= 0){
            throw new IllegalArgumentException("Invalid configuration for Capsule Layer (layer name = \""
                    + layerName + "\"):"
                    + " capsules, capsuleDimensions, and routings must be > 0.  Got: "
                    + capsules + ", " + capsuleDimensions + ", " + routings);
        }

        if(inputCapsules < 0 || inputCapsuleDimensions < 0){
            throw new IllegalArgumentException("Invalid configuration for Capsule Layer (layer name = \""
                    + layerName + "\"):"
                    + " inputCapsules and inputCapsuleDimensions must be >= 0 if set.  Got: "
                    + inputCapsules + ", " + inputCapsuleDimensions);
        }

    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if(inputType == null || inputType.getType() != Type.RNN) {
            throw new IllegalStateException("Invalid input for Capsule layer (layer name = \""
                    + layerName + "\"): expect RNN input.  Got: " + inputType);
        }

        if(inputCapsules <= 0 || inputCapsuleDimensions <= 0){
            InputType.InputTypeRecurrent ir = (InputTypeRecurrent) inputType;
            inputCapsules = ir.getSize();
            inputCapsuleDimensions = ir.getTimeSeriesLength();
        }

    }

    @Override
    public SDVariable defineLayer(SameDiff SD, SDVariable input, Map<String, SDVariable> paramTable) {
        SDVariable expanded = SD.expandDims(SD.expandDims(input, 2), 4);
        SDVariable tiled = SD.tile(expanded, new int[]{1, 1, capsules * capsuleDimensions, 1, 1});

        SDVariable weights = paramTable.get(WEIGHT_PARAM);
        SDVariable uHat = weights.times(tiled).sum(true, 3)
                .reshape(-1, inputCapsules, capsules, capsuleDimensions, 1);

        SDVariable b = SD.zerosLike(uHat).get(SDIndex.all(), SDIndex.all(), SDIndex.all(), SDIndex.interval(0, 1), SDIndex.interval(0, 1));

        //TODO convert to SameDiff.whileLoop?
        for(int i = 0 ; i < routings ; i++){
            SDVariable c = CapsuleUtils.softmax(SD, b, 2);

            SDVariable temp = c.times(uHat).sum(true, 1);
            if(hasBias){
                temp = temp.plus(paramTable.get(BIAS_PARAM));
            }

            SDVariable v = CapsuleUtils.squash(SD, temp);

            if(i == routings - 1){
                return v;
            }

            SDVariable vTiled = SD.tile(v, new int[]{1, (int) inputCapsules, 1, 1, 1});

            b = b.plus(uHat.times(vTiled).sum(true, 3));
        }

        return null; // will always return in the loop
    }

    @Override
    public void defineParameters(SDLayerParams params) {
        params.clear();
        params.addWeightParam(WEIGHT_PARAM,
                1, inputCapsules, capsules * capsuleDimensions, inputCapsuleDimensions, 1);

        if(hasBias){
            params.addBiasParam(BIAS_PARAM,
                    1, 1, capsules, capsuleDimensions, 1);
        }
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            for (Map.Entry<String, INDArray> e : params.entrySet()) {
                if (BIAS_PARAM.equals(e.getKey())) {
                    e.getValue().assign(0);
                } else if(WEIGHT_PARAM.equals(e.getKey())){
                    //TODO use weightInit
                    e.getValue().assign(0);
                }
            }
        }
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        return InputType.recurrent(capsules, capsuleDimensions);
    }

    //TODO builder
    @Getter
    @Setter
    public static class Builder extends SameDiffLayer.Builder<Builder>{

        private int capsules;
        private int capsuleDimensions;

        private int routings;

        private boolean hasBias = false;

        private int inputCapsules = 0;
        private int inputCapsuleDimensions = 0;

        public Builder(int capsules, int capsuleDimensions, int routings){
            super();
            this.setCapsules(capsules);
            this.setCapsuleDimensions(capsuleDimensions);
            this.setRoutings(routings);
        }

        @Override
        public <E extends Layer> E build() {
            return (E) new CapsuleLayer(this);
        }

        public Builder capsules(int capsules){
            this.setCapsules(capsules);
            return this;
        }

        public Builder capsuleDimensions(int capsuleDimensions){
            this.setCapsuleDimensions(capsuleDimensions);
            return this;
        }

        public Builder routings(int routings){
            this.setRoutings(routings);
            return this;
        }

        public Builder inputCapsules(int inputCapsules){
            this.setInputCapsules(inputCapsules);
            return this;
        }

        public Builder inputCapsuleDimensions(int inputCapsuleDimensions){
            this.setInputCapsuleDimensions(inputCapsuleDimensions);
            return this;
        }

        public Builder inputShape(int... inputShape){
            int[] input = ValidationUtils.validate2NonNegative(inputShape, false, "inputShape");
            this.setInputCapsules(input[0]);
            this.setInputCapsuleDimensions(input[1]);
            return this;
        }

        public Builder hasBias(boolean hasBias){
            this.setHasBias(hasBias);
            return this;
        }

    }
}
