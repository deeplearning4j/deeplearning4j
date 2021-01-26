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

package org.deeplearning4j.nn.conf.layers.wrapper;

import lombok.Data;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.params.WrapperLayerParamInitializer;
import org.nd4j.linalg.learning.regularization.Regularization;

import java.util.List;

/**
 * Base wrapper layer: the idea is to pass through all methods to the underlying layer, and selectively override
 * them as required. This is to save implementing every single passthrough method for all 'wrapper' layer subtypes
 *
 * @author Alex Black
 */
@Data
public abstract class BaseWrapperLayer extends Layer {

    protected Layer underlying;

    protected BaseWrapperLayer(Builder builder) {
        super(builder);
    }

    protected BaseWrapperLayer() {}

    public BaseWrapperLayer(Layer underlying) {
        this.underlying = underlying;
    }

    @Override
    public ParamInitializer initializer() {
        return WrapperLayerParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        return underlying.getOutputType(layerIndex, inputType);
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        underlying.setNIn(inputType, override);
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return underlying.getPreProcessorForInputType(inputType);
    }

    @Override
    public List<Regularization> getRegularizationByParam(String paramName){
        return underlying.getRegularizationByParam(paramName);
    }

    @Override
    public GradientNormalization getGradientNormalization() {
        return underlying.getGradientNormalization();
    }

    @Override
    public double getGradientNormalizationThreshold() {
        return underlying.getGradientNormalizationThreshold();
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return underlying.isPretrainParam(paramName);
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        return underlying.getMemoryReport(inputType);
    }

    @Override
    public void setLayerName(String layerName) {
        super.setLayerName(layerName);
        if (underlying != null) {
            //May be null at some points during JSON deserialization
            underlying.setLayerName(layerName);
        }
    }
}
