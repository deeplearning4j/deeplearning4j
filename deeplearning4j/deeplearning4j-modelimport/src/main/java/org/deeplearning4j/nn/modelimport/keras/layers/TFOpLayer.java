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

package org.deeplearning4j.nn.modelimport.keras.layers;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.modelimport.keras.layers.TFOpLayerImpl;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.regularization.Regularization;

import java.util.Collection;
import java.util.List;
import java.util.Map;


public class TFOpLayer extends Layer {

    private Map nodeDef;
    private Map constants;
    public TFOpLayer(Map nodeDef, Map constants){
        super();
        this.nodeDef = nodeDef;
        this.constants = constants;
    }

    @Override
    public ParamInitializer initializer() {
        return EmptyParamInitializer.getInstance();
    }
    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return null;
    }

    @Override
    public boolean isPretrainParam(String param){
        return false;
    }

    @Override
    public InputType getOutputType(int idx, InputType inputType){
        long[] shape = inputType.getShape(true);
        TFOpLayerImpl tempLayer = new TFOpLayerImpl(nodeDef, constants, null, null);
        long[] outputShape = tempLayer.getOutputShape(shape);
        if (outputShape.length == 3){
            return InputType.recurrent(outputShape[2], outputShape[1], RNNFormat.NWC);
        }
        return InputType.inferInputType(Nd4j.create(outputShape));

    }

    @Override
    public  void setNIn(InputType inputType, boolean override){}


    @Override
    public GradientNormalization getGradientNormalization(){return null;}


    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                                                                Collection<TrainingListener> trainingListeners, int layerIndex, INDArray layerParamsView,
                                                                boolean initializeParams, DataType networkDataType) {

        TFOpLayerImpl tfOpLayerImpl = new TFOpLayerImpl(nodeDef, constants, conf, networkDataType);
        tfOpLayerImpl.setListeners(trainingListeners);
        tfOpLayerImpl.setIndex(layerIndex);
        return tfOpLayerImpl;
    }

    @Override
    public double getGradientNormalizationThreshold(){return 0.;}

    @Override
    public List<Regularization> getRegularizationByParam(String paramName){return null;}

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        return new LayerMemoryReport(); //TODO
    }





}
