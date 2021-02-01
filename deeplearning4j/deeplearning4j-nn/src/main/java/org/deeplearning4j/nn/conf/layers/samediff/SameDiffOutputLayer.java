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

package org.deeplearning4j.nn.conf.layers.samediff;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

public abstract class SameDiffOutputLayer extends AbstractSameDiffLayer {


    protected SameDiffOutputLayer() {
        //No op constructor for Jackson
    }

    /**
     * Define the output layer
     * @param sameDiff   SameDiff instance
     * @param layerInput Input to the layer
     * @param labels     Labels variable (or null if {@link #labelsRequired()} returns false
     * @param paramTable Parameter table - keys as defined by {@link #defineParameters(SDLayerParams)}
     * @return The final layer variable corresponding to the score/loss during forward pass. This must be a single scalar value.
     */
    public abstract SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput, SDVariable labels,
                    Map<String, SDVariable> paramTable);

    /**
     * Output layers should terminate in a single scalar value (i.e., a score) - however, sometimes the output activations
     * (such as softmax probabilities) need to be returned. When this is the case, we need to know the name of the
     * SDVariable that corresponds to these.<br>
     * If the final network activations are just the input to the layer, simply return "input"
     *
     * @return The name of the activations to return when performing forward pass
     */
    public abstract String activationsVertexName();

    /**
     * Whether labels are required for calculating the score. Defaults to true - however, if the score
     * can be calculated without labels (for example, in some output layers used for unsupervised learning)
     * this can be set to false.
     * @return True if labels are required to calculate the score/output, false otherwise.
     */
    public boolean labelsRequired() {
        return true;
    }

    //==================================================================================================================

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                                                       Collection<TrainingListener> trainingListeners, int layerIndex, INDArray layerParamsView,
                                                       boolean initializeParams, DataType networkDataType) {
        org.deeplearning4j.nn.layers.samediff.SameDiffOutputLayer ret =
                        new org.deeplearning4j.nn.layers.samediff.SameDiffOutputLayer(conf, networkDataType);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

}
