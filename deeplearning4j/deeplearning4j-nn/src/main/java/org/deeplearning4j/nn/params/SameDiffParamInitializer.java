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

package org.deeplearning4j.nn.params;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.samediff.AbstractSameDiffLayer;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffVertex;
import org.deeplearning4j.nn.layers.samediff.SameDiffGraphVertex;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

@Slf4j
public class SameDiffParamInitializer implements ParamInitializer {

    private static final SameDiffParamInitializer INSTANCE = new SameDiffParamInitializer();

    public static SameDiffParamInitializer getInstance() {
        return INSTANCE;
    }

    @Override
    public long numParams(NeuralNetConfiguration conf) {
        return numParams(conf.getLayer());
    }

    @Override
    public long numParams(Layer layer) {
        AbstractSameDiffLayer sd = (AbstractSameDiffLayer)layer;
        Map<String,long[]> m = sd.getLayerParams().getParamShapes();
        int n = 0;
        for(val arr : m.values()){
            n += ArrayUtil.prod(arr);
        }
        return n;
    }

    @Override
    public List<String> paramKeys(Layer layer) {
        AbstractSameDiffLayer sd = (AbstractSameDiffLayer)layer;
        return sd.getLayerParams().getParameterKeys();
    }

    @Override
    public List<String> weightKeys(Layer layer) {
        AbstractSameDiffLayer sd = (AbstractSameDiffLayer)layer;
        return sd.getLayerParams().getWeightParameterKeys();
    }

    @Override
    public List<String> biasKeys(Layer layer) {
        AbstractSameDiffLayer sd = (AbstractSameDiffLayer)layer;
        return sd.getLayerParams().getBiasParameterKeys();
    }

    @Override
    public boolean isWeightParam(Layer layer, String key) {
        return weightKeys(layer).contains(key);
    }

    @Override
    public boolean isBiasParam(Layer layer, String key) {
        return biasKeys(layer).contains(key);
    }

    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        AbstractSameDiffLayer sd = (AbstractSameDiffLayer) conf.getLayer();
        Map<String,INDArray> out = subsetAndReshape(sd.getLayerParams().getParameterKeys(),
                sd.getLayerParams().getParamShapes(), paramsView, sd);
        if(initializeParams){
            sd.initializeParameters(out);
        }

        for(String s : sd.getLayerParams().getParameterKeys()){
            conf.addVariable(s);
        }

        return out;
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        AbstractSameDiffLayer sd = (AbstractSameDiffLayer) conf.getLayer();
        return subsetAndReshape(sd.getLayerParams().getParameterKeys(), sd.getLayerParams().getParamShapes(),
                gradientView, sd);
    }

    private Map<String,INDArray> subsetAndReshape(List<String> params, Map<String,long[]> paramShapes, INDArray view,
                                                  AbstractSameDiffLayer sdl){
        return subsetAndReshape(params, paramShapes, view, sdl, null);
    }

    public Map<String,INDArray> subsetAndReshape(List<String> params, Map<String,long[]> paramShapes, INDArray view,
                                                 AbstractSameDiffLayer sdl, SameDiffVertex sdv){
        Class<?> clazz = (sdl != null ? sdl.getClass() : sdv.getClass());
        String layerName = (sdl != null ? sdl.getLayerName() : ""); //TODO

        Map<String,INDArray> out = new LinkedHashMap<>();
        int soFar = 0;
        for(String s : params){
            val sh = paramShapes.get(s);
            val length = ArrayUtil.prodLong(sh);
            if(length <= 0){
                throw new IllegalStateException("Invalid array state for parameter \"" + s + "\" in layer " + layerName
                        + " of type " + clazz.getSimpleName() + ": parameter length (" + length
                        + ") must be > 0 - parameter array shape: " + Arrays.toString(sh));
            }
            INDArray sub = view.get(point(0), interval(soFar, soFar + length));

            if(!Arrays.equals(sub.shape(), sh)){
                char order = (sdl != null ? sdl.paramReshapeOrder(s) : sdv.paramReshapeOrder(s));
                sub = sub.reshape(order, sh);
            }
            out.put(s, sub);

            soFar += length;
        }
        return out;
    }
}
