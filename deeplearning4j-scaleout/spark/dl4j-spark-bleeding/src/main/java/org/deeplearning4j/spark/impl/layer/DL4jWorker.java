/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.spark.impl.layer;

import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

/**
 * This is considered the "Worker"
 * This is the code that will run the .fitDataSet() method on the network
 *
 * the issue here is that this is getting called 1x per record
 * and before we could call it in a more controlled mini-batch setting
 *
 * @author josh
 * @author Adam Gibson
 */
public class DL4jWorker implements Function<DataSet, INDArray> {

    private final Model network;

    public DL4jWorker(String json,INDArray params) {
        NeuralNetConfiguration conf = NeuralNetConfiguration.fromJson(json);
        LayerFactory layerFactory = LayerFactories.getFactory(conf.getLayer());
        if(layerFactory == null)
            throw new IllegalStateException("Please specify a layer factory");
        this.network = layerFactory.create(conf);
        int numParams = this.network.numParams();
        if(numParams != params.length())
            throw new IllegalStateException("Number of params for configured network was " + numParams + " while the specified parameter vector length was " + params.length());
        Layer network = (Layer) this.network;
        network.setParams(params);
    }

    @Override
    public INDArray call(DataSet v1) throws Exception {
        try {
            Layer network = (Layer) this.network;
            if(network instanceof OutputLayer) {
                OutputLayer o = (OutputLayer) network;
                o.fit(v1);
            }
            else
                network.fit(v1.getFeatureMatrix());
            return network.params();
        }catch(Exception e) {
            System.err.println("Error with dataset " + v1.numExamples());
            throw e;
        }

    }
}