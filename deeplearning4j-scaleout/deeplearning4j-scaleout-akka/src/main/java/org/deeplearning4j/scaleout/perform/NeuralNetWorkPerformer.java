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

package org.deeplearning4j.scaleout.perform;

import org.canova.api.conf.Configuration;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.scaleout.job.Job;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;

/**
 * Neural network work performer
 * @author Adam Gibson
 */
public class NeuralNetWorkPerformer implements WorkerPerformer {
    protected Layer neuralNetwork;

    public NeuralNetWorkPerformer() {

    }

    @Override
    public void perform(Job job) {
        Serializable work = job.getWork();
        if(work instanceof DataSet) {
            DataSet data = (DataSet) work;
            neuralNetwork.fit(data.getFeatureMatrix());
        }
        else if(work instanceof INDArray) {
            neuralNetwork.fit((INDArray) work);
        }

        job.setResult(neuralNetwork.params());


    }

    @Override
    public void update(Object... o) {
        INDArray arr = (INDArray) o[0];
        neuralNetwork.setParams(arr);

    }

    @Override
    public void setup(Configuration conf) {
        NeuralNetConfiguration conf2 = NeuralNetConfiguration.fromJson(conf.get(NEURAL_NET_CONF));
        LayerFactory layerFactory = LayerFactories.getFactory(conf2.getLayer());
        int numParams = layerFactory.initializer().numParams(conf2,true);
        INDArray params = Nd4j.create(1, numParams);
        this.neuralNetwork = LayerFactories.getFactory(conf2.getLayer()).create(conf2, null, 0, params);
    }
}
