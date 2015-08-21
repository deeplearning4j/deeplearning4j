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

import static org.junit.Assume.*;

import org.canova.api.conf.Configuration;
import org.deeplearning4j.datasets.fetchers.IrisDataFetcher;
import org.deeplearning4j.nn.layers.feedforward.rbm.RBM;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.DeepLearningConfigurable;
import org.deeplearning4j.scaleout.job.Job;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class NeuralNetWorkPerformerTest extends BaseWorkPerformerTest {
    @Test
    public void testRbm() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration
                .Builder()
                .layer(new org.deeplearning4j.nn.conf.layers.RBM.Builder().nIn(4).nOut(3).build())
                .build();
        String json = conf.toJson();
        NeuralNetConfiguration conf3 = NeuralNetConfiguration.fromJson(json);
        assumeNotNull(conf3);

        Configuration conf2 = new Configuration();
        conf2.set(DeepLearningConfigurable.NEURAL_NET_CONF,json);
        conf2.set(DeepLearningConfigurable.CLASS,RBM.class.getName());
        WorkerPerformer performer = new NeuralNetWorkPerformer();
        performer.setup(conf2);
        IrisDataFetcher fetcher = new IrisDataFetcher();
        fetcher.fetch(10);
        DataSet d = fetcher.next();
        Job j = new Job(d,"1");
        assumeJobResultNotNull(performer,j);
        performer.update(j.getResult());
    }


}
