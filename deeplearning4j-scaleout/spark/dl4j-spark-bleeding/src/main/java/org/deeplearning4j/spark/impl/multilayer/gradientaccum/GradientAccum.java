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

package org.deeplearning4j.spark.impl.multilayer.gradientaccum;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Runs iterative reduce on the dataset
 * @author Adam Gibson
 */
public class GradientAccum implements Function<DataSet, Gradient> {
  private static final Logger log = LoggerFactory.getLogger(GradientAccum.class);

  private String json;
  private Broadcast<INDArray> params;

  /**
   * Train and average over mini batches from a dataset
   * @param params the parameters that were broadcast
   * @param json   the configuration for the network
   */
  public GradientAccum(Broadcast<INDArray> params, String json) {
    this.params = params;
    this.json = json;
  }

  @Override
  public Gradient call(DataSet dataSet) throws Exception {
    log.info("Training on " + dataSet.numExamples());
    MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson(json);
    MultiLayerNetwork network = new MultiLayerNetwork(conf);
    network.init();
    network.setParameters(params.value());
    network.fit(dataSet);
    return network.gradient();
  }

}
