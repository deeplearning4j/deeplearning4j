/*
 *
 *  * Copyright 2016 Skymind,Inc.
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

package org.deeplearning4j.spark.impl.multilayer.scoring;

import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**Function to score examples individually, where each example is associated with a particular key<br>
 * Note that scoring is batched for computational efficiency.<br>
 * This is the Spark implementation of t he {@link MultiLayerNetwork#scoreExamples(DataSet, boolean)} method<br>
 * <b>Note:</b> The DataSet objects passed in must have exactly one example in them (otherwise: can't have a 1:1 association
 * between keys and data sets to score)
 * @author Alex Black
 * @param <K> Type of key, associated with each example. Used to keep track of which score belongs to which example
 * @see ScoreExamplesFunction
 */
public class ScoreExamplesWithKeyFunction<K> implements PairFlatMapFunction<Iterator<Tuple2<K,DataSet>>,K,Double> {

    protected static Logger log = LoggerFactory.getLogger(ScoreExamplesWithKeyFunction.class);

    private final Broadcast<INDArray> params;
    private final Broadcast<String> jsonConfig;
    private final boolean addRegularization;
    private final int batchSize;

    /**
     * @param params MultiLayerNetwork parameters
     * @param jsonConfig MultiLayerConfiguration, as json
     * @param addRegularizationTerms if true: add regulari
     * @param batchSize
     */
    public ScoreExamplesWithKeyFunction(Broadcast<INDArray> params, Broadcast<String> jsonConfig, boolean addRegularizationTerms,
                                        int batchSize){
        this.params = params;
        this.jsonConfig = jsonConfig;
        this.addRegularization = addRegularizationTerms;
        this.batchSize = batchSize;
    }


    @Override
    public Iterable<Tuple2<K, Double>> call(Iterator<Tuple2<K, DataSet>> iterator) throws Exception {
        if (!iterator.hasNext()) {
            return Collections.emptyList();
        }

        MultiLayerNetwork network = new MultiLayerNetwork(MultiLayerConfiguration.fromJson(jsonConfig.getValue()));
        network.init();
        INDArray val = params.value();
        if (val.length() != network.numParams(false))
            throw new IllegalStateException("Network did not have same number of parameters as the broadcasted set parameters");
        network.setParameters(val);

        List<Tuple2<K,Double>> ret = new ArrayList<>();

        List<DataSet> collect = new ArrayList<>(batchSize);
        List<K> collectKey = new ArrayList<>(batchSize);
        int totalCount = 0;
        while (iterator.hasNext()) {
            collect.clear();
            collectKey.clear();
            int nExamples = 0;
            while (iterator.hasNext() && nExamples < batchSize) {
                Tuple2<K,DataSet> t2 = iterator.next();
                DataSet ds = t2._2();
                int n = ds.numExamples();
                if(n != 1) throw new IllegalStateException("Cannot score examples with one key per data set if "
                    + "data set contains more than 1 example (numExamples: " + n + ")");
                collect.add(ds);
                collectKey.add(t2._1());
                nExamples += n;
            }
            totalCount += nExamples;

            DataSet data = DataSet.merge(collect, false);


            INDArray scores = network.scoreExamples(data,addRegularization);
            double[] doubleScores = scores.data().asDouble();

            for(int i=0; i<doubleScores.length; i++ ){
                ret.add(new Tuple2<>(collectKey.get(i),doubleScores[i]));
            }
        }

        if (log.isDebugEnabled()) {
            log.debug("Scored {} examples ", totalCount);
        }

        return ret;
    }
}
