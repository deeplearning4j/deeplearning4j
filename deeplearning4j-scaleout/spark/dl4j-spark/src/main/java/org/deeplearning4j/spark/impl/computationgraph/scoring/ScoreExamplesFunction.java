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

package org.deeplearning4j.spark.impl.computationgraph.scoring;

import org.apache.spark.api.java.function.DoubleFlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**Function to score examples individually. Note that scoring is batched for computational efficiency.<br>
 * This is essentially a Spark implementation of the {@link ComputationGraph#scoreExamples(MultiDataSet, boolean)} method<br>
 * <b>Note:</b> This method returns a score for each example, but the association between examples and scores is lost. In
 * cases where we need to know the score for particular examples, use {@link ScoreExamplesWithKeyFunction}
 * @author Alex Black
 * @see ScoreExamplesWithKeyFunction
 */
public class ScoreExamplesFunction implements DoubleFlatMapFunction<Iterator<MultiDataSet>> {
    protected static final Logger log = LoggerFactory.getLogger(ScoreExamplesFunction.class);

    private final Broadcast<INDArray> params;
    private final Broadcast<String> jsonConfig;
    private final boolean addRegularization;
    private final int batchSize;

    public ScoreExamplesFunction(Broadcast<INDArray> params, Broadcast<String> jsonConfig, boolean addRegularizationTerms,
                                 int batchSize){
        this.params = params;
        this.jsonConfig = jsonConfig;
        this.addRegularization = addRegularizationTerms;
        this.batchSize = batchSize;
    }


    @Override
    public Iterable<Double> call(Iterator<MultiDataSet> iterator) throws Exception {
        if (!iterator.hasNext()) {
            return Collections.emptyList();
        }

        ComputationGraph network = new ComputationGraph(ComputationGraphConfiguration.fromJson(jsonConfig.getValue()));
        network.init();
        INDArray val = params.value();
        if (val.length() != network.numParams(false))
            throw new IllegalStateException("Network did not have same number of parameters as the broadcast set parameters");
        network.setParams(val);

        List<Double> ret = new ArrayList<>();

        List<MultiDataSet> collect = new ArrayList<>(batchSize);
        int totalCount = 0;
        while (iterator.hasNext()) {
            collect.clear();
            int nExamples = 0;
            while (iterator.hasNext() && nExamples < batchSize) {
                MultiDataSet ds = iterator.next();
                int n = ds.getFeatures(0).size(0);
                collect.add(ds);
                nExamples += n;
            }
            totalCount += nExamples;


            MultiDataSet data = org.nd4j.linalg.dataset.MultiDataSet.merge(collect);


            INDArray scores = network.scoreExamples(data,addRegularization);
            double[] doubleScores = scores.data().asDouble();

            for (double doubleScore : doubleScores) {
                ret.add(doubleScore);
            }
        }

        if (log.isDebugEnabled()) {
            log.debug("Scored {} examples ", totalCount);
        }

        return ret;
    }
}
