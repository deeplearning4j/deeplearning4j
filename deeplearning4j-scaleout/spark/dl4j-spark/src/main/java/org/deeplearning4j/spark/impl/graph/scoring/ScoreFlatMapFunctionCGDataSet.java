/*-
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

package org.deeplearning4j.spark.impl.graph.scoring;

import org.apache.spark.broadcast.Broadcast;
import org.datavec.spark.functions.FlatMapFunctionAdapter;
import org.datavec.spark.transform.BaseFlatMapFunctionAdaptee;
import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;
import lombok.val;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/** Function used to score a DataSet using a ComputationGraph */
public class ScoreFlatMapFunctionCGDataSet
                extends BaseFlatMapFunctionAdaptee<Iterator<DataSet>, Tuple2<Integer, Double>> {

    public ScoreFlatMapFunctionCGDataSet(String json, Broadcast<INDArray> params, int minibatchSize) {
        super(new ScoreFlatMapFunctionCGDataSetAdapter(json, params, minibatchSize));
    }
}


/** Function used to score a DataSet using a ComputationGraph */
class ScoreFlatMapFunctionCGDataSetAdapter
                implements FlatMapFunctionAdapter<Iterator<DataSet>, Tuple2<Integer, Double>> {

    private static final Logger log = LoggerFactory.getLogger(ScoreFlatMapFunctionCGDataSet.class);
    private String json;
    private Broadcast<INDArray> params;
    private int minibatchSize;


    public ScoreFlatMapFunctionCGDataSetAdapter(String json, Broadcast<INDArray> params, int minibatchSize) {
        this.json = json;
        this.params = params;
        this.minibatchSize = minibatchSize;
    }

    @Override
    public Iterable<Tuple2<Integer, Double>> call(Iterator<DataSet> dataSetIterator) throws Exception {
        if (!dataSetIterator.hasNext()) {
            return Collections.singletonList(new Tuple2<>(0, 0.0));
        }

        DataSetIterator iter = new IteratorDataSetIterator(dataSetIterator, minibatchSize); //Does batching where appropriate

        ComputationGraph network = new ComputationGraph(ComputationGraphConfiguration.fromJson(json));
        network.init();
        INDArray val = params.value().unsafeDuplication(); //.value() is shared by all executors on single machine -> OK, as params are not changed in score function
        if (val.length() != network.numParams(false))
            throw new IllegalStateException(
                            "Network did not have same number of parameters as the broadcast set parameters");
        network.setParams(val);

        List<Tuple2<Integer, Double>> out = new ArrayList<>();
        while (iter.hasNext()) {
            DataSet ds = iter.next();
            double score = network.score(ds, false);

            // FIXME: int cast
            int numExamples = (int) ds.getFeatureMatrix().size(0);
            out.add(new Tuple2<>(numExamples, score * numExamples));
        }

        Nd4j.getExecutioner().commit();

        return out;
    }
}
