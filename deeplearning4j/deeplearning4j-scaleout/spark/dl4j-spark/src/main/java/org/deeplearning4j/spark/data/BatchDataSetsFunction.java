/*-
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
 */

package org.deeplearning4j.spark.data;

import org.datavec.spark.functions.FlatMapFunctionAdapter;
import org.datavec.spark.transform.BaseFlatMapFunctionAdaptee;
import org.nd4j.linalg.dataset.DataSet;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Function used to batch DataSet objects together. Typically used to combine singe-example DataSet objects out of
 * something like {@link org.deeplearning4j.spark.datavec.DataVecDataSetFunction} together into minibatches.<br>
 *
 * Usage:
 * <pre>
 * {@code
 *      RDD<DataSet> mySingleExampleDataSets = ...;
 *      RDD<DataSet> batchData = mySingleExampleDataSets.mapPartitions(new BatchDataSetsFunction(batchSize));
 * }
 * </pre>
 *
 * @author Alex Black
 */
public class BatchDataSetsFunction extends BaseFlatMapFunctionAdaptee<Iterator<DataSet>, DataSet> {

    public BatchDataSetsFunction(int minibatchSize) {
        super(new BatchDataSetsFunctionAdapter(minibatchSize));
    }
}


/**
 * Function used to batch DataSet objects together. Typically used to combine singe-example DataSet objects out of
 * something like {@link org.deeplearning4j.spark.datavec.DataVecDataSetFunction} together into minibatches.<br>
 *
 * Usage:
 * <pre>
 * {@code
 *      RDD<DataSet> mySingleExampleDataSets = ...;
 *      RDD<DataSet> batchData = mySingleExampleDataSets.mapPartitions(new BatchDataSetsFunction(batchSize));
 * }
 * </pre>
 *
 * @author Alex Black
 */
class BatchDataSetsFunctionAdapter implements FlatMapFunctionAdapter<Iterator<DataSet>, DataSet> {
    private final int minibatchSize;

    public BatchDataSetsFunctionAdapter(int minibatchSize) {
        this.minibatchSize = minibatchSize;
    }

    @Override
    public Iterable<DataSet> call(Iterator<DataSet> iter) throws Exception {
        List<DataSet> out = new ArrayList<>();
        while (iter.hasNext()) {
            List<DataSet> list = new ArrayList<>();

            int count = 0;
            while (count < minibatchSize && iter.hasNext()) {
                DataSet ds = iter.next();
                count += ds.getFeatureMatrix().size(0);
                list.add(ds);
            }

            DataSet next;
            if (list.isEmpty())
                next = list.get(0);
            else
                next = DataSet.merge(list);

            out.add(next);
        }
        return out;
    }
}
