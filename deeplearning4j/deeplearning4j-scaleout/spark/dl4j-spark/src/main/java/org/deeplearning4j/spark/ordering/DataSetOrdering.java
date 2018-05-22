/*-
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

package org.deeplearning4j.spark.ordering;

import org.nd4j.linalg.dataset.DataSet;
import scala.Function1;
import scala.Some;
import scala.math.Ordering;

/**
 * Orders by data set size.
 * This will force the dataset with a certain number of mini batches to be grouped at th end.
 */
public class DataSetOrdering implements Ordering<DataSet> {
    @Override
    public Some<Object> tryCompare(DataSet dataSet, DataSet t1) {
        return null;
    }

    @Override
    public int compare(DataSet dataSet, DataSet t1) {
        return 0;
    }

    @Override
    public boolean lteq(DataSet dataSet, DataSet t1) {
        return dataSet.numExamples() >= t1.numExamples();
    }

    @Override
    public boolean gteq(DataSet dataSet, DataSet t1) {
        return !lteq(dataSet, t1);
    }

    @Override
    public boolean lt(DataSet dataSet, DataSet t1) {
        return dataSet.numExamples() >= t1.numExamples();
    }

    @Override
    public boolean gt(DataSet dataSet, DataSet t1) {
        return !lt(dataSet, t1);
    }

    @Override
    public boolean equiv(DataSet dataSet, DataSet t1) {
        return dataSet.numExamples() == t1.numExamples();
    }

    @Override
    public DataSet max(DataSet dataSet, DataSet t1) {
        return gt(dataSet, t1) ? dataSet : t1;
    }

    @Override
    public DataSet min(DataSet dataSet, DataSet t1) {
        return max(dataSet, t1) == dataSet ? t1 : dataSet;
    }

    @Override
    public Ordering<DataSet> reverse() {
        return null;
    }

    @Override
    public <U> Ordering<U> on(Function1<U, DataSet> function1) {
        return null;
    }

    @Override
    public Ops mkOrderingOps(DataSet dataSet) {
        return null;
    }
}
