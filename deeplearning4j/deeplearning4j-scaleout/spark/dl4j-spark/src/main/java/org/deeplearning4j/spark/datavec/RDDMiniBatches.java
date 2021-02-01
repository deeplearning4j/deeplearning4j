/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.spark.datavec;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.nd4j.linalg.dataset.DataSet;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * RDD mini batch partitioning
 * @author Adam Gibson
 */
public class RDDMiniBatches implements Serializable {
    private int miniBatches;
    private JavaRDD<DataSet> toSplitJava;

    public RDDMiniBatches(int miniBatches, JavaRDD<DataSet> toSplit) {
        this.miniBatches = miniBatches;
        this.toSplitJava = toSplit;
    }

    public JavaRDD<DataSet> miniBatchesJava() {
        //need a new mapping function, doesn't handle mini batches properly
        return toSplitJava.mapPartitions(new MiniBatchFunction(miniBatches));
    }

    @AllArgsConstructor
    public static class MiniBatchFunction implements FlatMapFunction<Iterator<DataSet>, DataSet> {
        private int batchSize;

        @Override
        public Iterator<DataSet> call(Iterator<DataSet> dataSetIterator) throws Exception {
            List<DataSet> ret = new ArrayList<>();
            List<DataSet> temp = new ArrayList<>();
            while (dataSetIterator.hasNext()) {
                temp.add(dataSetIterator.next().copy());
                if (temp.size() == batchSize) {
                    ret.add(DataSet.merge(temp));
                    temp.clear();
                }
            }

            //Add remaining ('left over') data
            if (temp.size() > 0)
                ret.add(DataSet.merge(temp));

            return ret.iterator();
        }
    }
}
