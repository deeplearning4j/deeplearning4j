/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.spark.impl.graph.scoring;

import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

@Slf4j
@AllArgsConstructor
public class GraphFeedForwardWithKeyFunction<K> implements PairFlatMapFunction<Iterator<Tuple2<K, INDArray[]>>, K, INDArray[]> {

    private final Broadcast<INDArray> params;
    private final Broadcast<String> jsonConfig;
    private final int batchSize;


    @Override
    public Iterator<Tuple2<K, INDArray[]>> call(Iterator<Tuple2<K, INDArray[]>> iterator) throws Exception {
        if (!iterator.hasNext()) {
            return Collections.emptyIterator();
        }

        ComputationGraph network = new ComputationGraph(ComputationGraphConfiguration.fromJson(jsonConfig.getValue()));
        network.init();
        INDArray val = params.value().unsafeDuplication();
        if (val.length() != network.numParams(false))
            throw new IllegalStateException(
                            "Network did not have same number of parameters as the broadcast set parameters");
        network.setParams(val);

        //Issue: for 2d data (MLPs etc) we can just stack the examples.
        //But: for 3d and 4d: in principle the data sizes could be different
        //We could handle that with mask arrays - but it gets messy. The approach used here is simpler but less efficient

        List<INDArray[]> featuresList = new ArrayList<>(batchSize);
        List<K> keyList = new ArrayList<>(batchSize);
        List<Long> origSizeList = new ArrayList<>();

        long[][] firstShapes = null;
        boolean sizesDiffer = false;
        int tupleCount = 0;
        while (iterator.hasNext()) {
            Tuple2<K, INDArray[]> t2 = iterator.next();
            if (firstShapes == null) {
                firstShapes = new long[t2._2().length][0];
                for (int i = 0; i < firstShapes.length; i++) {
                    firstShapes[i] = t2._2()[i].shape();
                }
            } else if (!sizesDiffer) {
                for (int i = 0; i < firstShapes.length; i++) {
                    for (int j = 1; j < firstShapes[i].length; j++) {
                        if (firstShapes[i][j] != featuresList.get(tupleCount - 1)[i].size(j)) {
                            sizesDiffer = true;
                            break;
                        }
                    }
                }
            }
            featuresList.add(t2._2());
            keyList.add(t2._1());

            origSizeList.add(t2._2()[0].size(0));
            tupleCount++;
        }

        if (tupleCount == 0) {
            return Collections.emptyIterator();
        }

        List<Tuple2<K, INDArray[]>> output = new ArrayList<>(tupleCount);
        int currentArrayIndex = 0;

        while (currentArrayIndex < featuresList.size()) {
            int firstIdx = currentArrayIndex;
            int nextIdx = currentArrayIndex;
            int examplesInBatch = 0;
            List<INDArray[]> toMerge = new ArrayList<>();
            firstShapes = null;
            while (nextIdx < featuresList.size() && examplesInBatch < batchSize) {
                INDArray[] f = featuresList.get(nextIdx);
                if (firstShapes == null) {
                    firstShapes = new long[f.length][0];
                    for (int i = 0; i < firstShapes.length; i++) {
                        firstShapes[i] = f[i].shape();
                    }
                } else if (sizesDiffer) {
                    boolean breakWhile = false;
                    for (int i = 0; i < firstShapes.length; i++) {
                        for (int j = 1; j < firstShapes[i].length; j++) {
                            if (firstShapes[i][j] != featuresList.get(nextIdx)[i].size(j)) {
                                //Next example has a different size. So: don't add it to the current batch, just process what we have
                                breakWhile = true;
                                break;
                            }
                        }
                    }
                    if (breakWhile) {
                        break;
                    }
                }

                toMerge.add(f);
                examplesInBatch += f[0].size(0);
                nextIdx++;
            }

            INDArray[] batchFeatures = new INDArray[toMerge.get(0).length];
            for (int i = 0; i < batchFeatures.length; i++) {
                INDArray[] tempArr = new INDArray[toMerge.size()];
                for (int j = 0; j < tempArr.length; j++) {
                    tempArr[j] = toMerge.get(j)[i];
                }
                batchFeatures[i] = Nd4j.concat(0, tempArr);
            }


            INDArray[] out = network.output(false, batchFeatures);

            examplesInBatch = 0;
            for (int i = firstIdx; i < nextIdx; i++) {
                long numExamples = origSizeList.get(i);
                INDArray[] outSubset = new INDArray[out.length];
                for (int j = 0; j < out.length; j++) {
                    outSubset[j] = getSubset(examplesInBatch, examplesInBatch + numExamples, out[j]);
                }
                examplesInBatch += numExamples;

                output.add(new Tuple2<>(keyList.get(i), outSubset));
            }

            currentArrayIndex += (nextIdx - firstIdx);
        }

        Nd4j.getExecutioner().commit();

        return output.iterator();
    }

    private INDArray getSubset(long exampleStart, long exampleEnd, INDArray from) {
        switch (from.rank()) {
            case 2:
                return from.get(NDArrayIndex.interval(exampleStart, exampleEnd), NDArrayIndex.all());
            case 3:
                return from.get(NDArrayIndex.interval(exampleStart, exampleEnd), NDArrayIndex.all(),
                                NDArrayIndex.all());
            case 4:
                return from.get(NDArrayIndex.interval(exampleStart, exampleEnd), NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.all());
            default:
                throw new RuntimeException("Invalid rank: " + from.rank());
        }
    }
}
