/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.spark.impl.graph.scoring;

import org.apache.spark.broadcast.Broadcast;
import org.datavec.spark.functions.FlatMapFunctionAdapter;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.spark.util.BasePairFlatMapFunctionAdaptee;
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

/**
 * Function to feed-forward examples, and get the network output (for example, class probabilities).
 * A key value is used to keep track of which output corresponds to which input.
 *
 * @param <K> Type of key, associated with each example. Used to keep track of which output belongs to which input example
 * @author Alex Black
 */
public class GraphFeedForwardWithKeyFunction<K>
                extends BasePairFlatMapFunctionAdaptee<Iterator<Tuple2<K, INDArray[]>>, K, INDArray[]> {

    public GraphFeedForwardWithKeyFunction(Broadcast<INDArray> params, Broadcast<String> jsonConfig, int batchSize) {
        super(new GraphFeedForwardWithKeyFunctionAdapter<K>(params, jsonConfig, batchSize));
    }
}


/**
 * Function to feed-forward examples, and get the network output (for example, class probabilities).
 * A key value is used to keey track of which output corresponds to which input.
 *
 * @param <K> Type of key, associated with each example. Used to keep track of which output belongs to which input example
 * @author Alex Black
 */
class GraphFeedForwardWithKeyFunctionAdapter<K>
                implements FlatMapFunctionAdapter<Iterator<Tuple2<K, INDArray[]>>, Tuple2<K, INDArray[]>> {

    protected static Logger log = LoggerFactory.getLogger(GraphFeedForwardWithKeyFunction.class);

    private final Broadcast<INDArray> params;
    private final Broadcast<String> jsonConfig;
    private final int batchSize;

    /**
     * @param params     MultiLayerNetwork parameters
     * @param jsonConfig MultiLayerConfiguration, as json
     * @param batchSize  Batch size to use for forward pass (use > 1 for efficiency)
     */
    public GraphFeedForwardWithKeyFunctionAdapter(Broadcast<INDArray> params, Broadcast<String> jsonConfig,
                    int batchSize) {
        this.params = params;
        this.jsonConfig = jsonConfig;
        this.batchSize = batchSize;
    }


    @Override
    public Iterable<Tuple2<K, INDArray[]>> call(Iterator<Tuple2<K, INDArray[]>> iterator) throws Exception {
        if (!iterator.hasNext()) {
            return Collections.emptyList();
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
        List<Integer> origSizeList = new ArrayList<>();

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

            // FIXME: int cast
            origSizeList.add((int) t2._2()[0].size(0));
            tupleCount++;
        }

        if (tupleCount == 0) {
            return Collections.emptyList();
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
                int numExamples = origSizeList.get(i);
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

        return output;
    }

    private INDArray getSubset(int exampleStart, int exampleEnd, INDArray from) {
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
