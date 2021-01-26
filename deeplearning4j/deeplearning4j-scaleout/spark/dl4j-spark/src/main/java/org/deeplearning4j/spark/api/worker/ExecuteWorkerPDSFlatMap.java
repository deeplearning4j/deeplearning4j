/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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

package org.deeplearning4j.spark.api.worker;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.input.PortableDataStream;
import org.deeplearning4j.spark.api.TrainingResult;
import org.deeplearning4j.spark.api.TrainingWorker;
import org.deeplearning4j.spark.iterator.PortableDataStreamDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;

import java.util.Iterator;

/**
 * A FlatMapFunction for executing training on serialized DataSet objects, that can be loaded using a PortableDataStream
 * Used in both SparkDl4jMultiLayer and SparkComputationGraph implementations
 *
 * @author Alex Black
 */
@Deprecated
public class ExecuteWorkerPDSFlatMap<R extends TrainingResult> implements FlatMapFunction<Iterator<PortableDataStream>, R> {
    private final FlatMapFunction<Iterator<DataSet>, R> workerFlatMap;

    public ExecuteWorkerPDSFlatMap(TrainingWorker<R> worker) {
        this.workerFlatMap = new ExecuteWorkerFlatMap<>(worker);
    }

    @Override
    public Iterator<R> call(Iterator<PortableDataStream> iter) throws Exception {
        return workerFlatMap.call(new PortableDataStreamDataSetIterator(iter));
    }
}
