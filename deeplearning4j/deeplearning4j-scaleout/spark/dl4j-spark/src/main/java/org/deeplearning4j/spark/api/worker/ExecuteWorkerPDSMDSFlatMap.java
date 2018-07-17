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

package org.deeplearning4j.spark.api.worker;

import org.apache.spark.input.PortableDataStream;
import org.datavec.spark.functions.FlatMapFunctionAdapter;
import org.datavec.spark.transform.BaseFlatMapFunctionAdaptee;
import org.deeplearning4j.spark.api.TrainingResult;
import org.deeplearning4j.spark.api.TrainingWorker;
import org.deeplearning4j.spark.iterator.PortableDataStreamMultiDataSetIterator;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.util.Iterator;

/**
 * A FlatMapFunction for executing training on serialized MultiDataSet objects, that can be loaded using a PortableDataStream
 * Used for SparkComputationGraph implementations only
 *
 * @author Alex Black
 */
public class ExecuteWorkerPDSMDSFlatMap<R extends TrainingResult>
                extends BaseFlatMapFunctionAdaptee<Iterator<PortableDataStream>, R> {

    public ExecuteWorkerPDSMDSFlatMap(TrainingWorker<R> worker) {
        super(new ExecuteWorkerPDSMDSFlatMapAdapter<>(worker));
    }
}


/**
 * A FlatMapFunction for executing training on serialized MultiDataSet objects, that can be loaded using a PortableDataStream
 * Used for SparkComputationGraph implementations only
 *
 * @author Alex Black
 */
class ExecuteWorkerPDSMDSFlatMapAdapter<R extends TrainingResult>
                implements FlatMapFunctionAdapter<Iterator<PortableDataStream>, R> {
    private final FlatMapFunctionAdapter<Iterator<MultiDataSet>, R> workerFlatMap;

    public ExecuteWorkerPDSMDSFlatMapAdapter(TrainingWorker<R> worker) {
        this.workerFlatMap = new ExecuteWorkerMultiDataSetFlatMapAdapter<>(worker);
    }

    @Override
    public Iterable<R> call(Iterator<PortableDataStream> iter) throws Exception {
        return workerFlatMap.call(new PortableDataStreamMultiDataSetIterator(iter));
    }
}
