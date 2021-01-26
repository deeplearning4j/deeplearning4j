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

package org.deeplearning4j.spark.parameterserver.functions;

import org.apache.commons.io.LineIterator;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.datavec.spark.util.SerializableHadoopConfig;
import org.deeplearning4j.core.loader.MultiDataSetLoader;
import org.deeplearning4j.spark.api.TrainingResult;
import org.deeplearning4j.spark.api.TrainingWorker;
import org.deeplearning4j.spark.iterator.PathSparkMultiDataSetIterator;
import org.deeplearning4j.spark.parameterserver.pw.SharedTrainingWrapper;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingResult;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingWorker;

import java.io.File;
import java.io.FileReader;
import java.util.Collections;
import java.util.Iterator;

/**
 * @author raver119@gmail.com
 */
public class SharedFlatMapPathsMDS<R extends TrainingResult> implements FlatMapFunction<Iterator<String>, R> {

    protected final SharedTrainingWorker worker;
    protected final MultiDataSetLoader loader;
    protected final Broadcast<SerializableHadoopConfig> hadoopConfig;

    public SharedFlatMapPathsMDS(TrainingWorker<R> worker, MultiDataSetLoader loader, Broadcast<SerializableHadoopConfig> hadoopConfig) {
        // we're not going to have anything but Shared classes here ever
        this.worker = (SharedTrainingWorker) worker;
        this.loader = loader;
        this.hadoopConfig = hadoopConfig;
    }

    @Override
    public Iterator<R> call(Iterator<String> dataSetIterator) throws Exception {
        //Under some limited circumstances, we might have an empty partition. In this case, we should return immediately
        if(!dataSetIterator.hasNext()){
            return Collections.emptyIterator();
        }
        // here we'll be converting out Strings coming out of iterator to DataSets
        // PathSparkDataSetIterator does that for us
        //For better fault tolerance, we'll pull all paths to a local file. This way, if the Iterator<String> is backed
        // by a remote source that later goes down, we won't fail (as long as the source is still available)
        File f = SharedFlatMapPaths.toTempFile(dataSetIterator);

        LineIterator lineIter = new LineIterator(new FileReader(f));    //Buffered reader added automatically
        try {
            // iterator should be silently attached to VirtualDataSetIterator, and used appropriately
            SharedTrainingWrapper.getInstance(worker.getInstanceId()).attachMDS(new PathSparkMultiDataSetIterator(lineIter, loader, hadoopConfig));

            // first callee will become master, others will obey and die
            SharedTrainingResult result = SharedTrainingWrapper.getInstance(worker.getInstanceId()).run(worker);

            return Collections.singletonList((R) result).iterator();
        } finally {
            lineIter.close();
            f.delete();
        }
    }
}
