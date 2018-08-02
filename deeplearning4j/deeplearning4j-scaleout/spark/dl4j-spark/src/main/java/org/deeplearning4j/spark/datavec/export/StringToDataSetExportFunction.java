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

package org.deeplearning4j.spark.datavec.export;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.function.VoidFunction;
import org.datavec.api.io.converters.SelfWritableConverter;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.split.StringSplit;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.util.UIDProvider;
import org.nd4j.linalg.dataset.DataSet;

import java.net.URI;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * A function (used in forEachPartition) to convert Strings to DataSet objects using a RecordReader (such as a CSVRecordReader).
 * Use with {@code JavaRDD<String>.foreachPartition()}
 *
 * @author Alex Black
 */
public class StringToDataSetExportFunction implements VoidFunction<Iterator<String>> {

    private static final Configuration conf = new Configuration();

    private final URI outputDir;
    private final RecordReader recordReader;
    private final int batchSize;
    private final boolean regression;
    private final int labelIndex;
    private final int numPossibleLabels;
    private String uid = null;

    private int outputCount;

    public StringToDataSetExportFunction(URI outputDir, RecordReader recordReader, int batchSize, boolean regression,
                    int labelIndex, int numPossibleLabels) {
        this.outputDir = outputDir;
        this.recordReader = recordReader;
        this.batchSize = batchSize;
        this.regression = regression;
        this.labelIndex = labelIndex;
        this.numPossibleLabels = numPossibleLabels;
    }

    @Override
    public void call(Iterator<String> stringIterator) throws Exception {
        String jvmuid = UIDProvider.getJVMUID();
        uid = Thread.currentThread().getId() + jvmuid.substring(0, Math.min(8, jvmuid.length()));

        List<List<Writable>> list = new ArrayList<>(batchSize);

        while (stringIterator.hasNext()) {
            String next = stringIterator.next();
            recordReader.initialize(new StringSplit(next));
            list.add(recordReader.next());

            processBatchIfRequired(list, !stringIterator.hasNext());
        }
    }

    private void processBatchIfRequired(List<List<Writable>> list, boolean finalRecord) throws Exception {
        if (list.isEmpty())
            return;
        if (list.size() < batchSize && !finalRecord)
            return;

        RecordReader rr = new CollectionRecordReader(list);
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(rr, null, batchSize, labelIndex, labelIndex, numPossibleLabels, -1, regression);

        DataSet ds = iter.next();

        String filename = "dataset_" + uid + "_" + (outputCount++) + ".bin";

        URI uri = new URI(outputDir.getPath() + "/" + filename);
        FileSystem file = FileSystem.get(uri, conf);
        try (FSDataOutputStream out = file.create(new Path(uri))) {
            ds.save(out);
        }

        list.clear();
    }
}
