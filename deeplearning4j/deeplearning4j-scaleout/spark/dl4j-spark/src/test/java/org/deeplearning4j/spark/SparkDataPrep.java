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

package org.deeplearning4j.spark;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.FileBatchRecordReader;
import org.deeplearning4j.api.loader.DataSetLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.api.loader.FileBatch;
import org.nd4j.api.loader.Source;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

import java.io.IOException;

public class SparkDataPrep {



    public static class RecordReaderFileBatchDataSetLoader implements DataSetLoader {
        private final RecordReader recordReader;
        private final int batchSize;
        private final int labelIndexFrom;
        private final int labelIndexTo;
        private final int numPossibleLabels;
        private final boolean regression;
        private DataSetPreProcessor preProcessor;

        public RecordReaderFileBatchDataSetLoader(RecordReader recordReader, int batchSize, int labelIndex, int numClasses){
            this(recordReader, batchSize, labelIndex, labelIndex, numClasses, false, null);
        }

        public RecordReaderFileBatchDataSetLoader(RecordReader recordReader, int batchSize, int labelIndexFrom, int labelIndexTo,
                                                  int numPossibleLabels, boolean regression, DataSetPreProcessor preProcessor) {
            this.recordReader = recordReader;
            this.batchSize = batchSize;
            this.labelIndexFrom = labelIndexFrom;
            this.labelIndexTo = labelIndexTo;
            this.numPossibleLabels = numPossibleLabels;
            this.regression = regression;
            this.preProcessor = preProcessor;
        }

        @Override
        public DataSet load(Source source) throws IOException {
            FileBatch fb = FileBatch.readFromZip(source.getInputStream());

            //Wrap file batch in RecordReader
            //Create RecordReaderDataSetIterator
            //Return dataset
            RecordReader rr = new FileBatchRecordReader(recordReader, fb);
            RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(rr, null, batchSize, labelIndexFrom, labelIndexTo, numPossibleLabels, -1, regression);
            if(preProcessor != null){
                iter.setPreProcessor(preProcessor);
            }
            DataSet ds = iter.next();
            return ds;
        }
    }
}
