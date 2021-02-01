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

package org.deeplearning4j.core.loader.impl;

import lombok.Getter;
import lombok.Setter;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.filebatch.FileBatchRecordReader;
import org.deeplearning4j.core.loader.DataSetLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.common.loader.FileBatch;
import org.nd4j.common.loader.Source;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

import java.io.IOException;

/**
 * A dataset loader for use with FileBatch objects.
 * The API (constructor arguments) mirrors {@link RecordReaderDataSetIterator} which it uses internally.
 * Can be used in the context of Spark - see SparkDataUtils methods for this purpose
 */
public class RecordReaderFileBatchLoader implements DataSetLoader {
    private final RecordReader recordReader;
    private final int batchSize;
    private final int labelIndexFrom;
    private final int labelIndexTo;
    private final int numPossibleLabels;
    private final boolean regression;
    @Getter @Setter
    private DataSetPreProcessor preProcessor;

    /**
     * Main constructor for classification. This will convert the input class index (at position labelIndex, with integer
     * values 0 to numPossibleLabels-1 inclusive) to the appropriate one-hot output/labels representation.
     *
     * @param recordReader RecordReader: provides the source of the data
     * @param batchSize    Batch size (number of examples) for the output DataSet objects
     * @param labelIndex   Index of the label Writable (usually an IntWritable), as obtained by recordReader.next()
     * @param numClasses   Number of classes (possible labels) for classification
     */
    public RecordReaderFileBatchLoader(RecordReader recordReader, int batchSize, int labelIndex, int numClasses) {
        this(recordReader, batchSize, labelIndex, labelIndex, numClasses, false, null);
    }

    /**
     * Main constructor for multi-label regression (i.e., regression with multiple outputs). Can also be used for single
     * output regression with labelIndexFrom == labelIndexTo
     *
     * @param recordReader      RecordReader to get data from
     * @param labelIndexFrom    Index of the first regression target
     * @param labelIndexTo      Index of the last regression target, inclusive
     * @param batchSize         Minibatch size
     * @param regression        Require regression = true. Mainly included to avoid clashing with other constructors previously defined :/
     */
    public RecordReaderFileBatchLoader(RecordReader recordReader, int batchSize, int labelIndexFrom, int labelIndexTo,
                                       boolean regression) {
        this(recordReader, batchSize, labelIndexFrom, labelIndexTo, -1, regression, null);
    }

    /**
     * Main constructor
     *
     * @param recordReader      the recordreader to use
     * @param batchSize         Minibatch size - number of examples returned for each call of .next()
     * @param labelIndexFrom    the index of the label (for classification), or the first index of the labels for multi-output regression
     * @param labelIndexTo      only used if regression == true. The last index <i>inclusive</i> of the multi-output regression
     * @param numPossibleLabels the number of possible labels for classification. Not used if regression == true
     * @param regression        if true: regression. If false: classification (assume labelIndexFrom is the class it belongs to)
     * @param preProcessor      Optional DataSetPreProcessor. May be null.
     */
    public RecordReaderFileBatchLoader(RecordReader recordReader, int batchSize, int labelIndexFrom, int labelIndexTo,
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
        if (preProcessor != null) {
            iter.setPreProcessor(preProcessor);
        }
        DataSet ds = iter.next();
        return ds;
    }
}
