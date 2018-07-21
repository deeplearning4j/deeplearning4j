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

package org.deeplearning4j.spark.iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.deeplearning4j.api.loader.MultiDataSetLoader;
import org.deeplearning4j.spark.data.loader.RemoteFileSource;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.util.Collection;
import java.util.Iterator;

/**
 * A DataSetIterator that loads serialized DataSet objects (saved with {@link MultiDataSet#save(OutputStream)}) from
 * a String that represents the path (for example, on HDFS)
 *
 * @author Alex Black
 */
public class PathSparkMultiDataSetIterator implements MultiDataSetIterator {

    public static final int BUFFER_SIZE = 4194304; //4 MB

    private final Collection<String> dataSetStreams;
    private MultiDataSetPreProcessor preprocessor;
    private Iterator<String> iter;
    private FileSystem fileSystem;
    private final MultiDataSetLoader loader;

    public PathSparkMultiDataSetIterator(Iterator<String> iter, MultiDataSetLoader loader) {
        this.dataSetStreams = null;
        this.iter = iter;
        this.loader = loader;
    }

    public PathSparkMultiDataSetIterator(Collection<String> dataSetStreams, MultiDataSetLoader loader) {
        this.dataSetStreams = dataSetStreams;
        iter = dataSetStreams.iterator();
        this.loader = loader;
    }

    @Override
    public MultiDataSet next(int num) {
        return next();
    }

    @Override
    public boolean resetSupported() {
        return dataSetStreams != null;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        if (dataSetStreams == null)
            throw new IllegalStateException("Cannot reset iterator constructed with an iterator");
        iter = dataSetStreams.iterator();
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        this.preprocessor = preProcessor;
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return preprocessor;
    }

    @Override
    public boolean hasNext() {
        return iter.hasNext();
    }

    @Override
    public MultiDataSet next() {
        MultiDataSet ds = load(iter.next());

        if (preprocessor != null)
            preprocessor.preProcess(ds);
        return ds;
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }


    private synchronized MultiDataSet load(String path) {
        if (fileSystem == null) {
            try {
                fileSystem = FileSystem.get(new URI(path), new Configuration());
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        try{
            return loader.load(new RemoteFileSource(path, fileSystem, BUFFER_SIZE));
        } catch (IOException e) {
            throw new RuntimeException("Error loading MultiDataSet at path " + path + " - DataSet may be corrupt or invalid." +
                    " Spark MultiDataSets can be validated using org.deeplearning4j.spark.util.data.SparkDataValidation", e);
        }
    }
}
