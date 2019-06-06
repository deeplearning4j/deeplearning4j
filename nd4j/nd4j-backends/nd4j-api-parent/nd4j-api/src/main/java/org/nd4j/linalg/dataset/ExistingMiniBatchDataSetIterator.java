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

package org.nd4j.linalg.dataset;

import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Read in existing mini batches created
 * by the mini batch file datasetiterator.
 *
 * @author Adam Gibson
 */
public class ExistingMiniBatchDataSetIterator implements DataSetIterator {

    public static final String DEFAULT_PATTERN = "dataset-%d.bin";

    private int currIdx;
    private File rootDir;
    private int totalBatches = -1;
    private DataSetPreProcessor dataSetPreProcessor;
    private final String pattern;

    /**
     * Create with the given root directory, using the default filename pattern {@link #DEFAULT_PATTERN}
     * @param rootDir the root directory to use
     */
    public ExistingMiniBatchDataSetIterator(File rootDir) {
        this(rootDir, DEFAULT_PATTERN);
    }

    /**
     *
     * @param rootDir    The root directory to use
     * @param pattern    The filename pattern to use. Used with {@code String.format(pattern,idx)}, where idx is an
     *                   integer, starting at 0.
     */
    public ExistingMiniBatchDataSetIterator(File rootDir, String pattern) {
        this.rootDir = rootDir;
        totalBatches = rootDir.list().length;
        this.pattern = pattern;
    }

    @Override
    public DataSet next(int num) {
        throw new UnsupportedOperationException("Unable to load custom number of examples");
    }

    @Override
    public int inputColumns() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int totalOutcomes() {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        currIdx = 0;
    }

    @Override
    public int batch() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.dataSetPreProcessor = preProcessor;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return dataSetPreProcessor;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public boolean hasNext() {
        return currIdx < totalBatches;
    }

    @Override
    public void remove() {
        //no opt;
    }

    @Override
    public DataSet next() {
        try {
            DataSet ret = read(currIdx);
            if (dataSetPreProcessor != null)
                dataSetPreProcessor.preProcess(ret);
            currIdx++;

            return ret;
        } catch (IOException e) {
            throw new IllegalStateException("Unable to read dataset");
        }
    }

    private DataSet read(int idx) throws IOException {
        File path = new File(rootDir, String.format(pattern, idx));
        DataSet d = new DataSet();
        d.load(path);
        return d;
    }
}
