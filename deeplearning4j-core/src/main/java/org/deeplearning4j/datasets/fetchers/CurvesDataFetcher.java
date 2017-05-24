/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.datasets.fetchers;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.util.SerializationUtils;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.File;
import java.io.IOException;
import java.net.URL;

/**
 * Curves data fetcher
 *
 * @author Adam Gibson
 */
@Deprecated
public class CurvesDataFetcher extends BaseDataFetcher {

    public final static String CURVES_URL = "https://s3.amazonaws.com/dl4j-distribution/curves.ser";
    public final static String LOCAL_DIR_NAME = "curves";
    public final static String CURVES_FILE_NAME = "curves.ser";
    private DataSet data;



    public CurvesDataFetcher() throws IOException {
        download();
        totalExamples = data.numExamples();


    }

    private void download() throws IOException {
        // mac gives unique tmp each run and we want to store this persist
        // this data across restarts
        File tmpDir = new File(System.getProperty("user.home"));

        File baseDir = new File(tmpDir, LOCAL_DIR_NAME);
        if (!(baseDir.isDirectory() || baseDir.mkdir())) {
            throw new IOException("Could not mkdir " + baseDir);
        }

        File dataFile = new File(baseDir, CURVES_FILE_NAME);

        if (!dataFile.exists() || !dataFile.isFile()) {
            log.info("Downloading curves dataset...");
            FileUtils.copyURLToFile(new URL(CURVES_URL), dataFile);
        }


        data = SerializationUtils.readObject(dataFile);



    }


    @Override
    public boolean hasMore() {
        return super.hasMore();
    }

    /**
     * Fetches the next dataset. You need to call this
     * to getFromOrigin a new dataset, otherwise {@link #next()}
     * just returns the last data applyTransformToDestination fetch
     *
     * @param numExamples the number of examples to fetch
     */
    @Override
    public void fetch(int numExamples) {
        if (cursor >= data.numExamples()) {
            cursor = data.numExamples();
        }

        curr = data.get(ArrayUtil.range(cursor, cursor + numExamples));
        log.info("Fetched " + curr.numExamples());
        if (cursor + numExamples < data.numExamples())
            cursor += numExamples;
        //always stay at the end
        else if (cursor + numExamples > data.numExamples())
            cursor = data.numExamples() - 1;

    }
}
