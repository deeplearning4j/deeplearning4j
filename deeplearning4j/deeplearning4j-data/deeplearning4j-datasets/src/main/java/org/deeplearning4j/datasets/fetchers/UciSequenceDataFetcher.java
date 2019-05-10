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

package org.deeplearning4j.datasets.fetchers;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.image.transform.ImageTransform;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

/**
 * Fetcher for UCI synthetic control chart time series dataset.
 *<br>
 * Details:     <a href="https://archive.ics.uci.edu/ml/datasets/Synthetic+Control+Chart+Time+Series">https://archive.ics.uci.edu/ml/datasets/Synthetic+Control+Chart+Time+Series</a><br>
 * Data:        <a href="https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data">https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data</a>
 * Image:       <a href="https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/data.jpeg">https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/data.jpeg</a>
 *
 * @author Briton Park (bpark738)
 */
@Slf4j
public class UciSequenceDataFetcher extends CacheableExtractableDataSetFetcher {

    public static int NUM_LABELS = 6;
    public static int NUM_EXAMPLES = NUM_LABELS * 100;
    private static String url = "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data";

    public static void setURL(String url){
        UciSequenceDataFetcher.url = url;
    }

    @Override
    public String remoteDataUrl() {
        return url;
    }

    @Override
    public String remoteDataUrl(DataSetType type) {
        return remoteDataUrl();
    }

    @Override
    public String localCacheName() {
        return "UCISequence_6";
    }

    @Override
    public long expectedChecksum() {
        return 104392751L;
    }

    @Override
    public long expectedChecksum(DataSetType type) {
        return expectedChecksum();
    }

    @Override
    public CSVSequenceRecordReader getRecordReader(long rngSeed, int[] shape, DataSetType set, ImageTransform transform) {
        return getRecordReader(rngSeed, set);
    }

    public CSVSequenceRecordReader getRecordReader(long rngSeed, DataSetType set) {

        // check empty cache
        File localCache = getLocalCacheDir();
        deleteIfEmpty(localCache);

        try {
            if (!localCache.exists()) downloadAndExtract();
        } catch (Exception e) {
            throw new RuntimeException("Could not download UCI Sequence data", e);
        }

        File dataPath;

        switch (set) {
            case TRAIN:
                dataPath = new File(localCache, "/train");
                break;
            case TEST:
                dataPath = new File(localCache, "/test");
                break;
            case VALIDATION:
                throw new IllegalArgumentException("You will need to manually iterate the directory, UCISequence data does not provide labels");

            default:
                dataPath = new File(localCache, "/train");
        }

        try {
            downloadUCIData(dataPath);
            CSVSequenceRecordReader data;
            switch (set) {
                case TRAIN:
                    data = new CSVSequenceRecordReader(0, ", ");
                    data.initialize(new NumberedFileInputSplit(dataPath.getAbsolutePath() + "/%d.csv", 0, 449));
                    break;
                case TEST:
                    data = new CSVSequenceRecordReader(0, ", ");
                    data.initialize(new NumberedFileInputSplit(dataPath.getAbsolutePath() + "/%d.csv", 450, 599));
                    break;
                default:
                    data = new CSVSequenceRecordReader(0, ", ");
                    data.initialize(new NumberedFileInputSplit(dataPath.getAbsolutePath() + "/%d.csv", 0, 449));
            }

            return data;
        } catch (Exception e) {
            throw new RuntimeException("Could not process UCI data", e);
        }
    }

    private static void downloadUCIData(File dataPath) throws Exception {
        //if (dataPath.exists()) return;

        String data = IOUtils.toString(new URL(url), Charset.defaultCharset());
        String[] lines = data.split("\n");

        int lineCount = 0;
        int index = 0;

        ArrayList<String> linesList = new ArrayList<>();

        for (String line : lines) {

            // label value
            int count = lineCount++ / 100;

            // replace white space with commas and label value + new line
            line = line.replaceAll("\\s+", ", " + count + "\n");

            // add label to last number
            line = line + ", " + count;
            linesList.add(line);
        }

        // randomly shuffle data
        Collections.shuffle(linesList, new Random(12345));

        for (String line : linesList) {
            File outPath = new File(dataPath, index + ".csv");
            FileUtils.writeStringToFile(outPath, line, Charset.defaultCharset());
            index++;
        }
    }
}
