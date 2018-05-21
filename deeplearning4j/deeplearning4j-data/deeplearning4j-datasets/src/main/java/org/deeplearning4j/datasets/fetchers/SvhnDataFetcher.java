/*-
 *  * Copyright 2017 Skymind, Inc.
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
 */

package org.deeplearning4j.datasets.fetchers;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.transform.ImageTransform;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * The Street View House Numbers (SVHN) Dataset is a real-world image dataset for developing machine learning
 * and object recognition algorithms with minimal requirement on data preprocessing and formatting.
 *
 * The SVHN datasets contain 10 classes (digits) with 73257 digits for training, 26032 digits for testing, and 531131 extra.
 *
 * Datasets in "Format 1: Full Numbers" are fetched.
 *
 * See: <a href="http://ufldl.stanford.edu/housenumbers/">http://ufldl.stanford.edu/housenumbers/</a>
 *
 * @author saudet
 */
public class SvhnDataFetcher extends CacheableExtractableDataSetFetcher {

    public static int NUM_LABELS = 10;

    @Override
    public String remoteDataUrl(DataSetType set) {
        switch (set) {
            case TRAIN:
                return "http://ufldl.stanford.edu/housenumbers/train.tar.gz";
            case TEST:
                return "http://ufldl.stanford.edu/housenumbers/test.tar.gz";
            case VALIDATION:
                return "http://ufldl.stanford.edu/housenumbers/extra.tar.gz";
            default:
                 throw new IllegalArgumentException("Unknown DataSetType:" + set);
        }
    }

    @Override
    public String localCacheName() {
        return "SVHN";
    }

    @Override
    public String dataSetName(DataSetType set) {
        switch (set) {
            case TRAIN:
                return "train";
            case TEST:
                return "test";
            case VALIDATION:
                return "extra";
            default:
                throw new IllegalArgumentException("Unknown DataSetType:" + set);
        }
    }

    @Override
    public long expectedChecksum(DataSetType set) {
        switch (set) {
            case TRAIN:
                return 979655493L;
            case TEST:
                return 1629515343L;
            case VALIDATION:
                return 132781169L;
            default:
                 throw new IllegalArgumentException("Unknown DataSetType:" + set);
        }
    }

    public File getDataSetPath(DataSetType set) throws IOException {
        // check empty cache
        if (LOCAL_CACHE.exists()) {
            if (LOCAL_CACHE.listFiles().length < 1) {
                LOCAL_CACHE.delete();
            }
        }

        File datasetPath;
        switch (set) {
            case TRAIN:
                datasetPath = new File(LOCAL_CACHE, "/train/");
                break;
            case TEST:
                datasetPath = new File(LOCAL_CACHE, "/test/");
                break;
            case VALIDATION:
                datasetPath = new File(LOCAL_CACHE, "/extra/");
                break;
            default:
                datasetPath = null;
        }

        if (!datasetPath.exists()) {
            downloadAndExtract(set);
        }
        return datasetPath;
    }

    @Override
    public RecordReader getRecordReader(long rngSeed, int[] imgDim, DataSetType set, ImageTransform imageTransform) {
        try {
            Random rng = new Random(rngSeed);
            File datasetPath = getDataSetPath(set);

            FileSplit data = new FileSplit(datasetPath, BaseImageLoader.ALLOWED_FORMATS, rng);
            ObjectDetectionRecordReader recordReader = new ObjectDetectionRecordReader(imgDim[1], imgDim[0], imgDim[2],
                            imgDim[4], imgDim[3], null);

            recordReader.initialize(data);
            return recordReader;
        } catch (IOException e) {
            throw new RuntimeException("Could not download SVHN", e);
        }
    }
}