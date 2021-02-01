/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.datasets.fetchers;

import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.common.resources.DL4JResources;
import org.nd4j.common.base.Preconditions;

import java.io.File;
import java.util.Random;

/**
 * CifarDataSetIterator is an iterator for CIFAR-10 dataset - 10 classes, with 32x32 images with 3 channels (RGB)
 *
 * This fetcher uses a cached version of the CIFAR dataset which is converted to PNG images,
 * see: <a href="https://pjreddie.com/projects/cifar-10-dataset-mirror/">https://pjreddie.com/projects/cifar-10-dataset-mirror/</a>.
 *
 * @author Justin Long (crockpotveggies)
 */
public class Cifar10Fetcher extends CacheableExtractableDataSetFetcher {
    public static final String LABELS_FILENAME = "labels.txt";
    public static final String LOCAL_CACHE_NAME = "cifar10";

    public static int INPUT_WIDTH = 32;
    public static int INPUT_HEIGHT = 32;
    public static int INPUT_CHANNELS = 3;
    public static int NUM_LABELS = 10;

    @Override
    public String remoteDataUrl(DataSetType set) {
        return DL4JResources.getURLString("datasets/cifar10_dl4j.v1.zip");
    }
    @Override
    public String localCacheName(){ return LOCAL_CACHE_NAME; }
    @Override
    public long expectedChecksum(DataSetType set) { return 292852033L; }
    @Override
    public RecordReader getRecordReader(long rngSeed, int[] imgDim, DataSetType set, ImageTransform imageTransform) {
        Preconditions.checkState(imgDim == null || imgDim.length == 2, "Invalid image dimensions: must be null or lenth 2. Got: %s", imgDim);
        // check empty cache
        File localCache = getLocalCacheDir();
        deleteIfEmpty(localCache);

        try {
            if (!localCache.exists()){
                downloadAndExtract();
            }
        } catch(Exception e) {
            throw new RuntimeException("Could not download CIFAR-10", e);
        }

        Random rng = new Random(rngSeed);
        File datasetPath;
        switch (set) {
            case TRAIN:
                datasetPath = new File(localCache, "/train/");
                break;
            case TEST:
                datasetPath = new File(localCache, "/test/");
                break;
            case VALIDATION:
                throw new IllegalArgumentException("You will need to manually create and iterate a validation directory, CIFAR-10 does not provide labels");

            default:
                datasetPath = new File(localCache, "/train/");
        }

        // set up file paths
        RandomPathFilter pathFilter = new RandomPathFilter(rng, BaseImageLoader.ALLOWED_FORMATS);
        FileSplit filesInDir = new FileSplit(datasetPath, BaseImageLoader.ALLOWED_FORMATS, rng);
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 1);

        int h = (imgDim == null ? Cifar10Fetcher.INPUT_HEIGHT : imgDim[0]);
        int w = (imgDim == null ? Cifar10Fetcher.INPUT_WIDTH : imgDim[1]);
        ImageRecordReader rr = new ImageRecordReader(h, w, Cifar10Fetcher.INPUT_CHANNELS, new ParentPathLabelGenerator(), imageTransform);

        try {
            rr.initialize(filesInDirSplit[0]);
        } catch(Exception e) {
            throw new RuntimeException(e);
        }

        return rr;
    }


}