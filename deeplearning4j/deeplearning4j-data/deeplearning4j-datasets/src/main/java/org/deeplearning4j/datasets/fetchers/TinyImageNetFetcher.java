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

import org.apache.commons.io.FileUtils;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.deeplearning4j.common.resources.DL4JResources;
import org.nd4j.base.Preconditions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Tiny ImageNet is a subset of the ImageNet database. TinyImageNet is the default course challenge for CS321n
 * at Stanford University.
 *
 * Tiny ImageNet has 200 classes, each consisting of 500 training images.
 *
 * See: <a href="http://cs231n.stanford.edu/">http://cs231n.stanford.edu/</a> and
 * <a href="https://tiny-imagenet.herokuapp.com/">https://tiny-imagenet.herokuapp.com/</a>
 *
 * @author Justin Long (crockpotveggies)
 */
public class TinyImageNetFetcher extends CacheableExtractableDataSetFetcher {

    private File fileDir;
    private static Logger log = LoggerFactory.getLogger(TinyImageNetFetcher.class);

    public static int INPUT_WIDTH = 64;
    public static int INPUT_HEIGHT = 64;
    public static int INPUT_CHANNELS = 3;
    public static int NUM_LABELS = 200;
    public static int NUM_EXAMPLES = NUM_LABELS*500;

    @Override
    public String remoteDataUrl(DataSetType set) {
        return DL4JResources.getURLString("datasets/tinyimagenet_200_dl4j.v1.zip");
    }
    @Override
    public String localCacheName(){ return "TINYIMAGENET_200"; }
    @Override
    public long expectedChecksum(DataSetType set) { return 33822361L; }
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
            throw new RuntimeException("Could not download TinyImageNet", e);
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
                throw new IllegalArgumentException("You will need to manually iterate the /validation/images/ directory, TinyImageNet does not provide labels");

            default:
                datasetPath = new File(localCache, "/train/");
        }

        // set up file paths
        RandomPathFilter pathFilter = new RandomPathFilter(rng, BaseImageLoader.ALLOWED_FORMATS);
        FileSplit filesInDir = new FileSplit(datasetPath, BaseImageLoader.ALLOWED_FORMATS, rng);
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 1);

        int h = (imgDim == null ? TinyImageNetFetcher.INPUT_HEIGHT : imgDim[0]);
        int w = (imgDim == null ? TinyImageNetFetcher.INPUT_WIDTH : imgDim[1]);
        ImageRecordReader rr = new ImageRecordReader(h, w,TinyImageNetFetcher.INPUT_CHANNELS, new ParentPathLabelGenerator(), imageTransform);

        try {
            rr.initialize(filesInDirSplit[0]);
        } catch(Exception e) {
            throw new RuntimeException(e);
        }

        return rr;
    }


}