/*-
 *  * Copyright 2016 Skymind, Inc.
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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.LinkedList;
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
    public String remoteDataUrl(DataSetType set) { return "http://blob.deeplearning4j.org/datasets/tinyimagenet_200_dl4j.v1.zip"; }
    @Override
    public String localCacheName(){ return "TINYIMAGENET_200"; }
    @Override
    public long expectedChecksum(DataSetType set) { return 33822361L; }
    @Override
    public RecordReader getRecordReader(long rngSeed, int[] imgDim, DataSetType set, ImageTransform imageTransform) {
        // check empty cache
        if(LOCAL_CACHE.exists()) {
            if(LOCAL_CACHE.listFiles().length<1) LOCAL_CACHE.delete();
        }

        try {
            if (!LOCAL_CACHE.exists()) downloadAndExtract();
        } catch(Exception e) {
            throw new RuntimeException("Could not download TinyImageNet", e);
        }

        Random rng = new Random(rngSeed);
        File datasetPath;
        switch (set) {
            case TRAIN:
                datasetPath = new File(LOCAL_CACHE, "/train/");
                break;
            case TEST:
                datasetPath = new File(LOCAL_CACHE, "/test/");
                break;
            case VALIDATION:
                throw new IllegalArgumentException("You will need to manually iterate the /validation/images/ directory, TinyImageNet does not provide labels");

            default:
                datasetPath = new File(LOCAL_CACHE, "/train/");
        }

        // set up file paths
        RandomPathFilter pathFilter = new RandomPathFilter(rng, BaseImageLoader.ALLOWED_FORMATS);
        FileSplit filesInDir = new FileSplit(datasetPath, BaseImageLoader.ALLOWED_FORMATS, rng);
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 1);

        // add transforms
        List<ImageTransform> transforms = new LinkedList<>();
        if(imgDim.length > 0) new ResizeImageTransform(imgDim[0], imgDim[1]);
        if(imageTransform != null) transforms.add(imageTransform);

        ImageRecordReader rr = new ImageRecordReader(TinyImageNetFetcher.INPUT_HEIGHT, TinyImageNetFetcher.INPUT_WIDTH,
                    TinyImageNetFetcher.INPUT_CHANNELS, new ParentPathLabelGenerator(), new MultiImageTransform(transforms.toArray(new ImageTransform[transforms.size()])));

        try {
            rr.initialize(filesInDirSplit[0]);
        } catch(Exception e) {
            throw new RuntimeException(e);
        }

        return rr;
    }


}