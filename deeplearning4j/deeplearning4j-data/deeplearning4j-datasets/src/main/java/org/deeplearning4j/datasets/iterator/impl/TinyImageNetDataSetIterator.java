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

package org.deeplearning4j.datasets.iterator.impl;

import lombok.Getter;
import org.apache.commons.io.FileUtils;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.common.resources.ResourceType;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.fetchers.TinyImageNetFetcher;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Tiny ImageNet is a subset of the ImageNet database. TinyImageNet is the default course challenge for CS321n
 * at Stanford University.
 *
 * Tiny ImageNet has 200 classes, each consisting of 500 training images.<br>
 * Images are 64x64 pixels, RGB.
 *
 * See: <a href="http://cs231n.stanford.edu/">http://cs231n.stanford.edu/</a> and
 * <a href="https://tiny-imagenet.herokuapp.com/">https://tiny-imagenet.herokuapp.com/</a>
 *
 * @author Justin Long (crockpotveggies)
 */
public class TinyImageNetDataSetIterator extends RecordReaderDataSetIterator {

    @Getter
    protected DataSetPreProcessor preProcessor;

    /**
     * Create an iterator for the training set, with random iteration order (RNG seed fixed to 123)
     *
     * @param batchSize Minibatch size for the iterator
     */
    public TinyImageNetDataSetIterator(int batchSize) {
        this(batchSize, null, DataSetType.TRAIN, null, 123);
    }

    /**
     * * Create an iterator for the training or test set, with random iteration order (RNG seed fixed to 123)
     *
     * @param batchSize Minibatch size for the iterator
     * @param set       The dataset (train or test)
     */
    public TinyImageNetDataSetIterator(int batchSize, DataSetType set) {
        this(batchSize, null, set, null, 123);
    }

    /**
     * Get the Tiny ImageNet iterator with specified train/test set, with random iteration order (RNG seed fixed to 123)
     *
     * @param batchSize Size of each patch
     * @param imgDim Dimensions of desired output - for example, {64, 64}
     * @param set Train, test, or validation
     */
    public TinyImageNetDataSetIterator(int batchSize, int[] imgDim, DataSetType set) {
        this(batchSize, imgDim, set, null, 123);
    }

    /**
     * Get the Tiny ImageNet iterator with specified train/test set and (optional) custom transform.
     *
     * @param batchSize Size of each patch
     * @param imgDim Dimensions of desired output - for example, {64, 64}
     * @param set Train, test, or validation
     * @param imageTransform Additional image transform for output
     * @param rngSeed random number generator seed to use when shuffling examples
     */
    public TinyImageNetDataSetIterator(int batchSize, int[] imgDim, DataSetType set,
                                       ImageTransform imageTransform, long rngSeed) {
        super(new TinyImageNetFetcher().getRecordReader(rngSeed, imgDim, set, imageTransform), batchSize, 1, TinyImageNetFetcher.NUM_LABELS);
    }

    /**
     * Get the labels - either in "categories" (imagenet synsets format, "n01910747" or similar) or human-readable format,
     * such as "jellyfish"
     * @param categories If true: return category/synset format; false: return "human readable" label format
     * @return Labels
     */
    public static List<String> getLabels(boolean categories){
        List<String> rawLabels = new TinyImageNetDataSetIterator(1).getLabels();
        if(categories){
            return rawLabels;
        }

        //Otherwise, convert to human-readable format, using 'words.txt' file
        File baseDir = DL4JResources.getDirectory(ResourceType.DATASET, TinyImageNetFetcher.LOCAL_CACHE_NAME);
        File labelFile = new File(baseDir, TinyImageNetFetcher.WORDS_FILENAME);
        List<String> lines;
        try {
            lines = FileUtils.readLines(labelFile, StandardCharsets.UTF_8);
        } catch (IOException e){
            throw new RuntimeException("Error reading label file", e);
        }

        Map<String,String> map = new HashMap<>();
        for(String line : lines){
            String[] split = line.split("\t");
            map.put(split[0], split[1]);
        }

        List<String> outLabels = new ArrayList<>(rawLabels.size());
        for(String s : rawLabels){
            String s2 = map.get(s);
            Preconditions.checkState(s2 != null, "Label \"%s\" not found in labels.txt file");
            outLabels.add(s2);
        }
        return outLabels;
    }
}
