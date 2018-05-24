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

package org.deeplearning4j.base;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;

import java.io.File;

/**
 * Downloader for EMNIST dataset
 *
 * @author Alex Black
 */
@Slf4j
public class EmnistFetcher extends MnistFetcher {

    private final EmnistDataSetIterator.Set ds;

    public EmnistFetcher(EmnistDataSetIterator.Set ds) {
        this.ds = ds;
    }

    private static String getImagesFileName(EmnistDataSetIterator.Set ds, boolean train) {
        return "emnist-" + name(ds) + "-" + (train ? "train" : "test") + "-images-idx3-ubyte.gz";
    }

    private static String getImagesFileNameUnzipped(EmnistDataSetIterator.Set ds, boolean train) {
        return "emnist-" + name(ds) + "-" + (train ? "train" : "test") + "-images-idx3-ubyte";
    }

    private static String getLabelsFileName(EmnistDataSetIterator.Set ds, boolean train) {
        return "emnist-" + name(ds) + "-" + (train ? "train" : "test") + "-labels-idx1-ubyte.gz";
    }

    private static String getLabelsFileNameUnzipped(EmnistDataSetIterator.Set ds, boolean train) {
        return "emnist-" + name(ds) + "-" + (train ? "train" : "test") + "-labels-idx1-ubyte";
    }

    private static String getMappingFileName(EmnistDataSetIterator.Set ds, boolean train) {
        return "emnist-" + name(ds) + "-mapping.txt";
    }

    private static String name(EmnistDataSetIterator.Set ds) {
        switch (ds) {
            case COMPLETE:
                return "byclass";
            case MERGE:
                return "bymerge";
            case BALANCED:
                return "balanced";
            case LETTERS:
                return "letters";
            case DIGITS:
                return "digits";
            case MNIST:
                return "mnist";
            default:
                throw new UnsupportedOperationException("Unknown DataSet: " + ds);
        }
    }

    @Override
    public String getName() {
        return "EMNIST";
    }

    // --- Train files ---
    @Override
    public String getTrainingFilesURL() {
        return DL4JResources.getURLString("datasets/emnist/" + getImagesFileName(ds, true));
    }

    @Override
    public String getTrainingFilesMD5() {
        switch (ds) {
            case COMPLETE:
                //byclass-train-images
                return "712dda0bd6f00690f32236ae4325c377";
            case MERGE:
                //bymerge-train-images
                return "4a792d4df261d7e1ba27979573bf53f3";
            case BALANCED:
                //balanced-train-images
                return "4041b0d6f15785d3fa35263901b5496b";
            case LETTERS:
                //letters-train-images
                return "8795078f199c478165fe18db82625747";
            case DIGITS:
                //digits-train-images
                return "d2662ecdc47895a6bbfce25de9e9a677";
            case MNIST:
                //mnist-train-images
                return "3663598a39195d030895b6304abb5065";
            default:
                throw new UnsupportedOperationException("Unknown DataSet: " + ds);
        }
    }

    @Override
    public String getTrainingFilesFilename() {
        return getImagesFileName(ds, true);
    }

    @Override
    public String getTrainingFilesFilename_unzipped() {
        return getImagesFileNameUnzipped(ds, true);
    }

    @Override
    public String getTrainingFileLabelsURL() {
        return DL4JResources.getURLString("datasets/emnist/" + getLabelsFileName(ds, true));
    }

    @Override
    public String getTrainingFileLabelsMD5() {
        switch (ds) {
            case COMPLETE:
                //byclass-train-labels
                return "ee299a3ee5faf5c31e9406763eae7e43";
            case MERGE:
                //bymerge-train-labels
                return "491be69ef99e1ab1f5b7f9ccc908bb26";
            case BALANCED:
                //balanced-train-labels
                return "7a35cc7b2b7ee7671eddf028570fbd20";
            case LETTERS:
                //letters-train-labels
                return "c16de4f1848ddcdddd39ab65d2a7be52";
            case DIGITS:
                //digits-train-labels
                return "2223fcfee618ac9c89ef20b6e48bcf9e";
            case MNIST:
                //mnist-train-labels
                return "6c092f03c9bb63e678f80f8bc605fe37";
            default:
                throw new UnsupportedOperationException("Unknown DataSet: " + ds);
        }
    }

    @Override
    public String getTrainingFileLabelsFilename() {
        return getLabelsFileName(ds, true);
    }

    @Override
    public String getTrainingFileLabelsFilename_unzipped() {
        return getLabelsFileNameUnzipped(ds, true);
    }


    // --- Test files ---

    @Override
    public String getTestFilesURL() {
        return DL4JResources.getURLString("datasets/emnist/" + getImagesFileName(ds, false));
    }

    @Override
    public String getTestFilesMD5() {
        switch (ds) {
            case COMPLETE:
                //byclass-test-images
                return "1435209e34070a9002867a9ab50160d7";
            case MERGE:
                //bymerge-test-images
                return "8eb5d34c91f1759a55831c37ec2a283f";
            case BALANCED:
                //balanced-test-images
                return "6818d20fe2ce1880476f747bbc80b22b";
            case LETTERS:
                //letters-test-images
                return "382093a19703f68edac6d01b8dfdfcad";
            case DIGITS:
                //digits-test-images
                return "a159b8b3bd6ab4ed4793c1cb71a2f5cc";
            case MNIST:
                //mnist-test-images
                return "fb51b6430fc4dd67deaada1bf25d4524";
            default:
                throw new UnsupportedOperationException("Unknown DataSet: " + ds);
        }
    }

    @Override
    public String getTestFilesFilename() {
        return getImagesFileName(ds, false);
    }

    @Override
    public String getTestFilesFilename_unzipped() {
        return getImagesFileNameUnzipped(ds, false);
    }

    @Override
    public String getTestFileLabelsURL() {
        return DL4JResources.getURLString("datasets/emnist/" + getLabelsFileName(ds, false));
    }

    @Override
    public String getTestFileLabelsMD5() {
        switch (ds) {
            case COMPLETE:
                //byclass-test-labels
                return "7a0f934bd176c798ecba96b36fda6657";
            case MERGE:
                //bymerge-test-labels
                return "c13f4cd5211cdba1b8fa992dae2be992";
            case BALANCED:
                //balanced-test-labels
                return "acd3694070dcbf620e36670519d4b32f";
            case LETTERS:
                //letters-test-labels
                return "d4108920cd86601ec7689a97f2de7f59";
            case DIGITS:
                //digits-test-labels
                return "8afde66ea51d865689083ba6bb779fac";
            case MNIST:
                //mnist-test-labels
                return "ae7f6be798a9a5d5f2bd32e078a402dd";
            default:
                throw new UnsupportedOperationException("Unknown DataSet: " + ds);
        }
    }

    @Override
    public String getTestFileLabelsFilename() {
        return getLabelsFileName(ds, false);
    }

    @Override
    public String getTestFileLabelsFilename_unzipped() {
        return getLabelsFileNameUnzipped(ds, false);
    }

    public static int numLabels(EmnistDataSetIterator.Set dataSet) {
        switch (dataSet) {
            case COMPLETE:
                return 62;
            case MERGE:
                return 47;
            case BALANCED:
                return 47;
            case LETTERS:
                return 26;
            case DIGITS:
                return 10;
            case MNIST:
                return 10;
            default:
                throw new UnsupportedOperationException("Unknown Set: " + dataSet);
        }
    }
}
