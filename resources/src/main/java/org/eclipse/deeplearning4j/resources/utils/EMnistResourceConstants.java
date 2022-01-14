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
package org.eclipse.deeplearning4j.resources.utils;

import org.deeplearning4j.common.resources.DL4JResources;

public class EMnistResourceConstants {

    public static String getTrainingFilesURL(EMnistSet ds) {
        return DL4JResources.getURLString("datasets/emnist/" + getImagesFileName(ds, true));
    }

    public static String getTrainingFilesMD5(EMnistSet ds) {
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

    public static String getTrainingFilesFilename(EMnistSet ds) {
        return getImagesFileName(ds, true);
    }

    public static String getTrainingFilesFilename_unzipped(EMnistSet ds) {
        return getImagesFileNameUnzipped(ds, true);
    }

    public static String getTrainingFileLabelsURL(EMnistSet ds) {
        return DL4JResources.getURLString("datasets/emnist/" + getLabelsFileName(ds, true));
    }

    public static String getTrainingFileLabelsMD5(EMnistSet ds) {
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

    public static String getTrainingFileLabelsFilename(EMnistSet ds) {
        return getLabelsFileName(ds, true);
    }

    public static String getTrainingFileLabelsFilename_unzipped(EMnistSet ds) {
        return getLabelsFileNameUnzipped(ds, true);
    }


    // --- Test files ---

    public static String getTestFilesURL(EMnistSet ds) {
        return DL4JResources.getURLString("datasets/emnist/" + getImagesFileName(ds, false));
    }

    public static String getTestFilesMD5(EMnistSet ds) {
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

    public static String getTestFilesFilename(EMnistSet ds) {
        return getImagesFileName(ds, false);
    }

    public static String getTestFilesFilename_unzipped(EMnistSet ds) {
        return getImagesFileNameUnzipped(ds, false);
    }

    public static String getTestFileLabelsURL(EMnistSet ds) {
        return DL4JResources.getURLString("datasets/emnist/" + getLabelsFileName(ds, false));
    }

    public static String getTestFileLabelsMD5(EMnistSet ds) {
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

    public static String getTestFileLabelsFilename(EMnistSet ds) {
        return getLabelsFileName(ds, false);
    }

    public static String getTestFileLabelsFilename_unzipped(EMnistSet ds) {
        return getLabelsFileNameUnzipped(ds, false);
    }


    public static String getImagesFileName(EMnistSet ds, boolean train) {
        return "emnist-" + name(ds) + "-" + (train ? "train" : "test") + "-images-idx3-ubyte.gz";
    }

    public static String getImagesFileNameUnzipped(EMnistSet ds, boolean train) {
        return "emnist-" + name(ds) + "-" + (train ? "train" : "test") + "-images-idx3-ubyte";
    }

    public static String getLabelsFileName(EMnistSet ds, boolean train) {
        return "emnist-" + name(ds) + "-" + (train ? "train" : "test") + "-labels-idx1-ubyte.gz";
    }

    public static String getLabelsFileNameUnzipped(EMnistSet ds, boolean train) {
        return "emnist-" + name(ds) + "-" + (train ? "train" : "test") + "-labels-idx1-ubyte";
    }

    public static String getMappingFileName(EMnistSet ds, boolean train) {
        return "emnist-" + name(ds) + "-mapping.txt";
    }

    public static String name(EMnistSet ds) {
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


}
