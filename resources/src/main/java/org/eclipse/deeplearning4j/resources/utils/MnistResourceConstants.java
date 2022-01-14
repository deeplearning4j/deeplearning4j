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

public class MnistResourceConstants {

    public static final String TRAINING_FILES_URL_RELATIVE = "datasets/mnist/train-images-idx3-ubyte.gz";
    public static final String TRAINING_FILES_MD_5 = "f68b3c2dcbeaaa9fbdd348bbdeb94873";
    public static final String TRAINING_FILES_FILENAME = "train-images-idx3-ubyte.gz";
    public static final String TRAINING_FILES_FILENAME_UNZIPPED = "train-images-idx3-ubyte";
    public static final String TRAINING_FILE_LABELS_URL_RELATIVE = "datasets/mnist/train-labels-idx1-ubyte.gz";
    public static final String TRAINING_FILE_LABELS_MD_5 = "d53e105ee54ea40749a09fcbcd1e9432";
    public static final String TRAINING_FILE_LABELS_FILENAME = "train-labels-idx1-ubyte.gz";
    public static final String TRAINING_FILE_LABELS_FILENAME_UNZIPPED = "train-labels-idx1-ubyte";
    //Test data:
    public static final String TEST_FILES_URL_RELATIVE = "datasets/mnist/t10k-images-idx3-ubyte.gz";
    public static final String TEST_FILES_MD_5 = "9fb629c4189551a2d022fa330f9573f3";
    public static final String TEST_FILES_FILENAME = "t10k-images-idx3-ubyte.gz";
    public static final String TEST_FILES_FILENAME_UNZIPPED = "t10k-images-idx3-ubyte";
    public static final String TEST_FILE_LABELS_URL_RELATIVE = "datasets/mnist/t10k-labels-idx1-ubyte.gz";
    public static final String TEST_FILE_LABELS_MD_5 = "ec29112dd5afa0611ce80d1b7f02629c";
    public static final String TEST_FILE_LABELS_FILENAME = "t10k-labels-idx1-ubyte.gz";
    public static final String TEST_FILE_LABELS_FILENAME_UNZIPPED = "t10k-labels-idx1-ubyte";

    // --- Train files ---
    public static String getMNISTTrainingFilesURL() {
        return DL4JResources.getURLString(TRAINING_FILES_URL_RELATIVE);
    }

    public static String getMNISTTrainingFilesMD5() {
        return TRAINING_FILES_MD_5;
    }

    public static String getMNISTTrainingFilesFilename() {
        return TRAINING_FILES_FILENAME;
    }

    public static String getMNISTTrainingFilesFilename_unzipped() {
        return TRAINING_FILES_FILENAME_UNZIPPED;
    }

    public static String getMNISTTrainingFileLabelsURL() {
        return DL4JResources.getURLString(TRAINING_FILE_LABELS_URL_RELATIVE);
    }

    public static String getMNISTTrainingFileLabelsMD5() {
        return TRAINING_FILE_LABELS_MD_5;
    }

    public static String getMNISTTrainingFileLabelsFilename() {
        return TRAINING_FILE_LABELS_FILENAME;
    }

    public static String getMNISTTrainingFileLabelsFilename_unzipped() {
        return TRAINING_FILE_LABELS_FILENAME_UNZIPPED;
    }

    public static String getTestFilesURL() {
        return DL4JResources.getURLString(TEST_FILES_URL_RELATIVE);
    }

    public static String getTestFilesMD5() {
        return TEST_FILES_MD_5;
    }

    public static String getTestFilesFilename() {
        return TEST_FILES_FILENAME;
    }

    public static String getTestFilesFilename_unzipped() {
        return TEST_FILES_FILENAME_UNZIPPED;
    }

    public  static String getTestFileLabelsURL() {
        return DL4JResources.getURLString(TEST_FILE_LABELS_URL_RELATIVE);
    }

    public static String getTestFileLabelsMD5() {
        return TEST_FILE_LABELS_MD_5;
    }

    public static String getTestFileLabelsFilename() {
        return TEST_FILE_LABELS_FILENAME;
    }

    public static String getTestFileLabelsFilename_unzipped() {
        return TEST_FILE_LABELS_FILENAME_UNZIPPED;
    }
}
