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
package org.eclipse.deeplearning4j.resources;

import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.common.resources.ResourceType;
import org.eclipse.deeplearning4j.resources.utils.*;

import java.io.File;

import static org.eclipse.deeplearning4j.resources.utils.EMnistResourceConstants.*;
import static org.eclipse.deeplearning4j.resources.utils.MnistResourceConstants.*;

/**
 * Top level class for leveraging pre curated datasets
 * The pattern of usage for a default directory is:
 *
 * DataSetResource resource = ResourceDataSets.(..);
 * For specifying a directory, for download it's:
 * DataSetResource customDirResource = ResourceDataSets(yourCustomDirPath);
 *
 *  A user can then specify a resource download with:
 *  resource.download(..);
 *
 *  A user won't normally use this class directly. It's common underlying
 *  tooling for the pre existing dataset iterators such as
 *  the MNISTDataSetIterator,EMnistDataSetIterator,LFWDataSetIterator
 *  as well as the BaseImageLoader
 *
 * @author Adam Gibson
 */
public class ResourceDataSets {

    private static File topLevelResourceDir = new File(System.getProperty("user.home"),".dl4jresources");


    // --- Test files ---


    /**
     * The default top level directory for storing data
     * @return
     */
    public static File topLevelResourceDir() {
        return topLevelResourceDir;
    }

    /**
     * Set the top level directory. If intended for use, this should be set
     * before using any datasets or other resources.
     * @param topLevelResourceDir the new directory to use
     */
    public static void setTopLevelResourceDir(File topLevelResourceDir) {
        ResourceDataSets.topLevelResourceDir = topLevelResourceDir;
    }


    /**
     * Download the training file for emnist mapping
     * @param set the set to download
     * @return the resource representing the subset
     */
    public static DataSetResource emnistMappingTrain(EMnistSet set) {
        return emnistMapping(set,DL4JResources.getDirectory(ResourceType.DATASET, "EMNIST"),true);
    }


    /**
     * The input mapping for the test set
     * @param set
     * @return
     */
    public static DataSetResource emnistMappingTest(EMnistSet set) {
        return emnistMapping(set,DL4JResources.getDirectory(ResourceType.DATASET, "EMNIST"),false);
    }

    /**
     * The input mapping for the training set
     * @param set the emnist subset to use
     * @param topLevelDir the top level directory to download to
     * @return
     */
    public static DataSetResource emnistMappingTrain(EMnistSet set, File topLevelDir) {
        return emnistMapping(set,topLevelDir,true);
    }


    /**
     * The input label mapping for the test set
     * @param set the subset of emnist to use
     * @param topLevelDir the custom top level directory
     * @return
     */
    public static DataSetResource emnistMappingTest(EMnistSet set, File topLevelDir) {
        return emnistMapping(set,topLevelDir,false);
    }


    private static DataSetResource emnistMapping(EMnistSet set,File topLevelDir,boolean train) {
        return new DataSetResource(
                EMnistResourceConstants.getMappingFileName(set,train),
                "",
                null,
                topLevelDir.getAbsolutePath(),
                DL4JResources.getURLString("datasets/emnist")
        );
    }


    /**
     *
     * @param set
     * @param topLevelDir
     * @return
     */
    public static DataSetResource emnistLabelsTrain(EMnistSet set, File topLevelDir) {
        return emnistLabels(set,topLevelDir,true);
    }


    /**
     * The emnist test labels
     * @param set the emnist subset to download
     * @param topLevelDir the top level directory to use
     * @return
     */
    public static DataSetResource emnistLabelsTest(EMnistSet set, File topLevelDir) {
        return emnistLabels(set,topLevelDir,false);
    }

    /**
     * The emnist train labels
     * @param set the emnist subset to download
     * @return
     */
    public static DataSetResource emnistLabelsTrain(EMnistSet set) {
        return emnistLabels(set,DL4JResources.getDirectory(ResourceType.DATASET, "EMNIST"),true);
    }

    /**
     * The emnist test labels
     * @param set the emnist subset to download
     * @return
     */
    public static DataSetResource emnistLabelsTest(EMnistSet set) {
        return emnistLabels(set,DL4JResources.getDirectory(ResourceType.DATASET, "EMNIST"),false);
    }


    private static DataSetResource emnistLabels(EMnistSet set,File topLevelDir,boolean train) {
        return new DataSetResource(
                getLabelsFileNameUnzipped(set,train),
                train ? EMnistResourceConstants.getTrainingFileLabelsMD5(set) : EMnistResourceConstants.getTestFileLabelsMD5(set),
                EMnistResourceConstants.getLabelsFileName(set,train),
                topLevelDir.getAbsolutePath(),
                DL4JResources.getURLString("datasets/emnist")
        );
    }

    /**
     * The emnist train labels
     * @param set the emnist subset to download
     * @param topLevelDir the top level directory to download to
     * @return
     */
    public static DataSetResource emnistTrain(EMnistSet set,File topLevelDir) {
        return emnist(set,topLevelDir,true);
    }

    /**
     * The emnist test labels
     * @param set the emnist subset to download
     * @param topLevelDir the top level directory to download to
     * @return
     */
    public static DataSetResource emnistTest(EMnistSet set,File topLevelDir) {
        return emnist(set,topLevelDir,false);
    }

    /**
     * The emnist test labels
     * @param set the emnist subset to download
     * @return
     */
    public static DataSetResource emnistTrain(EMnistSet set) {
        return emnist(set, DL4JResources.getDirectory(ResourceType.DATASET, "EMNIST"),true);
    }

    /**
     * The emnist test labels
     * @param set the emnist subset to download
     * @return
     */
    public static DataSetResource emnistTest(EMnistSet set) {
        return emnist(set,DL4JResources.getDirectory(ResourceType.DATASET, "EMNIST"),false);
    }


    private static DataSetResource emnist(EMnistSet set,File topLevelDir,boolean train) {
        return new DataSetResource(
                EMnistResourceConstants.getImagesFileNameUnzipped(set,train),
                train ? EMnistResourceConstants.getTrainingFilesMD5(set) : EMnistResourceConstants.getTestFilesMD5(set),
                EMnistResourceConstants.getImagesFileName(set,train),
                topLevelDir.getAbsolutePath(),
                DL4JResources.getURLString("datasets/emnist")
        );
    }


    /**
     * The mnist training set
     * @param topLevelDir the top level directory ot use
     * @return
     */
    public static DataSetResource mnistTrain(File topLevelDir) {
        return new DataSetResource(
                MnistResourceConstants.getMNISTTrainingFilesFilename_unzipped(),
                MnistResourceConstants.getMNISTTrainingFilesMD5(),
                MnistResourceConstants.getMNISTTrainingFilesFilename(),
                topLevelDir.getAbsolutePath(),
                DL4JResources.getURLString(MnistResourceConstants.MNIST_ROOT)
        );
    }

    /**
     * The mnist training input data with a default directory of:
     * {@link DL4JResources#getBaseDirectory()}
     * @return
     */
    public static DataSetResource mnistTrain() {
        return mnistTrain(DL4JResources.getBaseDirectory());
    }


    /**
     * The mnist test images
     * @param topLevelDir the top level directory to download to
     * @return
     */
    public static DataSetResource mnistTest(File topLevelDir) {
        return new DataSetResource(
                MnistResourceConstants.getTestFilesFilename_unzipped(),
                MnistResourceConstants.getTestFilesMD5(),
                MnistResourceConstants.getTestFilesFilename(),
                topLevelDir.getAbsolutePath(),
                DL4JResources.getURLString(MnistResourceConstants.MNIST_ROOT)
        );
    }

    /**
     * The mnist test set input with a top level directory of:
     * {@link DL4JResources#getBaseDirectory()}
     * @return
     */
    public static DataSetResource mnistTest() {
        return mnistTest(DL4JResources.getBaseDirectory());
    }


    /**
     * The mnist training labels
     * @param topLevelDir the top level directory to download to
     * @return
     */
    public static DataSetResource mnistTrainLabels(File topLevelDir) {
        return new DataSetResource(
                MnistResourceConstants.getMNISTTrainingFileLabelsFilename_unzipped(),
                MnistResourceConstants.getMNISTTrainingFileLabelsMD5(),
                MnistResourceConstants.getMNISTTrainingFileLabelsFilename(),
                topLevelDir.getAbsolutePath(),
                DL4JResources.getURLString(MnistResourceConstants.MNIST_ROOT));
    }

    /**
     * The mnist train labels with a top level directory of:
     * {@link DL4JResources#getBaseDirectory()}
     * @return
     */
    public static DataSetResource mnistTrainLabels() {
        return mnistTrainLabels(DL4JResources.getBaseDirectory());
    }


    /**
     * The mnist test labels
     * @param topLevelDir the top level directory to download to
     * @return
     */
    public static DataSetResource mnistTestLabels(File topLevelDir) {
        return new DataSetResource(
                MnistResourceConstants.getTestFileLabelsFilename_unzipped(),
                MnistResourceConstants.getTestFileLabelsMD5(),
                MnistResourceConstants.getTestFileLabelsFilename(),
                topLevelDir.getAbsolutePath(),
                DL4JResources.getURLString(MnistResourceConstants.MNIST_ROOT)
        );
    }


    /**
     * The mnist test set labels with a top level directory of:
     * {@link DL4JResources#getBaseDirectory}
     * @return
     */
    public static DataSetResource mnistTestLabels() {
        return mnistTestLabels(DL4JResources.getBaseDirectory());
    }




    /**
     *
     * Labels for the LFW full dataset
     * @param topLevelDir the top level directory to download to
     * @return
     */
    public static DataSetResource lfwFullLabels(File topLevelDir) {
        return new DataSetResource(
                LFWResourceConstants.LFW_LABEL_FILE,
                topLevelDir.getAbsolutePath(),
                LFWResourceConstants.LFW_ROOT_URL
        );
    }

    /**
     *
     * Input images for the LFW full dataset
     * @param topLevelDir the top level directory to download to
     * @return
     */
    public static DataSetResource lfwFullData(File topLevelDir) {
        return new DataSetResource(
                LFWResourceConstants.LFW_FULL_DIR,
                LFWResourceConstants.LFW_DATA_URL,
                topLevelDir.getAbsolutePath(),
                LFWResourceConstants.LFW_ROOT_URL
        );
    }


    /**
     *
     * Input images for the LFW smaller dataset
     * @param topLevelDir the top level directory to download to
     * @return
     */
    public static DataSetResource lfwSubData(File topLevelDir) {
        return new DataSetResource(
                LFWResourceConstants.LFW_SUB_DIR,
                LFWResourceConstants.LFW_SUBSET_URL,
                topLevelDir.getAbsolutePath(),
                LFWResourceConstants.LFW_ROOT_URL
        );
    }


    /**
     * The cifar10 dataset
     * @param topLevelDir the top level directory to download to
     * @return
     */
    public static DataSetResource cifar10(File topLevelDir) {
        return new DataSetResource(
                CifarResourceConstants.CIFAR_DEFAULT_DIR.getAbsolutePath(),
                CifarResourceConstants.CIFAR_ARCHIVE_FILE,
                topLevelDir.getAbsolutePath(),
                CifarResourceConstants.CIFAR_ROOT_URL);
    }


    /**
     *
     * Input labels for the LFW full dataset
     * with a top level directory of:
     * {@link LFWResourceConstants#LFW_FULL_DIR}
     * @return
     */
    public static DataSetResource lfwFullLabels() {
        return lfwFullLabels(new File(topLevelResourceDir, LFWResourceConstants.LFW_FULL_DIR));
    }

    /**
     *
     * Input images for the LFW full dataset
     * with a top level directory of:
     * {@link LFWResourceConstants#LFW_FULL_DIR}
     * @return
     */
    public static DataSetResource lfwFullData() {
        return lfwFullData(new File(topLevelResourceDir, LFWResourceConstants.LFW_FULL_DIR));
    }

    /**
     *
     * Input images for the LFW smaller dataset
     * with a top level directory of:
     * {@link LFWResourceConstants#LFW_SUB_DIR}
     * @return
     */
    public static DataSetResource lfwSubData() {
        return lfwSubData(new File(topLevelResourceDir, LFWResourceConstants.LFW_SUB_DIR));
    }

    /**
     * The cifar 10 dataset with a default directory of:
     * {@link CifarResourceConstants#CIFAR_DEFAULT_DIR}
     * @return
     */
    public static DataSetResource cifar10() {
        return cifar10(CifarResourceConstants.CIFAR_DEFAULT_DIR);
    }


}
