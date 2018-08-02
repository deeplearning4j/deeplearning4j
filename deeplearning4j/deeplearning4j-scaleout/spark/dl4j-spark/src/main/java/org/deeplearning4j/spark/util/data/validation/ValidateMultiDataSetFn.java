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

package org.deeplearning4j.spark.util.data.validation;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.spark.util.data.ValidationResult;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;

import java.net.URI;
import java.util.List;

import static org.deeplearning4j.spark.util.data.validation.ValidateDataSetFn.validateArrayShape;

/**
 * Function used to validate MultiDataSets on HDFS - see {@link org.deeplearning4j.spark.util.data.SparkDataValidation} for
 * further details
 *
 * @author Alex Black
 */
public class ValidateMultiDataSetFn implements Function<String, ValidationResult> {
    public static final int BUFFER_SIZE = 4194304; //4 MB

    private final boolean deleteInvalid;
    private final int numFeatures;
    private final int numLabels;
    private final List<int[]> featuresShape;
    private final List<int[]> labelsShape;
    private transient FileSystem fileSystem;

    public ValidateMultiDataSetFn(boolean deleteInvalid, int numFeatures, int numLabels, List<int[]> featuresShape, List<int[]> labelsShape) {
        this.deleteInvalid = deleteInvalid;
        this.numFeatures = numFeatures;
        this.numLabels = numLabels;
        this.featuresShape = featuresShape;
        this.labelsShape = labelsShape;
    }

    @Override
    public ValidationResult call(String path) throws Exception {
        if (fileSystem == null) {
            try {
                fileSystem = FileSystem.get(new URI(path), new Configuration());
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        ValidationResult ret = new ValidationResult();
        ret.setCountTotal(1);

        boolean shouldDelete = false;
        boolean loadSuccessful = false;
        MultiDataSet ds = new MultiDataSet();
        Path p = new Path(path);

        if(fileSystem.isDirectory(p)){
            ret.setCountTotal(0);
            return ret;
        }

        if (!fileSystem.exists(p)) {
            ret.setCountMissingFile(1);
            return ret;
        }

        try (FSDataInputStream inputStream = fileSystem.open(p, BUFFER_SIZE)) {
            ds.load(inputStream);
            loadSuccessful = true;
        } catch (Throwable t) {
            shouldDelete = deleteInvalid;
            ret.setCountLoadingFailure(1);
        }


        boolean isValid = loadSuccessful;
        if (loadSuccessful) {
            //Validate
            if (invalidArray(ds.getFeatures())) {
                ret.setCountMissingFeatures(1);
                isValid = false;
            } else {
                if(featuresShape != null && !validateArrayShapes(numFeatures, featuresShape, ds.getFeatures())){
                    ret.setCountInvalidFeatures(1);
                    isValid = false;
                }
            }

            if(ds.getLabels() == null){
                ret.setCountMissingLabels(1);
                isValid = false;
            } else {
                if(labelsShape != null && !validateArrayShapes(numLabels, labelsShape, ds.getLabels())){
                    ret.setCountInvalidLabels(1);
                    isValid = false;
                }
            }

            if(!isValid && deleteInvalid){
                shouldDelete = true;
            }
        }

        if (isValid) {
            ret.setCountTotalValid(1);
        } else {
            ret.setCountTotalInvalid(1);
        }

        if (shouldDelete) {
            fileSystem.delete(p, false);
            ret.setCountInvalidDeleted(1);
        }

        return ret;
    }

    private static boolean invalidArray(INDArray[] array){
        if(array == null || array.length == 0)
            return true;
        for( int i=0; i<array.length; i++ ){
            if(array[i] == null){
                return true;
            }
        }
        return false;
    }

    private boolean validateArrayShapes(int numFeatures, List<int[]> shapes, INDArray[] arr){
        if(arr.length != numFeatures){
            return false;
        }

        if(shapes == null)
            return true;
        if(shapes.size() != arr.length)
            return false;

        for( int i=0; i<shapes.size(); i++ ){
            if(!validateArrayShape(shapes.get(i), arr[i]))
                return false;
        }
        return true;
    }
}
