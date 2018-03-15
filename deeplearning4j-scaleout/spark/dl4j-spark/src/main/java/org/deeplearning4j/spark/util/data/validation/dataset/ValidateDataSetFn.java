package org.deeplearning4j.spark.util.data.validation.dataset;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.spark.util.data.ValidationResult;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.io.IOException;
import java.net.URI;

public class ValidateDataSetFn implements Function<String, ValidationResult> {
    public static final int BUFFER_SIZE = 4194304; //4 MB

    private final boolean deleteInvalid;
    private final int[] featuresShape;
    private final int[] labelsShape;
    private transient FileSystem fileSystem;

    public ValidateDataSetFn(boolean deleteInvalid, int[] featuresShape, int[] labelsShape) {
        this.deleteInvalid = deleteInvalid;
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
        DataSet ds = new DataSet();
        Path p = new Path(path);

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
            if (ds.getFeatures() == null) {
                ret.setCountMissingFeatures(1);
                isValid = false;
            } else {
                if(featuresShape != null && !validateArrayShape(featuresShape, ds.getFeatures())){
                    ret.setCountInvalidFeatures(1);
                    isValid = false;
                }
            }

            if(ds.getLabels() == null){
                ret.setCountMissingLabels(1);
                isValid = false;
            } else {
                if(labelsShape != null && !validateArrayShape(labelsShape, ds.getLabelsMaskArray())){
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

    private static boolean validateArrayShape(int[] featuresShape, INDArray array){
        if(featuresShape.length != array.rank()){
            return false;
        } else {
            for( int i=0; i<featuresShape.length; i++ ){
                if(featuresShape[i] <= 0)
                    continue;
                if(featuresShape[i] != array.size(i)){
                    return false;
                }
            }
        }
        return true;
    }
}
