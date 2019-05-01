package org.deeplearning4j.util;

import lombok.NonNull;
import org.nd4j.validation.Nd4jCommonValidator;
import org.nd4j.validation.ValidationResult;

import java.io.File;

public class DL4JModelValidator {

    private DL4JModelValidator(){ }

    public static ValidationResult isValidMultiLayerNetwork(@NonNull File f){

        ValidationResult vr = Nd4jCommonValidator.isValidFile(f, "MultiLayerNetwork", false);
        if(vr != null)
            return vr;



    }

}
