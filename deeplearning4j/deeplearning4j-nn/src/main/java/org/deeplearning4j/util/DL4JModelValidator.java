package org.deeplearning4j.util;

import lombok.NonNull;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.validation.Nd4jCommonValidator;
import org.nd4j.validation.ValidationResult;

import java.io.File;
import java.util.Arrays;
import java.util.List;

public class DL4JModelValidator {

    private DL4JModelValidator(){ }

    public static ValidationResult isValidMultiLayerNetwork(@NonNull File f){

        List<String> requiredEntries = Arrays.asList(ModelSerializer.CONFIGURATION_JSON, ModelSerializer.COEFFICIENTS_BIN);     //TODO no-params models... might be OK to have no params

        ValidationResult vr = Nd4jCommonValidator.isValidZipFile(f, false, requiredEntries);
        if(vr != null) {
            vr.setFormatClass(MultiLayerNetwork.class);
            vr.setFormatType("MultiLayerNetwork");
            return vr;
        }

        //Check that configuration (JSON) can actually be deserialized correctly...

    }

    public static ValidationResult isValidComputationGraph(@NonNull File f){

        List<String> requiredEntries = Arrays.asList(ModelSerializer.CONFIGURATION_JSON, ModelSerializer.COEFFICIENTS_BIN);     //TODO no-params models... might be OK to have no params

        ValidationResult vr = Nd4jCommonValidator.isValidZipFile(f, false, requiredEntries);
        if(vr != null) {
            vr.setFormatClass(ComputationGraph.class);
            vr.setFormatType("ComputationGraph");
            return vr;
        }

        //Check that configuration (JSON) can actually be deserialized correctly...

    }

}
