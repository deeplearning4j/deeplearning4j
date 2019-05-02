package org.deeplearning4j.util;

import lombok.NonNull;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.validation.Nd4jCommonValidator;
import org.nd4j.validation.ValidationResult;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

public class DL4JModelValidator {

    private DL4JModelValidator(){ }

    public static ValidationResult isValidMultiLayerNetwork(@NonNull File f){

        List<String> requiredEntries = Arrays.asList(ModelSerializer.CONFIGURATION_JSON, ModelSerializer.COEFFICIENTS_BIN);     //TODO no-params models... might be OK to have no params

        ValidationResult vr = Nd4jCommonValidator.isValidZipFile(f, false, requiredEntries);
        if(vr != null && !vr.isValid()) {
            vr.setFormatClass(MultiLayerNetwork.class);
            vr.setFormatType("MultiLayerNetwork");
            return vr;
        }

        //Check that configuration (JSON) can actually be deserialized correctly...
        String config;
        try(ZipFile zf = new ZipFile(f)){
            ZipEntry ze = zf.getEntry(ModelSerializer.CONFIGURATION_JSON);
            config = IOUtils.toString(new BufferedReader(new InputStreamReader(zf.getInputStream(ze), StandardCharsets.UTF_8)));
        } catch (IOException e){
            return ValidationResult.builder()
                    .formatType("MultiLayerNetwork")
                    .formatClass(MultiLayerNetwork.class)
                    .valid(false)
                    .path(Nd4jCommonValidator.getPath(f))
                    .issues(Collections.singletonList("Unable to read configuration from model zip file"))
                    .exception(e)
                    .build();
        }

        try{
            MultiLayerConfiguration.fromJson(config);
        } catch (Throwable t){
            return ValidationResult.builder()
                    .formatType("MultiLayerNetwork")
                    .formatClass(MultiLayerNetwork.class)
                    .valid(false)
                    .path(Nd4jCommonValidator.getPath(f))
                    .issues(Collections.singletonList("Zip file JSON model configuration does not appear to represent a valid MultiLayerConfiguration"))
                    .exception(t)
                    .build();
        }

        //TODO should we check params too?

        return ValidationResult.builder()
                .formatType("MultiLayerNetwork")
                .formatClass(MultiLayerNetwork.class)
                .valid(true)
                .path(Nd4jCommonValidator.getPath(f))
                .build();
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
        String config;
        try(ZipFile zf = new ZipFile(f)){
            ZipEntry ze = zf.getEntry(ModelSerializer.CONFIGURATION_JSON);
            config = IOUtils.toString(new BufferedReader(new InputStreamReader(zf.getInputStream(ze), StandardCharsets.UTF_8)));
        } catch (IOException e){
            return ValidationResult.builder()
                    .formatType("ComputationGraph")
                    .formatClass(ComputationGraph.class)
                    .valid(false)
                    .path(Nd4jCommonValidator.getPath(f))
                    .issues(Collections.singletonList("Unable to read configuration from model zip file"))
                    .exception(e)
                    .build();
        }

        try{
            MultiLayerConfiguration.fromJson(config);
        } catch (Throwable t){
            return ValidationResult.builder()
                    .formatType("ComputationGraph")
                    .formatClass(ComputationGraph.class)
                    .valid(false)
                    .path(Nd4jCommonValidator.getPath(f))
                    .issues(Collections.singletonList("Zip file JSON model configuration does not appear to represent a valid ComputationGraphConfiguration"))
                    .exception(t)
                    .build();
        }

        //TODO should we check params too? (a) that it can be read, and (b) that it matches config (number of parameters, etc)

        return ValidationResult.builder()
                .formatType("ComputationGraph")
                .formatClass(ComputationGraph.class)
                .valid(true)
                .path(Nd4jCommonValidator.getPath(f))
                .build();
    }

    //TODO also check if updater is valid?

}
