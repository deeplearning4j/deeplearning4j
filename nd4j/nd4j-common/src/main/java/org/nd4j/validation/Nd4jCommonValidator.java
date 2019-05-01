package org.nd4j.validation;

import lombok.NonNull;

import java.io.File;
import java.util.Collections;

public class Nd4jCommonValidator {

    private Nd4jCommonValidator(){ }

    protected ValidationResult isValidFile(@NonNull File f, String formatType, boolean allowEmpty){
        String path;
        try{
            //Very occasionally: getAbsolutePath not possible (files in JARs etc)
            path = f.getAbsolutePath();
        } catch (Throwable t ){
            path = f.getPath();
        }

        if(!f.exists() || !f.isFile()){
            return ValidationResult.builder()
                    .valid(false)
                    .formatType(formatType)
                    .path(path)
                    .issues(Collections.singletonList("File does not exist"))
                    .build();
        }

        if(!f.isFile()){
            return ValidationResult.builder()
                    .valid(false)
                    .formatType(formatType)
                    .path(path)
                    .issues(Collections.singletonList(f.isDirectory() ? "Specified path is a directory" : "Specified path is not a file"))
                    .build();
        }

        if(f.length() <= 0){
            return ValidationResult.builder()
                    .valid(false)
                    .formatType(formatType)
                    .path(path)
                    .issues(Collections.singletonList("File is empty (length 0)"))
                    .build();
        }

        return null;    //OK
    }

    public ValidationResult isValidJSON(@NonNull File f){

        //ValidationResult vr = isValidFile()

        return null;
    }

    public ValidationResult isValidJSON(String s){

        return null;
    }

}
