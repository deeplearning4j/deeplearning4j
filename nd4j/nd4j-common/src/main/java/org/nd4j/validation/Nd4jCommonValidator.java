package org.nd4j.validation;

import lombok.NonNull;
import org.apache.commons.io.FileUtils;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.JavaType;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

public class Nd4jCommonValidator {

    private Nd4jCommonValidator(){ }

    public static ValidationResult isValidFile(@NonNull File f, String formatType, boolean allowEmpty){
        String path;
        try{
            path = f.getAbsolutePath(); //Very occasionally: getAbsolutePath not possible (files in JARs etc)
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

        if(!allowEmpty && f.length() <= 0){
            return ValidationResult.builder()
                    .valid(false)
                    .formatType(formatType)
                    .path(path)
                    .issues(Collections.singletonList("File is empty (length 0)"))
                    .build();
        }

        return null;    //OK
    }

    public static ValidationResult isValidJsonUTF8(@NonNull File f) {
        return isValidJson(f, StandardCharsets.UTF_8);
    }

    public static ValidationResult isValidJson(@NonNull File f, Charset charset){

        ValidationResult vr = isValidFile(f, "JSON", false);
        if(vr != null)
            return vr;

        String content;
        try{
            content = FileUtils.readFileToString(f, charset);
        } catch (IOException e){
            return ValidationResult.builder()
                    .valid(false)
                    .formatType("JSON")
                    .path(getPath(f))
                    .issues(Collections.singletonList("Unable to read file (IOException)"))
                    .exception(e)
                    .build();
        }


        return isValidJson(content, f);
    }

    public static ValidationResult isValidJSON(String s) {
        return isValidJson(s, null);
    }

    protected static ValidationResult isValidJson(String content, File f){
        try{
            ObjectMapper om = new ObjectMapper();
            JavaType javaType = om.getTypeFactory().constructMapType(Map.class, String.class, Object.class);
            om.readValue(content, javaType);    //Don't care about result, just that it can be parsed successfully
        } catch (Throwable t){
            //Jackson should tell us specifically where error occurred also
            return ValidationResult.builder()
                    .valid(false)
                    .formatType("JSON")
                    .path(getPath(f))
                    .issues(Collections.singletonList("File does not appear to be valid JSON"))
                    .exception(t)
                    .build();
        }


        return ValidationResult.builder()
                .valid(true)
                .formatType("JSON")
                .path(getPath(f))
                .build();
    }


    public static ValidationResult isValidZipFile(@NonNull File f, boolean allowEmpty) {
        return isValidZipFile(f, allowEmpty, null);
    }

    public static ValidationResult isValidZipFile(@NonNull File f, boolean allowEmpty, List<String> requiredEntries){
        ValidationResult vr = isValidFile(f, "Zip File", false);
        if(vr != null)
            return vr;

        ZipFile zf;
        try {
            zf = new ZipFile(f);
        } catch (Throwable e) {
            return ValidationResult.builder()
                    .valid(false)
                    .formatType("Zip File")
                    .path(getPath(f))
                    .issues(Collections.singletonList("File does not appear to be valid zip file"))
                    .exception(e)
                    .build();
        }

        try{
            int numEntries = zf.size();
            if(!allowEmpty && numEntries <= 0){
                return ValidationResult.builder()
                        .valid(false)
                        .formatType("Zip File")
                        .path(getPath(f))
                        .issues(Collections.singletonList("Zip file is empty"))
                        .build();
            }

            if(requiredEntries != null && !requiredEntries.isEmpty()) {
                List<String> missing = null;
                for (String s : requiredEntries){
                    ZipEntry ze = zf.getEntry(s);
                    if(ze == null){
                        if(missing == null)
                            missing = new ArrayList<>();
                        missing.add(s);
                    }
                }

                if(missing != null){
                    String s = "Zip file is missing " + missing.size() + " of " + requiredEntries.size() + " required entries: " + requiredEntries;
                    return ValidationResult.builder()
                            .valid(false)
                            .formatType("Zip File")
                            .path(getPath(f))
                            .issues(Collections.singletonList(s))
                            .build();
                }
            }

        } catch (Throwable t){
            return ValidationResult.builder()
                    .valid(false)
                    .formatType("Zip File")
                    .path(getPath(f))
                    .issues(Collections.singletonList("Error reading zip file"))
                    .exception(t)
                    .build();
        } finally {
            try {
                zf.close();
            } catch (IOException e){ }  //Ignore, can't do anything about it...
        }

        return ValidationResult.builder()
                .valid(true)
                .formatType("Zip File")
                .path(getPath(f))
                .build();
    }


    private static String getPath(File f){
        if(f == null)
            return null;
        try{
            return f.getAbsolutePath(); //Very occasionally: getAbsolutePath not possible (files in JARs etc)
        } catch (Throwable t ){
            return f.getPath();
        }
    }



}
