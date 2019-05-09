package org.nd4j.linalg.util;

import lombok.NonNull;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.graph.FlatGraph;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.validation.Nd4jCommonValidator;
import org.nd4j.validation.ValidationResult;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.util.Collections;
import java.util.Map;

public class Nd4jValidator {

    private Nd4jValidator(){ }

    public static ValidationResult validateINDArrayFile(@NonNull File f) {
        return validateINDArrayFile(f, (DataType[])null);
    }

    public static ValidationResult validateINDArrayFile(@NonNull File f, DataType... allowableDataTypes){

        ValidationResult vr = Nd4jCommonValidator.isValidFile(f, "INDArray File", false);
        if(vr != null && !vr.isValid())
            return vr;

        //TODO let's do this without reading the whole thing into memory - check header + length...
        try (INDArray arr = Nd4j.readBinary(f)) {   //Using the fact that INDArray.close() exists -> deallocate memory as soon as reading is done
            if(allowableDataTypes != null){
                ArrayUtils.contains(allowableDataTypes, arr.dataType());
            }
        } catch (IOException e) {
            return ValidationResult.builder()
                    .valid(false)
                    .formatType("INDArray File")
                    .path(Nd4jCommonValidator.getPath(f))
                    .issues(Collections.singletonList("Unable to read file (IOException)"))
                    .exception(e)
                    .build();
        } catch (Throwable t) {
            if (t instanceof OutOfMemoryError || t.getMessage().toLowerCase().contains("failed to allocate")) {
                //This is a memory exception during reading... result is indeterminant (might be valid, might not be, can't tell here)
                return ValidationResult.builder()
                        .valid(true)
                        .formatType("INDArray File")
                        .path(Nd4jCommonValidator.getPath(f))
                        .build();
            }

            return ValidationResult.builder()
                    .valid(false)
                    .formatType("INDArray File")
                    .path(Nd4jCommonValidator.getPath(f))
                    .issues(Collections.singletonList("File may be corrupt or is not a binary INDArray file"))
                    .exception(t)
                    .build();
        }

        return ValidationResult.builder()
                .valid(true)
                .formatType("INDArray File")
                .path(Nd4jCommonValidator.getPath(f))
                .build();
    }

    public static ValidationResult validateINDArrayTextFile(@NonNull File f){

        ValidationResult vr = Nd4jCommonValidator.isValidFile(f, "INDArray Text File", false);
        if(vr != null && !vr.isValid())
            return vr;

        //TODO let's do this without reading the whole thing into memory - check header + length...
        try (INDArray arr = Nd4j.readTxt(f.getPath())) {   //Using the fact that INDArray.close() exists -> deallocate memory as soon as reading is done
        } catch (Throwable t) {
            if (t instanceof OutOfMemoryError || t.getMessage().toLowerCase().contains("failed to allocate")) {
                //This is a memory exception during reading... result is indeterminant (might be valid, might not be, can't tell here)
                return ValidationResult.builder()
                        .valid(true)
                        .formatType("INDArray Text File")
                        .path(Nd4jCommonValidator.getPath(f))
                        .build();
            }

            return ValidationResult.builder()
                    .valid(false)
                    .formatType("INDArray Text File")
                    .path(Nd4jCommonValidator.getPath(f))
                    .issues(Collections.singletonList("File may be corrupt or is not a binary INDArray file"))
                    .exception(t)
                    .build();
        }

        return ValidationResult.builder()
                .valid(true)
                .formatType("INDArray Text File")
                .path(Nd4jCommonValidator.getPath(f))
                .build();
    }


    public static ValidationResult validateNpyFile(@NonNull File f){

        ValidationResult vr = Nd4jCommonValidator.isValidFile(f, "Numpy .npy File", false);
        if(vr != null && !vr.isValid())
            return vr;

        //TODO let's do this without reading whole thing into memory
        try (INDArray arr = Nd4j.createFromNpyFile(f)) {   //Using the fact that INDArray.close() exists -> deallocate memory as soon as reading is done
        } catch (Throwable t) {
            if (t instanceof OutOfMemoryError || t.getMessage().toLowerCase().contains("failed to allocate")) {
                //This is a memory exception during reading... result is indeterminant (might be valid, might not be, can't tell here)
                return ValidationResult.builder()
                        .valid(true)
                        .formatType("Numpy .npy File")
                        .path(Nd4jCommonValidator.getPath(f))
                        .build();
            }

            return ValidationResult.builder()
                    .valid(false)
                    .formatType("Numpy .npy File")
                    .path(Nd4jCommonValidator.getPath(f))
                    .issues(Collections.singletonList("File may be corrupt or is not a Numpy .npy file"))
                    .exception(t)
                    .build();
        }

        return ValidationResult.builder()
                .valid(true)
                .formatType("Numpy .npy File")
                .path(Nd4jCommonValidator.getPath(f))
                .build();
    }

    public static ValidationResult validateNpzFile(@NonNull File f){
        ValidationResult vr = Nd4jCommonValidator.isValidFile(f, "Numpy .npz File", false);
        if(vr != null && !vr.isValid())
            return vr;

        Map<String,INDArray> m = null;
        try{
            m = Nd4j.createFromNpzFile(f);
        } catch (Throwable t){
            return ValidationResult.builder()
                    .valid(false)
                    .formatType("Numpy .npz File")
                    .path(Nd4jCommonValidator.getPath(f))
                    .issues(Collections.singletonList("File may be corrupt or is not a Numpy .npz file"))
                    .exception(t)
                    .build();
        } finally {
            //Deallocate immediately
            if(m != null){
                for(INDArray arr : m.values()){
                    if(arr != null){
                        arr.close();
                    }
                }
            }
        }

        return ValidationResult.builder()
                .valid(true)
                .formatType("Numpy .npz File")
                .path(Nd4jCommonValidator.getPath(f))
                .build();
    }

    public static ValidationResult validateNumpyTxtFile(@NonNull File f, @NonNull String delimiter, @NonNull Charset charset){
        ValidationResult vr = Nd4jCommonValidator.isValidFile(f, "Numpy text file", false);
        if(vr != null && !vr.isValid())
            return vr;

        String s;
        try{
            s = FileUtils.readFileToString(f, charset);
        } catch (Throwable t){
            return ValidationResult.builder()
                    .valid(false)
                    .formatType("Numpy text file")
                    .path(Nd4jCommonValidator.getPath(f))
                    .issues(Collections.singletonList("File may be corrupt or is not a Numpy text file"))
                    .exception(t)
                    .build();
        }

        String[] lines = s.split("\n");
        int countPerLine = 0;
        for(int i=0; i<lines.length; i++ ){
            String[] lineSplit = lines[i].split(delimiter);
            if(i == 0 ){
                countPerLine = lineSplit.length;
            } else if(!lines[i].isEmpty()){
                if(countPerLine != lineSplit.length){
                    return ValidationResult.builder()
                            .valid(false)
                            .formatType("Numpy text file")
                            .path(Nd4jCommonValidator.getPath(f))
                            .issues(Collections.singletonList("Number of values in each line is not the same for all lines: File may be corrupt, is not a Numpy text file, or delimiter \"" + delimiter + "\" is incorrect"))
                            .build();
                }
            }

            for( int j=0; j<lineSplit.length; j++ ){
                try{
                    Double.parseDouble(lineSplit[j]);
                } catch (NumberFormatException e){
                    return ValidationResult.builder()
                            .valid(false)
                            .formatType("Numpy text file")
                            .path(Nd4jCommonValidator.getPath(f))
                            .issues(Collections.singletonList("File may be corrupt, is not a Numpy text file, or delimiter \"" + delimiter + "\" is incorrect"))
                            .exception(e)
                            .build();
                }
            }
        }
        return ValidationResult.builder()
                .valid(true)
                .formatType("Numpy text file")
                .path(Nd4jCommonValidator.getPath(f))
                .build();
    }


    public static ValidationResult validateSameDiffFlatBuffers(@NonNull File f){
        ValidationResult vr = Nd4jCommonValidator.isValidFile(f, "SameDiff FlatBuffers file", false);
        if(vr != null && !vr.isValid())
            return vr;

        try{
            byte[] bytes;
            try (InputStream is = new BufferedInputStream(new FileInputStream(f))) {
                bytes = IOUtils.toByteArray(is);
            }

            ByteBuffer bbIn = ByteBuffer.wrap(bytes);
            FlatGraph fg = FlatGraph.getRootAsFlatGraph(bbIn);
            int vl = fg.variablesLength();
            int ol = fg.nodesLength();
            System.out.println();
        } catch (Throwable t){
            return ValidationResult.builder()
                    .valid(false)
                    .formatType("SameDiff FlatBuffers file")
                    .path(Nd4jCommonValidator.getPath(f))
                    .issues(Collections.singletonList("File may be corrupt or is not a SameDiff file in FlatBuffers format"))
                    .exception(t)
                    .build();
        }

        return ValidationResult.builder()
                .valid(true)
                .formatType("SameDiff FlatBuffers file")
                .path(Nd4jCommonValidator.getPath(f))
                .build();
    }
}
