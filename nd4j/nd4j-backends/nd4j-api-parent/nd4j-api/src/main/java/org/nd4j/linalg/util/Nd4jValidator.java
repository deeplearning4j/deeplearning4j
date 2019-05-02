package org.nd4j.linalg.util;

import lombok.NonNull;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.validation.Nd4jCommonValidator;
import org.nd4j.validation.ValidationResult;

import java.io.File;
import java.io.IOException;
import java.util.Collections;

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

}
