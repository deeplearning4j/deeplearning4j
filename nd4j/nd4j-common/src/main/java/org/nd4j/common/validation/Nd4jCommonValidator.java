/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.common.validation;

import lombok.NonNull;
import org.apache.commons.io.FileUtils;
import org.nd4j.shade.jackson.databind.JavaType;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * A utility for validating multiple file formats that ND4J and other tools can read
 *
 * @author Alex Black
 */
public class Nd4jCommonValidator {

    private Nd4jCommonValidator() {
    }

    /**
     * Validate whether the specified file is a valid file (must exist and be non-empty)
     *
     * @param f File to check
     * @return Result of validation
     */
    public static ValidationResult isValidFile(@NonNull File f) {
        ValidationResult vr = isValidFile(f, "File", false);
        if (vr != null)
            return vr;
        return ValidationResult.builder()
                .valid(true)
                .formatType("File")
                .path(getPath(f))
                .build();
    }

    /**
     * Validate whether the specified file is a valid file
     *
     * @param f          File to check
     * @param formatType Name of the file format to include in validation results
     * @param allowEmpty If true: allow empty files to pass. False: empty files will fail validation
     * @return Result of validation
     */
    public static ValidationResult isValidFile(@NonNull File f, String formatType, boolean allowEmpty) {
        String path;
        try {
            path = f.getAbsolutePath(); //Very occasionally: getAbsolutePath not possible (files in JARs etc)
        } catch (Throwable t) {
            path = f.getPath();
        }

        if (f.exists() && !f.isFile()) {
            return ValidationResult.builder()
                    .valid(false)
                    .formatType(formatType)
                    .path(path)
                    .issues(Collections.singletonList(f.isDirectory() ? "Specified path is a directory" : "Specified path is not a file"))
                    .build();
        }

        if (!f.exists() || !f.isFile()) {
            return ValidationResult.builder()
                    .valid(false)
                    .formatType(formatType)
                    .path(path)
                    .issues(Collections.singletonList("File does not exist"))
                    .build();
        }

        if (!allowEmpty && f.length() <= 0) {
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

    /**
     * Validate whether the specified file is a valid JSON file. Note that this does not match the JSON content against a specific schema
     *
     * @param f       File to check
     * @param charset Character set for file
     * @return Result of validation
     */
    public static ValidationResult isValidJson(@NonNull File f, Charset charset) {

        ValidationResult vr = isValidFile(f, "JSON", false);
        if (vr != null)
            return vr;

        String content;
        try {
            content = FileUtils.readFileToString(f, charset);
        } catch (IOException e) {
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

    /**
     * Validate whether the specified String is valid JSON. Note that this does not match the JSON content against a specific schema
     *
     * @param s JSON String to check
     * @return Result of validation
     */
    public static ValidationResult isValidJSON(String s) {
        return isValidJson(s, null);
    }


    protected static ValidationResult isValidJson(String content, File f) {
        try {
            ObjectMapper om = new ObjectMapper();
            JavaType javaType = om.getTypeFactory().constructMapType(Map.class, String.class, Object.class);
            om.readValue(content, javaType);    //Don't care about result, just that it can be parsed successfully
        } catch (Throwable t) {
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


    /**
     * Validate whether the specified file is a valid Zip file
     *
     * @param f          File to check
     * @param allowEmpty If true: allow empty zip files to pass validation. False: empty zip files will fail validation.
     * @return Result of validation
     */
    public static ValidationResult isValidZipFile(@NonNull File f, boolean allowEmpty) {
        return isValidZipFile(f, allowEmpty, (List<String>) null);
    }

    /**
     * Validate whether the specified file is a valid Zip file
     *
     * @param f          File to check
     * @param allowEmpty If true: allow empty zip files to pass validation. False: empty zip files will fail validation.
     * @return Result of validation
     */
    public static ValidationResult isValidZipFile(@NonNull File f, boolean allowEmpty, String... requiredEntries) {
        return isValidZipFile(f, allowEmpty, requiredEntries == null ? null : Arrays.asList(requiredEntries));
    }

    /**
     * Validate whether the specified file is a valid Zip file, and contains all of the required entries
     *
     * @param f               File to check
     * @param allowEmpty      If true: allow empty zip files to pass validation. False: empty zip files will fail validation.
     * @param requiredEntries If non-null, all of the specified entries must be present for the file to pass validation
     * @return Result of validation
     */
    public static ValidationResult isValidZipFile(@NonNull File f, boolean allowEmpty, List<String> requiredEntries) {
        ValidationResult vr = isValidFile(f, "Zip File", false);
        if (vr != null)
            return vr;

        ZipFile zf;
        try {
            zf = new ZipFile(f);
        } catch (Throwable e) {
            return ValidationResult.builder()
                    .valid(false)
                    .formatType("Zip File")
                    .path(getPath(f))
                    .issues(Collections.singletonList("File does not appear to be valid zip file (not a zip file or content is corrupt)"))
                    .exception(e)
                    .build();
        }

        try {
            int numEntries = zf.size();
            if (!allowEmpty && numEntries <= 0) {
                return ValidationResult.builder()
                        .valid(false)
                        .formatType("Zip File")
                        .path(getPath(f))
                        .issues(Collections.singletonList("Zip file is empty"))
                        .build();
            }

            if (requiredEntries != null && !requiredEntries.isEmpty()) {
                List<String> missing = null;
                for (String s : requiredEntries) {
                    ZipEntry ze = zf.getEntry(s);
                    if (ze == null) {
                        if (missing == null)
                            missing = new ArrayList<>();
                        missing.add(s);
                    }
                }

                if (missing != null) {
                    String s = "Zip file is missing " + missing.size() + " of " + requiredEntries.size() + " required entries: " + missing;
                    return ValidationResult.builder()
                            .valid(false)
                            .formatType("Zip File")
                            .path(getPath(f))
                            .issues(Collections.singletonList(s))
                            .build();
                }
            }

        } catch (Throwable t) {
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
            } catch (IOException e) {
            }  //Ignore, can't do anything about it...
        }

        return ValidationResult.builder()
                .valid(true)
                .formatType("Zip File")
                .path(getPath(f))
                .build();
    }


    /**
     * Null-safe and "no absolute path exists" safe method for getting the path of a file for validation purposes
     */
    public static String getPath(File f) {
        if (f == null)
            return null;
        try {
            return f.getAbsolutePath(); //Very occasionally: getAbsolutePath not possible (files in JARs etc)
        } catch (Throwable t) {
            return f.getPath();
        }
    }


}
