/*
 *  ******************************************************************************
 *  *
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

package org.deeplearning4j.util;

import lombok.NonNull;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.common.validation.Nd4jCommonValidator;
import org.nd4j.common.validation.ValidationResult;

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

/**
 * A utility for validating Deeplearning4j Serialized model file formats
 *
 * @author Alex Black
 */
public class DL4JModelValidator {

    private DL4JModelValidator(){ }

    /**
     * Validate whether the file represents a valid MultiLayerNetwork saved previously with {@link MultiLayerNetwork#save(File)}
     * or {@link ModelSerializer#writeModel(Model, File, boolean)}, to be read with {@link MultiLayerNetwork#load(File, boolean)}
     *
     * @param f File that should represent an saved MultiLayerNetwork
     * @return Result of validation
     */
    public static ValidationResult validateMultiLayerNetwork(@NonNull File f){

        List<String> requiredEntries = Arrays.asList(ModelSerializer.CONFIGURATION_JSON, ModelSerializer.COEFFICIENTS_BIN);     //TODO no-params models... might be OK to have no params, but basically useless in practice

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

    /**
     * Validate whether the file represents a valid ComputationGraph saved previously with {@link ComputationGraph#save(File)}
     * or {@link ModelSerializer#writeModel(Model, File, boolean)}, to be read with {@link ComputationGraph#load(File, boolean)}
     *
     * @param f File that should represent an saved MultiLayerNetwork
     * @return Result of validation
     */
    public static ValidationResult validateComputationGraph(@NonNull File f){

        List<String> requiredEntries = Arrays.asList(ModelSerializer.CONFIGURATION_JSON, ModelSerializer.COEFFICIENTS_BIN);     //TODO no-params models... might be OK to have no params, but basically useless in practice

        ValidationResult vr = Nd4jCommonValidator.isValidZipFile(f, false, requiredEntries);
        if(vr != null && !vr.isValid()) {
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
            ComputationGraphConfiguration.fromJson(config);
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
}
