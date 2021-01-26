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

package org.deeplearning4j.nn.modelimport.keras.utils;

import lombok.NonNull;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.Hdf5Archive;
import org.deeplearning4j.nn.modelimport.keras.config.KerasModelConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.common.validation.Nd4jCommonValidator;
import org.nd4j.common.validation.ValidationResult;

import java.io.File;
import java.util.Collections;

/**
 * A utility for validating serialized Keras sequential and functional models for import into DL4J
 *
 * @author Alex Black
 */
public class DL4JKerasModelValidator {

    private DL4JKerasModelValidator(){ }

    /**
     * Validate whether the file represents a valid Keras Sequential model (HDF5 archive)
     *
     * @param f File that should represent an saved Keras Sequential model (HDF5 archive)
     * @return Result of validation
     */
    public static ValidationResult validateKerasSequential(@NonNull File f){
        return validateKeras(f, "Keras Sequential Model HDF5", MultiLayerNetwork.class);
    }

    /**
     * Validate whether the file represents a valid Keras Functional model (HDF5 archive)
     *
     * @param f File that should represent an saved Keras Functional model (HDF5 archive)
     * @return Result of validation
     */
    public static ValidationResult validateKerasFunctional(@NonNull File f){
        return validateKeras(f, "Keras Functional Model HDF5", ComputationGraph.class);
    }

    protected static ValidationResult validateKeras(@NonNull File f, String format, Class<?> cl){
        ValidationResult vr = Nd4jCommonValidator.isValidFile(f, format, false);
        if(vr != null && !vr.isValid()) {
            return vr;
        }

        KerasModelConfiguration c = new KerasModelConfiguration();
        Hdf5Archive archive = null;
        try{
            archive = new Hdf5Archive(f.getPath());

            //Check JSON
            try{
                String json = archive.readAttributeAsJson(c.getTrainingModelConfigAttribute());
                vr = Nd4jCommonValidator.isValidJSON(json);
                if(vr != null && !vr.isValid()){
                    vr.setFormatType(format);
                    return vr;
                }
            } catch (Throwable t){
                return ValidationResult.builder()
                        .formatType(format)
                        .formatClass(cl)
                        .valid(false)
                        .path(Nd4jCommonValidator.getPath(f))
                        .issues(Collections.singletonList("Unable to read JSON configuration from Keras Sequential model HDF5 file"))
                        .exception(t)
                        .build();
            }

        } catch (Throwable t){
            return ValidationResult.builder()
                    .formatType(format)
                    .formatClass(cl)
                    .valid(false)
                    .path(Nd4jCommonValidator.getPath(f))
                    .issues(Collections.singletonList("Unable to read from " + format + " file - file is corrupt or not a valid Keras HDF5 archive?"))
                    .exception(t)
                    .build();
        }


        return ValidationResult.builder()
                .formatType(format)
                .formatClass(cl)
                .valid(true)
                .path(Nd4jCommonValidator.getPath(f))
                .build();
    }
}
