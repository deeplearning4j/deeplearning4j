/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.spark.parameterserver.modelimport.elephas;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.parameterserver.BaseSparkTest;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

import static java.io.File.createTempFile;
import static org.junit.Assert.assertTrue;

public class TestElephasImport extends BaseSparkTest {

    @Test
    public void testElephasSequentialImport() throws Exception {
        String modelPath = "modelimport/elephas/elephas_sequential.h5";
        SparkDl4jMultiLayer model = importElephasSequential(sc, modelPath);
        // System.out.println(model.getNetwork().summary());
        assertTrue(model.getTrainingMaster() instanceof ParameterAveragingTrainingMaster);
    }

    @Test
    public void testElephasSequentialImportAsync() throws Exception {
        String modelPath = "modelimport/elephas/elephas_sequential_async.h5";
        SparkDl4jMultiLayer model = importElephasSequential(sc, modelPath);
        // System.out.println(model.getNetwork().summary());
        assertTrue(model.getTrainingMaster() instanceof SharedTrainingMaster);
    }

    private SparkDl4jMultiLayer importElephasSequential(JavaSparkContext sc, String modelPath) throws Exception {

        ClassPathResource modelResource =
                new ClassPathResource(modelPath,
                        TestElephasImport.class.getClassLoader());
        File modelFile = createTempFile("tempModel", "h5");
        Files.copy(modelResource.getInputStream(), modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        SparkDl4jMultiLayer model = ElephasModelImport.importElephasSequentialModelAndWeights(sc, modelFile.getAbsolutePath());
        return model;
    }


    @Test
    public void testElephasModelImport() throws Exception {

        String modelPath = "modelimport/elephas/elephas_model.h5";
        SparkComputationGraph model = importElephasModel(sc, modelPath);
        // System.out.println(model.getNetwork().summary());
        assertTrue(model.getTrainingMaster() instanceof ParameterAveragingTrainingMaster);
    }

    @Test
    public void testElephasJavaAveragingModelImport() throws Exception {

        String modelPath = "modelimport/elephas/java_param_averaging_model.h5";
        SparkComputationGraph model = importElephasModel(sc, modelPath);
        // System.out.println(model.getNetwork().summary());
        assert model.getTrainingMaster() instanceof ParameterAveragingTrainingMaster;
    }

    @Test
    public void testElephasJavaSharingModelImport() throws Exception {

        String modelPath = "modelimport/elephas/java_param_sharing_model.h5";
        SparkComputationGraph model = importElephasModel(sc, modelPath);
        // System.out.println(model.getNetwork().summary());
        assert model.getTrainingMaster() instanceof SharedTrainingMaster;
    }
    
    @Test
    public void testElephasModelImportAsync() throws Exception {

        String modelPath = "modelimport/elephas/elephas_model_async.h5";
        SparkComputationGraph model = importElephasModel(sc, modelPath);
        // System.out.println(model.getNetwork().summary());
        assertTrue(model.getTrainingMaster() instanceof SharedTrainingMaster);
    }

    private SparkComputationGraph importElephasModel(JavaSparkContext sc, String modelPath) throws Exception {

        ClassPathResource modelResource =
                new ClassPathResource(modelPath,
                        TestElephasImport.class.getClassLoader());
        File modelFile = createTempFile("tempModel", "h5");
        Files.copy(modelResource.getInputStream(), modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        SparkComputationGraph model = ElephasModelImport.importElephasModelAndWeights(sc, modelFile.getAbsolutePath());
        return model;
    }
}
