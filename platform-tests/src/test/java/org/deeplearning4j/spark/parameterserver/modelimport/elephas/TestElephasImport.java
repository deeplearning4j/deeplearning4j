/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.spark.parameterserver.modelimport.elephas;

import com.sun.jna.Platform;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.parameterserver.BaseSparkTest;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.resources.Downloader;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;

import java.io.File;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

import static java.io.File.createTempFile;
import static org.junit.jupiter.api.Assertions.assertTrue;
@Tag(TagNames.FILE_IO)
@Tag(TagNames.SPARK)
@Tag(TagNames.DIST_SYSTEMS)
@NativeTag
@Slf4j
public class TestElephasImport extends BaseSparkTest {


    @BeforeAll
    @SneakyThrows
    public static void beforeAll() {
        if(Platform.isWindows()) {
            File hadoopHome = new File(System.getProperty("java.io.tmpdir"),"hadoop-tmp");
            File binDir = new File(hadoopHome,"bin");
            if(!binDir.exists())
                binDir.mkdirs();
            File outputFile = new File(binDir,"winutils.exe");
            if(!outputFile.exists()) {
                log.info("Fixing spark for windows");
                Downloader.download("winutils.exe",
                        URI.create("https://github.com/cdarlint/winutils/blob/master/hadoop-2.6.5/bin/winutils.exe?raw=true").toURL(),
                        outputFile,"db24b404d2331a1bec7443336a5171f1",3);
            }

            System.setProperty("hadoop.home.dir", hadoopHome.getAbsolutePath());
        }
    }

    @Test
    public void testElephasSequentialImport() throws Exception {
        String modelPath = "modelimport/elephas/elephas_sequential.h5";
        SparkDl4jMultiLayer model = importElephasSequential(sc, modelPath);
        // System.out.println(model.getNetwork().summary());
        assertTrue(model.getTrainingMaster() instanceof ParameterAveragingTrainingMaster);
    }

    @Test
    public void testElephasSequentialImportAsync() throws Exception {
        if(Platform.isWindows()) {
            //Spark tests don't run on windows
            return;
        }
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
