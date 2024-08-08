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
package org.eclipse.deeplearning4j.omnihub;

import lombok.SneakyThrows;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.omnihub.OmnihubConfig;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter;
import org.nd4j.samediff.frameworkimport.tensorflow.importer.TensorflowFrameworkImporter;

import java.io.File;
import java.io.IOException;
import java.util.Collections;

/**
 * Main class for taking the local downloads of an omnihub download directory
 * and converting them to output formats.
 *
 * @author Adam Gibson
 */
public class BootstrapFromLocal {

    public static void main(String...args) {
        File localOmnihubHome = OmnihubConfig.getOmnihubHome();
        File[] frameworks = localOmnihubHome.listFiles();
        OnnxFrameworkImporter onnxFrameworkImporter = new OnnxFrameworkImporter();
        TensorflowFrameworkImporter tensorflowFrameworkImporter = new TensorflowFrameworkImporter();
        for(File frameworkFile : frameworks) {
            Framework framework = Framework.valueOf(frameworkFile.getName().toUpperCase());
            if(Framework.isInput(framework)) {
                File[] inputFiles = frameworkFile.listFiles();
                for(File inputFile : inputFiles) {
                    try {
                        extracted(localOmnihubHome, onnxFrameworkImporter, tensorflowFrameworkImporter, framework, inputFile);
                    } catch (Exception e) {
                        System.err.println("Failed to import model at path " + inputFile.getAbsolutePath());
                        e.printStackTrace();
                    }
                }

            }
        }

    }

    private static void extracted(File localOmnihubHome, OnnxFrameworkImporter onnxFrameworkImporter,
                                  TensorflowFrameworkImporter tensorflowFrameworkImporter,
                                  Framework framework,
                                  File inputFile) throws Exception {
        String inputFileNameMinusFormat = FilenameUtils.getBaseName(inputFile.getName());
        String format = FilenameUtils.getExtension(inputFile.getName());
        Framework outputFramework = Framework.outputFrameworkFor(framework);
        File saveModelDir = new File(localOmnihubHome, outputFramework.name().toLowerCase());
        if(!saveModelDir.exists()) {
            saveModelDir.mkdirs();
        }
        switch(outputFramework) {
            case SAMEDIFF:
                importTfOnnxSameDiff(onnxFrameworkImporter, tensorflowFrameworkImporter, framework, inputFile, inputFileNameMinusFormat, format, saveModelDir);
                break;
            case DL4J:
                File saveModel2 = new File(saveModelDir,inputFileNameMinusFormat + ".zip");
                //filter out invalid file formats
                if(format.equals("h5")) {
                    importKerasDl4j(inputFile, saveModel2);
                }

                break;
        }


    }

    private static void importTfOnnxSameDiff(OnnxFrameworkImporter onnxFrameworkImporter, TensorflowFrameworkImporter tensorflowFrameworkImporter, Framework framework, File inputFile, String inputFileNameMinusFormat, String format, File saveModelDir) throws IOException {
        SameDiff sameDiff = null;
        switch(framework) {
            case ONNX:
            case PYTORCH:
                //filter out invalid files
                if(format.equals("onnx"))
                    sameDiff = onnxFrameworkImporter.runImport(inputFile.getAbsolutePath(), Collections.emptyMap(),true, false);
                break;
            case TENSORFLOW:
                if(format.equals("pb"))
                    sameDiff = tensorflowFrameworkImporter.runImport(inputFile.getAbsolutePath(), Collections.emptyMap(),true, false);
                break;
        }

        //reuse the same model name but with the samediff format
        File saveModel = new File(saveModelDir, inputFileNameMinusFormat + ".fb");
        if(sameDiff != null)
            sameDiff.asFlatFile(saveModel,true);
        else {
            System.err.println("Skipping model " + inputFile.getAbsolutePath());
        }
    }

    @SneakyThrows
    private static void importKerasDl4j(File inputFile, File saveModel2) {
        try {
            ComputationGraph computationGraph = KerasModelImport.importKerasModelAndWeights(inputFile.getAbsolutePath(),true);
            computationGraph.save(saveModel2,true);
        }catch(Exception e) {
            if(e instanceof InvalidKerasConfigurationException) {
                e.printStackTrace();
            } else {
                MultiLayerNetwork multiLayerNetwork = KerasModelImport.importKerasSequentialModelAndWeights(inputFile.getAbsolutePath(), true);
                multiLayerNetwork.save(saveModel2,true);
            }


        }
    }

}
