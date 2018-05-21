/*-
 *
 *  * Copyright 2017 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
package org.deeplearning4j.nn.modelimport.keras.e2e;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasSpaceToDepth;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

/**
 * Import previously stored YOLO9000 Keras net from https://github.com/allanzelener/YAD2K.
 * <p>
 * git clone https://github.com/allanzelener/YAD2K
 * cd YAD2K
 * wget http://pjreddie.com/media/files/yolo.weights
 * wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolo.cfg
 * python3 yad2k.py yolo.cfg yolo.weights yolo.h5
 * <p>
 * To run this test put the output of this script on the test resources path.
 *
 * @author Max Pumperla
 */
@Slf4j
public class KerasYolo9000PredictTest {

    private static final String DL4J_MODEL_FILE_NAME = ".";
    private static ImagePreProcessingScaler IMAGE_PREPROCESSING_SCALER = new ImagePreProcessingScaler(0, 1);

    @Ignore
    @Test
    public void testYoloPredictionImport() throws Exception {


        int HEIGHT = 416;
        int WIDTH = 416;
        INDArray indArray = Nd4j.create(HEIGHT, WIDTH, 3);
        IMAGE_PREPROCESSING_SCALER.transform(indArray);

        KerasLayer.registerCustomLayer("Lambda", KerasSpaceToDepth.class);

        String h5_FILENAME = "modelimport/keras/examples/yolo/yolo-voc.h5";
        ComputationGraph graph = KerasModelImport.importKerasModelAndWeights(h5_FILENAME, false);

        double[][] priorBoxes = {{1.3221, 1.73145}, {3.19275, 4.00944}, {5.05587, 8.09892}, {9.47112, 4.84053}, {11.2364, 10.0071}};
        INDArray priors = Nd4j.create(priorBoxes);

        ComputationGraph model = new TransferLearning.GraphBuilder(graph)
                .addLayer("outputs",
                        new org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer.Builder()
                                .boundingBoxPriors(priors)
                                .build(),
                        "conv2d_23")
                .setOutputs("outputs")
                .build();

        ModelSerializer.writeModel(model, DL4J_MODEL_FILE_NAME, false);

        ComputationGraph computationGraph = ModelSerializer.restoreComputationGraph(new File(DL4J_MODEL_FILE_NAME));

        System.out.println(computationGraph.summary(InputType.convolutional(416, 416, 3)));

        INDArray results = computationGraph.outputSingle(indArray);


    }

}

