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
package org.eclipse.deeplearning4j.frameworkimport.frameworkimport.onnx;

import onnx.Onnx;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.samediff.frameworkimport.onnx.OnnxConverter;
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

@Tag(TagNames.ONNX)
public class TestOnnxConverter {

    @TempDir
    Path tempDir;


    @Test
    public void testOnnxTraining() throws Exception {
        ClassPathResource classPathResource = new ClassPathResource("onnx_graphs/output_cnn_mnist.onnx");
        OnnxFrameworkImporter onnxFrameworkImporter = new OnnxFrameworkImporter();
        Map<String, INDArray> arr = new HashMap<>();
        arr.put("label", Nd4j.ones(10));
        arr.put("input.1",Nd4j.ones(1,1,28,28));
        SameDiff sameDiff = onnxFrameworkImporter.runImport(classPathResource.getFile().getAbsolutePath(),arr, true);
        SDVariable labels = sameDiff.placeHolder("labels", DataType.FLOAT);
        sameDiff.setEagerMode(false);

        SDVariable sdVariable = sameDiff.loss().softmaxCrossEntropy(labels, sameDiff.getVariable("22"),sameDiff.constant(1.0f));
        TrainingConfig trainingConfig = TrainingConfig.builder()
                .dataSetFeatureMapping("input.1")
                .dataSetLabelMapping(labels.name())
                .updater(new Adam())
                .lossVariables(Collections.singletonList(sdVariable.name()))
                .build();
        sameDiff.setTrainingConfig(trainingConfig);
        sameDiff.prepareForTraining();
        System.out.println(sameDiff.summary(true));
        DataSetIterator setIterator = new MnistDataSetIterator(10,10,true);
        setIterator.setPreProcessor((DataSetPreProcessor) toPreProcess -> toPreProcess.setFeatures(toPreProcess.getFeatures().reshape(10,1,28,28)));

        sameDiff.fit(setIterator,1);
    }

    @Test
    public void test() throws Exception {
        ClassPathResource classPathResource = new ClassPathResource("mnist.onnx");
        File f = classPathResource.getFile();
        OnnxConverter onnxConverter = new OnnxConverter();
        Onnx.ModelProto modelProto = Onnx.ModelProto.parseFrom(new FileInputStream(f));
        Onnx.GraphProto graphProto = onnxConverter.addConstValueInfoToGraph(modelProto.getGraph());
        modelProto = modelProto.toBuilder().setGraph(graphProto).build();
        File newModel = new File(tempDir.toFile(),"postprocessed.onnx");
        IOUtils.write(modelProto.toByteArray(),new FileOutputStream(newModel));
        File file = new File(tempDir.toFile(), "newfile.onnx");
        onnxConverter.convertModel(newModel,file);
    }





}
