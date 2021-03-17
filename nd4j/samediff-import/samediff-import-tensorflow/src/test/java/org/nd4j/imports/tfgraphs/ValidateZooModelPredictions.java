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

package org.nd4j.imports.tfgraphs;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;

import org.junit.jupiter.api.Test;

import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.common.io.ClassPathResource;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.*;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Slf4j
@Disabled
public class ValidateZooModelPredictions extends BaseNd4jTestWithBackends {


    @Override
    public char ordering() {
        return 'c';
    }



    @BeforeEach
    public void before() {
        Nd4j.create(1);
        Nd4j.setDataType(DataType.DOUBLE);
        Nd4j.getRandom().setSeed(123);
    }

    @Override
    public long getTimeoutMilliseconds() {
        return Long.MAX_VALUE;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMobilenetV1(@TempDir Path testDir,Nd4jBackend backend) throws Exception {
        TFGraphTestZooModels.currentTestDir = testDir.toFile();

        //Load model
        String path = "tf_graphs/zoo_models/mobilenet_v1_0.5_128/tf_model.txt";
        File resource = new ClassPathResource(path).getFile();
        SameDiff sd = TFGraphTestZooModels.LOADER.apply(resource, "mobilenet_v1_0.5_128");

        //Load data
        //Because we don't have DataVec NativeImageLoader in ND4J tests due to circular dependencies, we'll load the image previously saved...
        File imgFile = new ClassPathResource("deeplearning4j-zoo/goldenretriever_rgb128_unnormalized_nchw_INDArray.bin").getFile();
        INDArray img = Nd4j.readBinary(imgFile).castTo(DataType.FLOAT);
        img = img.permute(0,2,3,1).dup();   //to NHWC

        //Mobilenet V1 - not sure, but probably using inception preprocessing
        //i.e., scale to (-1,1) range
        //Image is originally 0 to 255
        img.divi(255).subi(0.5).muli(2);

        double min = img.minNumber().doubleValue();
        double max = img.maxNumber().doubleValue();

        assertTrue(min >= -1 && min <= -0.6);
        assertTrue(max <= 1 && max >= 0.6);

        //Perform inference
        List<String> inputs = sd.inputs();
        assertEquals(1, inputs.size());

        String out = "MobilenetV1/Predictions/Softmax";
        Map<String,INDArray> m = sd.output(Collections.singletonMap(inputs.get(0), img), out);

        INDArray outArr = m.get(out);


        System.out.println("SHAPE: " + Arrays.toString(outArr.shape()));
        System.out.println(outArr);

        INDArray argmax = outArr.argMax(1);

        //Load labels
        List<String> labels = labels();

        int classIdx = argmax.getInt(0);
        String className = labels.get(classIdx);
        String expClass = "golden retriever";
        double prob = outArr.getDouble(classIdx);

        System.out.println("Predicted class: \"" + className + "\" - probability = " + prob);
        assertEquals(expClass, className);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testResnetV2(@TempDir Path testDir,Nd4jBackend backend) throws Exception {
        if(TFGraphTestZooModels.isPPC()){
            /*
            Ugly hack to temporarily disable tests on PPC only on CI
            Issue logged here: https://github.com/eclipse/deeplearning4j/issues/7657
            These will be re-enabled for PPC once fixed - in the mean time, remaining tests will be used to detect and prevent regressions
             */

            log.warn("TEMPORARILY SKIPPING TEST ON PPC ARCHITECTURE DUE TO KNOWN JVM CRASH ISSUES - SEE https://github.com/eclipse/deeplearning4j/issues/7657");
            //OpValidationSuite.ignoreFailing();
        }

        TFGraphTestZooModels.currentTestDir = testDir.toFile();

        //Load model
        String path = "tf_graphs/zoo_models/resnetv2_imagenet_frozen_graph/tf_model.txt";
        File resource = new ClassPathResource(path).getFile();
        SameDiff sd = TFGraphTestZooModels.LOADER.apply(resource, "resnetv2_imagenet_frozen_graph");

        //Load data
        //Because we don't have DataVec NativeImageLoader in ND4J tests due to circular dependencies, we'll load the image previously saved...
        File imgFile = new ClassPathResource("deeplearning4j-zoo/goldenretriever_rgb224_unnormalized_nchw_INDArray.bin").getFile();
        INDArray img = Nd4j.readBinary(imgFile).castTo(DataType.FLOAT);
        img = img.permute(0,2,3,1).dup();   //to NHWC

        //Resnet v2 - NO external normalization, just resize and center crop
        // https://github.com/tensorflow/models/blob/d32d957a02f5cffb745a4da0d78f8432e2c52fd4/research/tensorrt/tensorrt.py#L70
        // https://github.com/tensorflow/models/blob/1af55e018eebce03fb61bba9959a04672536107d/official/resnet/imagenet_preprocessing.py#L253-L256

        //Perform inference
        List<String> inputs = sd.inputs();
        assertEquals(1, inputs.size());

        String out = "softmax_tensor";
        Map<String,INDArray> m = sd.output(Collections.singletonMap(inputs.get(0), img), out);

        INDArray outArr = m.get(out);


        System.out.println("SHAPE: " + Arrays.toString(outArr.shape()));
        System.out.println(outArr);

        INDArray argmax = outArr.argMax(1);

        //Load labels
        List<String> labels = labels();

        int classIdx = argmax.getInt(0);
        String className = labels.get(classIdx);
        String expClass = "golden retriever";
        double prob = outArr.getDouble(classIdx);

        System.out.println("Predicted class: " + classIdx + " - \"" + className + "\" - probability = " + prob);
        assertEquals(expClass, className);
    }


    public static List<String> labels() throws Exception {
        File labelsFile = new ClassPathResource("tf_graphs/zoo_models/labels/imagenet_labellist.txt").getFile();
        List<String> labels = FileUtils.readLines(labelsFile, StandardCharsets.UTF_8);
        return labels;
    }
}
