/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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

package org.deeplearning4j.nn.modelimport.keras;

import org.apache.commons.io.FileUtils;
import org.datavec.python.keras.Model;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.resources.Resources;

import java.io.File;
import java.util.Arrays;
import java.util.Collection;


@RunWith(Parameterized.class)
public class TestTFKerasModelImport extends BaseDL4JTest{

    private File modelFile;

    @Parameterized.Parameters
    public static Object[] params() throws Exception {
        return new File[]{Resources.asFile("modelimport/keras/tfkeras/reshape.h5"),
                          Resources.asFile("modelimport/keras/tfkeras/reshape.h5")};
//       return FileUtils.listFiles(Resources.asFile("modelimport/keras/tfkeras"),
//                                  new String[]{"h5"}, true).toArray(new File[0]);
    }

    public TestTFKerasModelImport(File modelFile){
        this.modelFile = modelFile;
    }

    @Test
    public void testModelImport() throws Exception{
        testModelImport(modelFile);
    }

    private void testModelImport(File modelFile) throws Exception{
        testModelImport(modelFile.getAbsolutePath());
    }
    private void testModelImport(String path) throws Exception{
        Model kerasModel = new Model(path);
        ComputationGraph dl4jModel = KerasModelImport.importKerasModelAndWeights(path);
        Assert.assertEquals(kerasModel.numInputs(), dl4jModel.getNumInputArrays());
        Assert.assertEquals(kerasModel.numOutputs(), dl4jModel.getNumOutputArrays());
        INDArray[] inputArrays = new INDArray[kerasModel.numInputs()];
        for (int i = 0; i < inputArrays.length; i ++){
            long[] shape = kerasModel.inputShapeAt(i);
            for (int j = 0; j < shape.length; j++){
                if (shape[j] < 0){
                    shape[j] = 1;
                }
            }
            inputArrays[i] = Nd4j.rand(shape);
        }

        INDArray[] kerasOut = kerasModel.predict(inputArrays);
        INDArray[] dl4jOut = dl4jModel.output(inputArrays);

        Assert.assertEquals(kerasOut.length, dl4jOut.length);

        for (int i = 0; i < kerasOut.length; i++){
            INDArray kerasOutArr = kerasOut[i].reshape(kerasOut[i].shape()[0], -1);  // bit of relaxation on shape
            INDArray dl4jOutArr = dl4jOut[i].reshape(dl4jOut[i].shape()[0], -1);
            Assert.assertEquals(kerasOutArr, dl4jOutArr);
        }
    }
}
