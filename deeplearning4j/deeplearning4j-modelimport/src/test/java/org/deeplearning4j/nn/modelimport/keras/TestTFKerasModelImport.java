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

package org.deeplearning4j.nn.modelimport.keras;

import org.apache.commons.io.FileUtils;
import org.datavec.python.keras.Model;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Assert;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.common.resources.Resources;
import org.nd4j.common.tests.ResourceUtils;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.List;


@RunWith(Parameterized.class)
public class TestTFKerasModelImport extends BaseDL4JTest{

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    private String modelFile;

    @Override
    public long getTimeoutMilliseconds(){
        return 300000;
    } // installing TF will take a while


    @Parameterized.Parameters(name = "file={0}")
    public static Object[] params() throws Exception {
        List<String> paths = ResourceUtils.listClassPathFiles("modelimport/keras/tfkeras", true, false);
       return paths.toArray(new String[0]);
    }

    public TestTFKerasModelImport(String modelFile){
        this.modelFile = modelFile;
    }

    @Test
    public void testModelImport() throws Exception{
        testModelImportWithData(modelFile);
    }

    private void testModelImportWithData(String path) throws Exception{
        System.out.println(path);
        // TODO multi input/output
        INDArray inputArray;
        INDArray expectedOutputArray;
        File f = Resources.asFile(path);        //May in in JAR that HDF5 can't read from
        File modelFile = new File(testDir.getRoot(), f.getName());
        FileUtils.copyFile(f, modelFile);

        synchronized (Hdf5Archive.LOCK_OBJECT){
            Hdf5Archive hdf5Archive = new Hdf5Archive(modelFile.getAbsolutePath());
            List<String> rootGroups = hdf5Archive.getGroups();
            if (rootGroups.contains("data")){
                String inputName = hdf5Archive.readAttributeAsString("input_names", "data");
                String outputName = hdf5Archive.readAttributeAsString("output_names", "data");
                inputArray = hdf5Archive.readDataSet(inputName, "data");
                expectedOutputArray = hdf5Archive.readDataSet(outputName, "data");
            }
            else{
                hdf5Archive.close();
                return;
            }
            hdf5Archive.close();
        }
        INDArray outputArray;

        ComputationGraph dl4jModel = KerasModelImport.importKerasModelAndWeights(path);
        outputArray = dl4jModel.outputSingle(inputArray);

        expectedOutputArray = expectedOutputArray.castTo(DataType.FLOAT);
        outputArray = outputArray.castTo(DataType.FLOAT);
        if (path.contains("misc_")){
            //shape relaxation
            expectedOutputArray = expectedOutputArray.reshape( -1);
            outputArray = outputArray.reshape(-1);
        }

        System.out.println(outputArray.toString());
        System.out.println(expectedOutputArray.toString());
        Assert.assertArrayEquals(expectedOutputArray.shape(), outputArray.shape());
        Assert.assertTrue(expectedOutputArray.equalsWithEps(outputArray, 1e-3));
    }

    private void testModelImportWithKeras(String path) throws Exception{
        Model kerasModel = new Model(path);
        ComputationGraph dl4jModel = KerasModelImport.importKerasModelAndWeights(path);
        Assert.assertEquals(kerasModel.numInputs(), dl4jModel.getNumInputArrays());
        Assert.assertEquals(kerasModel.numOutputs(), dl4jModel.getNumOutputArrays());
        INDArray[] kerasInputArrays = new INDArray[kerasModel.numInputs()];
        INDArray[] dl4jInputArrays = new INDArray[kerasModel.numInputs()];

        for (int i = 0; i < kerasInputArrays.length; i ++) {
            long[] shape = kerasModel.inputShapeAt(i);
            for (int j = 0; j < shape.length; j++) {
                if (shape[j] < 0) {
                    shape[j] = 1;
                }
            }

            kerasInputArrays[i] = Nd4j.rand(shape);
        }

        INDArray[] kerasOut = kerasModel.predict(kerasInputArrays);
        INDArray[] dl4jOut = dl4jModel.output(dl4jInputArrays);

        Assert.assertEquals(kerasOut.length, dl4jOut.length);

        for (int i = 0; i < kerasOut.length; i++){
            INDArray kerasOutArr = kerasOut[i];
            kerasOutArr = kerasOutArr.reshape(1, -1);// bit of relaxation on shape
            kerasOutArr= kerasOutArr.castTo(DataType.DOUBLE);
            Nd4j.getAffinityManager().ensureLocation(dl4jOut[i], AffinityManager.Location.HOST);
            INDArray dl4jOutArr = dl4jOut[i].reshape(1, -1);
            System.out.println(kerasOutArr.shapeInfoToString());
            System.out.println(dl4jOutArr.shapeInfoToString());
            Assert.assertEquals(kerasOutArr, dl4jOutArr);
        }
    }
}
