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

package org.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;

import org.junit.jupiter.api.Test;

import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.graph.FlatConfiguration;
import org.nd4j.graph.FlatGraph;
import org.nd4j.graph.FlatNode;
import org.nd4j.graph.FlatVariable;
import org.nd4j.graph.IntPair;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling3DConfig;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.config.AMSGrad;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.AdaMax;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.learning.regularization.L1Regularization;
import org.nd4j.linalg.learning.regularization.L2Regularization;
import org.nd4j.linalg.learning.regularization.WeightDecay;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
public class FlatBufferSerdeTest extends BaseNd4jTestWithBackends {


    @Override
    public char ordering(){
        return 'c';
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBasic(@TempDir Path testDir,Nd4jBackend backend) throws Exception {
        SameDiff sd = SameDiff.create();
        INDArray arr = Nd4j.linspace(1,12,12).reshape(3,4);
        SDVariable in = sd.placeHolder("in", arr.dataType(), arr.shape() );
        SDVariable tanh = sd.math().tanh(in);
        tanh.markAsLoss();

        ByteBuffer bb = sd.asFlatBuffers(true);

        File f = Files.createTempFile(testDir,"some-file","bin").toFile();
        f.delete();

        try(FileChannel fc = new FileOutputStream(f, false).getChannel()){
            fc.write(bb);
        }

        byte[] bytes;
        try(InputStream is = new BufferedInputStream(new FileInputStream(f))){
            bytes = IOUtils.toByteArray(is);
        }
        ByteBuffer bbIn = ByteBuffer.wrap(bytes);

        FlatGraph fg = FlatGraph.getRootAsFlatGraph(bbIn);

        int numNodes = fg.nodesLength();
        int numVars = fg.variablesLength();
        List<FlatNode> nodes = new ArrayList<>(numNodes);
        for( int i=0; i<numNodes; i++ ){
            nodes.add(fg.nodes(i));
        }
        List<FlatVariable> vars = new ArrayList<>(numVars);
        for( int i=0; i<numVars; i++ ){
            vars.add(fg.variables(i));
        }

        FlatConfiguration conf = fg.configuration();

        int numOutputs = fg.outputsLength();
        List<IntPair> outputs = new ArrayList<>(numOutputs);
        for( int i=0; i<numOutputs; i++ ){
            outputs.add(fg.outputs(i));
        }

        assertEquals(2, numVars);
        assertEquals(1, numNodes);

        //Check placeholders:
        assertEquals(1, fg.placeholdersLength());
        assertEquals("in", fg.placeholders(0));

        //Check loss variables:
        assertEquals(sd.getLossVariables().size(), fg.lossVariablesLength());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSimple(@TempDir Path testDir,Nd4jBackend backend) throws Exception {
        for( int i = 0; i < 10; i++ ) {
            for(boolean execFirst : new boolean[]{false, true}) {
                log.info("Starting test: i={}, execFirst={}", i, execFirst);
                SameDiff sd = SameDiff.create();
                INDArray arr = Nd4j.linspace(1, 12, 12).reshape(3, 4);
                SDVariable in = sd.placeHolder("in", arr.dataType(), arr.shape());
                SDVariable x;
                switch (i) {
                    case 0:
                        //Custom op
                        x = sd.cumsum("out", in, false, false, 1);
                        break;
                    case 1:
                        //Transform
                        x = sd.math().tanh(in);
                        break;
                    case 2:
                    case 3:
                        //Reduction
                        x = sd.mean("x", in, i == 2, 1);
                        break;
                    case 4:
                        //Transform
                        x = sd.math().square(in);
                        break;
                    case 5:
                    case 6:
                        //Index reduction
                        x = sd.argmax("x", in, i == 5, 1);
                        break;
                    case 7:
                        //Scalar:
                        x = in.add(10);
                        break;
                    case 8:
                        //Reduce 3:
                        SDVariable y = sd.var("in2", Nd4j.linspace(1,12,12).muli(0.1).addi(0.5).reshape(3,4));
                        x = sd.math().cosineSimilarity(in, y);
                        break;
                    case 9:
                        //Reduce 3 (along dim)
                        SDVariable z = sd.var("in2", Nd4j.linspace(1,12,12).muli(0.1).addi(0.5).reshape(3,4));
                        x = sd.math().cosineSimilarity(in, z, 1);
                        break;
                    default:
                        throw new RuntimeException();
                }
                if(x.dataType().isFPType()) {
                    //Can't mark argmax as loss, because it's not FP
                    x.markAsLoss();
                }

                if(execFirst){
                    sd.output(Collections.singletonMap("in", arr), Collections.singletonList(x.name()));
                }

                File f = Files.createTempFile(testDir,"some-file","fb").toFile();
                f.delete();
                sd.asFlatFile(f);

                SameDiff restored = SameDiff.fromFlatFile(f);

                List<SDVariable> varsOrig = sd.variables();
                List<SDVariable> varsRestored = restored.variables();
                assertEquals(varsOrig.size(), varsRestored.size());
                for (int j = 0; j < varsOrig.size(); j++) {
                    assertEquals(varsOrig.get(j).name(), varsRestored.get(j).name());
                }

                DifferentialFunction[] fOrig = sd.ops();
                DifferentialFunction[] fRestored = restored.ops();
                assertEquals(fOrig.length, fRestored.length);

                for (int j = 0; j < sd.ops().length; j++) {
                    assertEquals(fOrig[j].getClass(), fRestored[j].getClass());
                }

                assertEquals(sd.getLossVariables(), restored.getLossVariables());


                Map<String,INDArray> m = sd.output(Collections.singletonMap("in", arr), Collections.singletonList(x.name()));
                INDArray outOrig = m.get(x.name());
                Map<String,INDArray> m2 = restored.output(Collections.singletonMap("in", arr), Collections.singletonList(x.name()));
                INDArray outRestored = m2.get(x.name());

                assertEquals(outOrig, outRestored,String.valueOf(i));


                //Check placeholders
                Map<String,SDVariable> vBefore = sd.variableMap();
                Map<String,SDVariable> vAfter = restored.variableMap();
                assertEquals(vBefore.keySet(), vAfter.keySet());
                for(String s : vBefore.keySet()){
                    assertEquals(vBefore.get(s).isPlaceHolder(), vAfter.get(s).isPlaceHolder(),s);
                    assertEquals(vBefore.get(s).isConstant(), vAfter.get(s).isConstant(),s);
                }


                //Check save methods
                for(boolean withUpdaterState : new boolean[]{false, true}) {

                    File f2 = Files.createTempFile(testDir,"some-file-2","fb").toFile();
                    sd.save(f2, withUpdaterState);
                    SameDiff r2 = SameDiff.load(f2, withUpdaterState);
                    assertEquals(varsOrig.size(), r2.variables().size());
                    assertEquals(fOrig.length, r2.ops().length);
                    assertEquals(sd.getLossVariables(), r2.getLossVariables());

                    //Save via stream:
                    File f3 = Files.createTempFile(testDir,"some-file-3","fb").toFile();
                    try(OutputStream os = new BufferedOutputStream(new FileOutputStream(f3))) {
                        sd.save(os, withUpdaterState);
                    }

                    //Load via stream:
                    try(InputStream is = new BufferedInputStream(new FileInputStream(f3))) {
                        SameDiff r3 = SameDiff.load(is, withUpdaterState);
                        assertEquals(varsOrig.size(), r3.variables().size());
                        assertEquals(fOrig.length, r3.ops().length);
                        assertEquals(sd.getLossVariables(), r3.getLossVariables());
                    }
                }
            }
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTrainingSerde(@TempDir Path testDir,Nd4jBackend backend) throws Exception {

        //Ensure 2 things:
        //1. Training config is serialized/deserialized correctly
        //2. Updater state

        for(IUpdater u : new IUpdater[]{
                new AdaDelta(), new AdaGrad(2e-3), new Adam(2e-3), new AdaMax(2e-3),
                new AMSGrad(2e-3), new Nadam(2e-3), new Nesterovs(2e-3), new NoOp(),
                new RmsProp(2e-3), new Sgd(2e-3)}) {

            log.info("Testing: {}", u.getClass().getName());

            SameDiff sd = SameDiff.create();
            SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 4);
            SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, 3);
            SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 4, 3));
            SDVariable b = sd.var("b", Nd4j.rand(DataType.FLOAT, 1, 3));

            SDVariable mmul = in.mmul(w).add(b);
            SDVariable softmax = sd.nn().softmax(mmul, 0);
            //SDVariable loss = sd.loss().logLoss("loss", label, softmax);

            sd.setTrainingConfig(TrainingConfig.builder()
                    .updater(u)
                    .regularization(new L1Regularization(1e-2), new L2Regularization(1e-2), new WeightDecay(1e-2, true))
                    .dataSetFeatureMapping("in")
                    .dataSetLabelMapping("label")
                    .build());

            INDArray inArr = Nd4j.rand(DataType.FLOAT, 3, 4);
            INDArray labelArr = Nd4j.rand(DataType.FLOAT, 3, 3);

            DataSet ds = new DataSet(inArr, labelArr);

            for (int i = 0; i < 10; i++) {
                sd.fit(ds);
            }


            File dir = testDir.toFile();
            File f = new File(dir, "samediff.bin");
            sd.asFlatFile(f);

            SameDiff sd2 = SameDiff.fromFlatFile(f);
            assertNotNull(sd2.getTrainingConfig());
            assertNotNull(sd2.getUpdaterMap());
            assertTrue(sd2.isInitializedTraining());

            assertEquals(sd.getTrainingConfig(), sd2.getTrainingConfig());
            assertEquals(sd.getTrainingConfig().toJson(), sd2.getTrainingConfig().toJson());
            Map<String, GradientUpdater> m1 = sd.getUpdaterMap();
            Map<String, GradientUpdater> m2 = sd2.getUpdaterMap();
            assertEquals(m1.keySet(), m2.keySet());
            for(String s : m1.keySet()){
                GradientUpdater g1 = m1.get(s);
                GradientUpdater g2 = m2.get(s);
                assertEquals(g1.getState(), g2.getState());
                assertEquals(g1.getConfig(), g2.getConfig());
            }


            //Check training post deserialization
            for( int i=0; i<3; i++ ){
                sd.fit(ds);
                sd2.fit(ds);
            }

            for(SDVariable v : sd.variables()){
                if(v.isPlaceHolder() || v.getVariableType() == VariableType.ARRAY)
                    continue;

                SDVariable v2 = sd2.getVariable(v.name());

                INDArray a1 = v.getArr();
                INDArray a2 = v2.getArr();

                assertEquals(a1, a2);
            }
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void pooling3DSerialization(Nd4jBackend backend){
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 28, 28);
        SDVariable o = sd.cnn.maxPooling3d("pool", x, Pooling3DConfig.builder().build());

        ByteBuffer bbSerialized = sd.asFlatBuffers(true);

        SameDiff deserialized;
        try{
            deserialized = SameDiff.fromFlatBuffers(bbSerialized);
        } catch (IOException e){
            throw new RuntimeException("IOException deserializing from FlatBuffers", e);
        }
        assertEquals(
                sd.getVariableOutputOp("pool").getClass(),
                deserialized.getVariableOutputOp("pool").getClass());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void pooling3DSerialization2(Nd4jBackend backend){
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 28, 28);
        SDVariable o = sd.cnn.avgPooling3d("pool", x, Pooling3DConfig.builder().build());

        ByteBuffer bbSerialized = sd.asFlatBuffers(true);

        SameDiff deserialized;
        try{
            deserialized = SameDiff.fromFlatBuffers(bbSerialized);
        } catch (IOException e){
            throw new RuntimeException("IOException deserializing from FlatBuffers", e);
        }
        assertEquals(
                sd.getVariableOutputOp("pool").getClass(),
                deserialized.getVariableOutputOp("pool").getClass());
    }
}
