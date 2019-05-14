/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.graph.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

@Slf4j
public class FlatBufferSerdeTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testBasic() throws Exception {
        SameDiff sd = SameDiff.create();
        INDArray arr = Nd4j.linspace(1,12,12).reshape(3,4);
        SDVariable in = sd.placeHolder("in", arr.dataType(), arr.shape() );
        SDVariable tanh = sd.nn().tanh("out", in);
        tanh.markAsLoss();

        ByteBuffer bb = sd.asFlatBuffers();

        File f = testDir.newFile();
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
        //assertEquals(sd.getLossVariables(), fg)
    }

    @Test
    public void testSimple() throws Exception {
        for( int i=0; i<10; i++ ) {
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
                        x = sd.nn().tanh("out", in);
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
                    sd.exec(Collections.singletonMap("in", arr), Collections.singletonList(x.getVarName()));
                }

                File f = testDir.newFile();
                f.delete();
                sd.asFlatFile(f);

                SameDiff restored = SameDiff.fromFlatFile(f);

                List<SDVariable> varsOrig = sd.variables();
                List<SDVariable> varsRestored = restored.variables();
                assertEquals(varsOrig.size(), varsRestored.size());
                for (int j = 0; j < varsOrig.size(); j++) {
                    assertEquals(varsOrig.get(j).getVarName(), varsRestored.get(j).getVarName());
                }

                DifferentialFunction[] fOrig = sd.functions();
                DifferentialFunction[] fRestored = restored.functions();
                assertEquals(fOrig.length, fRestored.length);

                for (int j = 0; j < sd.functions().length; j++) {
                    assertEquals(fOrig[j].getClass(), fRestored[j].getClass());
                }

                assertEquals(sd.getLossVariables(), restored.getLossVariables());


                Map<String,INDArray> m = sd.exec(Collections.singletonMap("in", arr), Collections.singletonList(x.getVarName()));
                INDArray outOrig = m.get(x.getVarName());
                Map<String,INDArray> m2 = restored.exec(Collections.singletonMap("in", arr), Collections.singletonList(x.getVarName()));
                INDArray outRestored = m2.get(x.getVarName());

                assertEquals(String.valueOf(i), outOrig, outRestored);


                //Check placeholders
                Map<String,SDVariable> vBefore = sd.variableMap();
                Map<String,SDVariable> vAfter = sd.variableMap();
                assertEquals(vBefore.keySet(), vAfter.keySet());
                for(String s : vBefore.keySet()){
                    assertEquals(s, vBefore.get(s).isPlaceHolder(), vAfter.get(s).isPlaceHolder());
                    assertEquals(s, vBefore.get(s).isConstant(), vAfter.get(s).isConstant());
                }
            }
        }
    }
}
