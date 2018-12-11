package org.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.graph.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.*;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
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
        SDVariable tanh = sd.tanh("out", in);

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
    }

    @Test
    public void testSimple() throws Exception {
        for( int i=0; i<10; i++ ) {
            for(boolean execFirst : new boolean[]{false, true}) {
                log.info("Starting test: i={}, execFirst={}", i, execFirst);
                SameDiff sd = SameDiff.create();
                INDArray arr = Nd4j.linspace(1, 12, 12).reshape(3, 4);
                SDVariable in = sd.var("in", arr.dataType(), arr.shape());
                SDVariable x;
                switch (i) {
                    case 0:
                        //Custom op
                        x = sd.cumsum("out", in, false, false, 1);
                        break;
                    case 1:
                        //Transform
                        x = sd.tanh("out", in);
                        break;
                    case 2:
                    case 3:
                        //Reduction
                        x = sd.mean("x", in, i == 2, 1);
                        break;
                    case 4:
                        //Transform
                        x = sd.square(in);
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
                        x = sd.cosineSimilarity(in, y);
                        break;
                    case 9:
                        //Reduce 3 (along dim)
                        SDVariable z = sd.var("in2", Nd4j.linspace(1,12,12).muli(0.1).addi(0.5).reshape(3,4));
                        x = sd.cosineSimilarity(in, z, 1);
                        break;
                    default:
                        throw new RuntimeException();
                }

                if(execFirst){
                    sd.execAndEndResult();
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


                INDArray outOrig = sd.execAndEndResult();
                INDArray outRestored = restored.execAndEndResult();

                assertEquals(outOrig, outRestored);


                //Check placeholders
                Map<String,SDVariable> vBefore = sd.variableMap();
                Map<String,SDVariable> vAfter = sd.variableMap();
                assertEquals(vBefore.keySet(), vAfter.keySet());
                for(String s : vBefore.keySet()){
                    assertEquals(vBefore.get(s).isPlaceHolder(), vAfter.get(s).isPlaceHolder());
                    assertEquals(vBefore.get(s).isConstant(), vAfter.get(s).isPlaceHolder());
                }
            }
        }
    }
}
