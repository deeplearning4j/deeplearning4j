package org.nd4j.autodiff.samediff;

import org.apache.commons.io.IOUtils;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.graph.*;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class FlatBufferSerdeTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testBasic() throws Exception {

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.linspace(1,12,12).reshape(3,4));
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

    }

    @Test
    public void testSimple() throws Exception {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.linspace(1,12,12).reshape(3,4));
//        SDVariable tanh = sd.tanh("out", in);
        SDVariable x = sd.mean("x", in, true, 1);

        File f = testDir.newFile();
        f.delete();

        sd.asFlatFile(f);

        SameDiff loaded = SameDiff.fromFlatFile(f);



    }

}
