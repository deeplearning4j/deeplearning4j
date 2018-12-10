package org.nd4j.autodiff;

import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SameDiffConditional;
import org.nd4j.autodiff.samediff.SameDiffFunctionDefinition;
import org.nd4j.autodiff.samediff.internal.AbstractSession;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.controlflow.While;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class TestSessions {

    @Test
    public void testInferenceSessionBasic(){
        //So far: trivial test to check execution order

        SameDiff sd = SameDiff.create();

        SDVariable ph1 = sd.placeHolder("x", DataType.FLOAT, 3,4);
        SDVariable ph2 = sd.placeHolder("y", DataType.FLOAT, 1,4);

        SDVariable out = ph1.add("out", ph2);

        //NOTE: normally sessions are internal and completely hidden from users

        InferenceSession is = new InferenceSession(sd);

        INDArray x = Nd4j.linspace(1, 12, 12).castTo(DataType.FLOAT).reshape(3,4);
        INDArray y = Nd4j.linspace(0.1, 0.4, 4).castTo(DataType.FLOAT).reshape(1,4);

        INDArray outExp = x.addRowVector(y);

        Map<String,INDArray> m = new HashMap<>();
        m.put("x", x);
        m.put("y", y);

        Map<String,INDArray> outMap = is.output(Collections.singletonList("out"), m, null);

        assertEquals(1, outMap.size());
        assertEquals(outExp, outMap.get("out"));
    }


    @Test
    public void testInferenceSessionBasic2(){
        //So far: trivial test to check execution order

        SameDiff sd = SameDiff.create();
        SDVariable ph1 = sd.placeHolder("x", DataType.FLOAT, 3,3);
        SDVariable ph2 = sd.placeHolder("y", DataType.FLOAT, 3,3);

        SDVariable a = ph1.add("a", ph2);
        SDVariable b = ph1.mmul("b", ph2);
        SDVariable c = ph1.sub("c", ph2);
        SDVariable d = a.add("d", b);

        //To get array d - need to execute: a, b, d - NOT the sub op (c)

        //NOTE: normally sessions are internal and completely hidden from users

        InferenceSession is = new InferenceSession(sd);
        INDArray x = Nd4j.linspace(1, 9, 9).castTo(DataType.FLOAT).reshape(3,3);
        INDArray y = Nd4j.linspace(0.0, 0.9, 9).castTo(DataType.FLOAT).reshape(3,3);

        INDArray aExp = x.add(y);
        INDArray bExp = x.mmul(y);
        INDArray dExp = aExp.add(bExp);

        Map<String,INDArray> m = new HashMap<>();
        m.put("x", x);
        m.put("y", y);

        System.out.println("----------------------------------");
        Map<String,INDArray> outMap = is.output(Collections.singletonList("d"), m, null);

        assertEquals(1, outMap.size());
        assertEquals(dExp, outMap.get("d"));
    }

    @Test
    public void testMergeSimple(){
        //This isn't really a sensible graph, as merge op behaviour is undefined when multiple inputs are available...

        SameDiff sd = SameDiff.create();
        SDVariable ph1 = sd.placeHolder("x", DataType.FLOAT, 3,3);
        SDVariable ph2 = sd.placeHolder("y", DataType.FLOAT, 3,3);

        SDVariable merge = sd.f().merge(ph1, ph2);

        SDVariable outVar = sd.identity(merge);

        INDArray x = Nd4j.linspace(1, 9, 9).castTo(DataType.FLOAT).reshape(3,3);
        INDArray y = Nd4j.linspace(0.0, 0.9, 9).castTo(DataType.FLOAT).reshape(3,3);
//        ph1.setArray(x);
//        ph2.setArray(y);
//        INDArray out = sd.execAndEndResult();
//        System.out.println(out);


        Map<String,INDArray> m = new HashMap<>();
        m.put("x", x);
        m.put("y", y);

        System.out.println("----------------------------------");
        InferenceSession is = new InferenceSession(sd);
//        String outName = merge.getVarName();
        String outName = outVar.getVarName();
        Map<String,INDArray> outMap = is.output(Collections.singletonList(outName), m, null);

        assertEquals(1, outMap.size());
        INDArray out = outMap.get(outName);
        assertTrue(x.equals(out) || y.equals(out));
    }


    @Test
    public void testSwitchSimple(){

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3,3);
        SDVariable b = sd.placeHolder("b", DataType.BOOL);

        SDVariable[] switchOut = sd.f().switchOp(x, b); //Order: false then true
        SDVariable falsePlusOne = switchOut[0].add("addFalseBranch", 1);
        SDVariable truePlusTen = switchOut[1].add("addTrueBranch", 10.0);

        SDVariable merge = sd.f().merge(falsePlusOne, truePlusTen);

        INDArray xArr = Nd4j.create(DataType.FLOAT, 3,3);
        INDArray bArr = Nd4j.scalar(true);

        INDArray expTrue = xArr.add(10.0);
        INDArray expFalse = xArr.add(1.0);

        Map<String,INDArray> m = new HashMap<>();
        m.put("x", xArr);
        m.put("b", bArr);

        InferenceSession is = new InferenceSession(sd);
        String n = merge.getVarName();

        System.out.println("----------------------------------");
        Map<String,INDArray> outMap = is.output(Collections.singletonList(n), m, null);
        assertEquals(1, outMap.size());
        assertEquals(expTrue, outMap.get(n));


        System.out.println("----------------------------------");
        //Check false case:
        bArr.assign(0);
        is = new InferenceSession(sd);
        outMap = is.output(Collections.singletonList(n), m, null);
        assertEquals(1, outMap.size());
        assertEquals(expFalse, outMap.get(n));
    }

    @Test(timeout = 50000L)
    public void testSwitchWhile() throws Exception{

        /*
        Test case:
        i=0, j=numIter
        while(i<j){
            i++
        }
        return (i,j)

        Therefore, expected output for 2 nodes is (numIter, numIter)
         */

        for( int numIter : new int[]{1,3}) {
            File f = new ClassPathResource("tf_graphs/examples/while1/iter_" + numIter + "/frozen_model.pb").getFile();
            SameDiff sd = TFGraphMapper.getInstance().importGraph(f);

            System.out.println(sd.summary());

            System.out.println("----------------------------------");
            //This particular test/graph doesn't use placeholders
            InferenceSession is = new InferenceSession(sd);
            String n = "while/Exit";
            String n2 = "while/Exit_1";

            Map<String, INDArray> m = is.output(Arrays.asList(n, n2), Collections.emptyMap(), null);
            assertEquals(2, m.size());

            INDArray exp = Nd4j.scalar((double)numIter);

            assertEquals(exp, m.get(n));
            assertEquals(exp, m.get(n2));

            Map<String,AbstractSession.FrameIter> frameParents = is.getFrameParents();
            Map<AbstractSession.VarId,INDArray> outputs = is.getNodeOutputs();
            //Some sanity checks on the internal state:
            //Check 1: "while/Less" should be executed numIter+1 times... i.e., numIter times through the loop, plus once to exit
            for( int i=0; i<numIter+1; i++ ){
                AbstractSession.VarId expVarId = new AbstractSession.VarId("while/Less","while/while_context", i);
                INDArray expLessVal = Nd4j.scalar(i != numIter);
                assertTrue(outputs.containsKey(expVarId));
                //assertEquals(expLessVal, outputs.get(expVarId));      //TODO can't test this due to in-place modifications of output arrays
            }
            AbstractSession.VarId expVarId = new AbstractSession.VarId("while/Less","while/while_context", numIter+1);
//            assertFalse(outputs.containsKey(expVarId));               //TODO can't test this due to in-place modifications of output arrays

            //Check 2: Add should be executed numIter times...
            for( int i=0; i<numIter; i++ ){
                expVarId = new AbstractSession.VarId("while/add","while/while_context", i);
                INDArray expAddVal = Nd4j.scalar(i);
                assertTrue(outputs.containsKey(expVarId));
                //assertEquals(expAddVal, outputs.get(expVarId));      //TODO can't test this due to in-place modifications of output arrays
            }
        }

    }
}
