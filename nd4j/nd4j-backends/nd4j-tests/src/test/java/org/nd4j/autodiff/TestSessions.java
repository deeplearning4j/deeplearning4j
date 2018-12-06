package org.nd4j.autodiff;

import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

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
        ph1.setArray(x);
        ph2.setArray(y);

        INDArray out = sd.execAndEndResult();
        System.out.println(out);

    }

}
