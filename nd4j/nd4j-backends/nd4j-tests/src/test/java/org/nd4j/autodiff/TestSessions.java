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

        SameDiff sd = SameDiff.create();

        SDVariable ph1 = sd.placeHolder("x", DataType.FLOAT, 3,4);
        SDVariable ph2 = sd.placeHolder("y", DataType.FLOAT, 1,4);

        SDVariable outExp = ph1.add("out", ph2);

        //NOTE: normally sessions are internal and completely hidden from users

        InferenceSession is = new InferenceSession(sd);

        INDArray x = Nd4j.linspace(1, 12, 12).castTo(DataType.FLOAT).reshape(3,4);
        INDArray y = Nd4j.linspace(0.1, 0.4, 4).castTo(DataType.FLOAT).reshape(1,4);

        //TODO REMOVE THIS
        ph1.setArray(x);
        ph2.setArray(y);
        sd.execAndEndResult();

        Map<String,INDArray> m = new HashMap<>();
        m.put("x", x);
        m.put("y", y);

        Map<String,INDArray> outMap = is.output(Collections.singletonList("out"), m, null);

        assertEquals(1, outMap.size());
        assertEquals(outExp, outMap.get("out"));

    }

}
