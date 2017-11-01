package org.nd4j.autodiff.gradcheck;

import org.junit.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;

import static org.junit.Assert.assertTrue;

/**
 * Created by agibsonccc on 5/7/17.
 */
public class GradCheckUtilTest {


    @Test
    public void testGradCheck() {
        Nd4j.create(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
        SameDiff sameDiff = SameDiff.create();
        INDArray scalar = Nd4j.ones(1);
        SDVariable var = sameDiff.var("x",scalar);
        SDVariable sigmoid = sameDiff.sigmoid(var);
        assertTrue(GradCheckUtil.checkGradients(sigmoid,var,1e-6,1e-6,false,
                Collections.singletonMap("x",scalar)));


    }

}
