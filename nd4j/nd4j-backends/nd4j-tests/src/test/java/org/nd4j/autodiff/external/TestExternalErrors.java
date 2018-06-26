package org.nd4j.autodiff.external;

import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.temp.ExternalErrorsFunction;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

public class TestExternalErrors {

    @Test
    public void testSimple(){
        INDArray externalGrad = Nd4j.linspace(1,12,12).reshape(3,4);

        SameDiff sd = SameDiff.create();
        SDVariable var = sd.var("var", externalGrad);
        SDVariable out = var.mul("out", 0.5);

        Map<String,INDArray> gradMap = new HashMap<>();
        gradMap.put("out", externalGrad);
        ExternalErrorsFunction fn = sd.f().externalErrors(out);
        //new ExternalErrorsFunction(sd, Collections.singletonList(out), gradMap);

        fn.updateVariable("out", externalGrad);
        sd.execAndEndResult();
        sd.execBackwards();

        INDArray gradOut = out.getGradient().getArr();
        INDArray gradVar = var.getGradient().getArr();

        assertEquals(externalGrad, gradOut);
        assertEquals(externalGrad.mul(0.5), gradVar);

        //Now, update and execute again:
        externalGrad = Nd4j.linspace(1,12,12).reshape(3,4).muli(10);
        fn.updateVariable("out", externalGrad);

        sd.execBackwards();

        gradOut = out.getGradient().getArr();
        gradVar = var.getGradient().getArr();

        assertEquals(externalGrad, gradOut);
        assertEquals(externalGrad.mul(0.5), gradVar);
    }

}
