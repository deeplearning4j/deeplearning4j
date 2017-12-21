package org.deeplearning4j.samediff;

import org.junit.Test;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

public class SameDiffGradTest {

    @Test
    public void test1(){
        Nd4j.getRandom().setSeed(12345);
        INDArray inArr = Nd4j.rand(1,4);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable s = sd.tanh("s", in);

        INDArray out = sd.execAndEndResult();
        INDArray outEx = Transforms.tanh(inArr, true);

        assertEquals(outEx, out);
        System.out.println(out);

        System.out.println("------------------");

        List<SDVariable> vs = sd.variables();
        for(SDVariable sdv : vs){
//            if(sdv.getVarName().equals("in")){
//                System.out.println(sdv.getVarName() + "\n" + sdv.getArr());
//            } else {
//                System.out.println(sdv.getVarName() + " - inputs: " + Arrays.toString(sd.getInputsForFunction(sdv)) + "\n" + sdv.getArr());
//            }
            System.out.println(sdv.getVarName() + "\n" + sdv.getArr());
        }

        System.out.println("------------------");

        Pair<Map<SDVariable,DifferentialFunction>,List<DifferentialFunction>> p = sd.execBackwards();

        System.out.println("------------------");

        System.out.println("GRAD variables:");
        SameDiff grad = sd.getFunction("grad");
        for(SDVariable sdv : grad.variables()){
            System.out.println(sdv.getVarName() + " - inputs: " + Arrays.toString(sd.getInputsForFunction(sdv)) + "\n" + sdv.getArr());
        }

        System.out.println("------------------");


    }

}
