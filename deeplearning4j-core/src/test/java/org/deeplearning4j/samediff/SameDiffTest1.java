package org.deeplearning4j.samediff;

import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class SameDiffTest1 {

    @Test
    public void test1() {

        SameDiff sd = SameDiff.create();

        SDVariable input = sd.var("input", new int[]{3,4});
        SDVariable weights = sd.var("weights", new int[]{4,5});
        SDVariable bias = sd.var("bias", new int[]{1,5});

        SDVariable mmul = sd.mmul("mmul", input, weights);
        SDVariable z = mmul.add("z", bias);
        SDVariable out = sd.sigmoid("out", z);

//        SDGraph g = sd.graph();
//        System.out.println(g);

        System.out.println(out);


        INDArray iInput = Nd4j.rand(3,4);
        INDArray iWeights = Nd4j.rand(4,5);
        INDArray iBias = Nd4j.rand(1,5);

        INDArray iZ = iInput.mmul(iWeights).addiRowVector(iBias);
        INDArray iOut = Transforms.sigmoid(iZ, true);

        Map<String,INDArray> values = new HashMap<>();
        values.put("input", iInput);
        values.put("weights", iWeights);
        values.put("bias", iBias);

        INDArray[] outAct = sd.eval(values);

        System.out.println();
    }


    @Test
    public void test2() {

        SameDiff sd = SameDiff.create();

        SDVariable input = sd.var("input", new int[]{3,4});
        SDVariable weights = sd.var("weights", new int[]{4,5});
        SDVariable bias = sd.var("bias", new int[]{1,5});

        SDVariable mmul = sd.mmul("mmul", input, weights);
        SDVariable z = mmul.add("z", bias);
        SDVariable out = sd.sigmoid("out", z);

//        SDGraph g = sd.graph();
//        System.out.println(g);

        System.out.println(out);


        INDArray iInput = Nd4j.rand(3,4);
        INDArray iWeights = Nd4j.rand(4,5);
        INDArray iBias = Nd4j.rand(1,5);

        INDArray iZ = iInput.mmul(iWeights).addiRowVector(iBias);
        INDArray iOut = Transforms.sigmoid(iZ, true);

        Map<String,INDArray> values = new HashMap<>();
        values.put("input", iInput);
        values.put("weights", iWeights);
        values.put("bias", iBias);

        INDArray[] outAct = sd.eval(values);

        System.out.println();
    }

    @Test
    public void test3() {

        SameDiff sd = SameDiff.create();

        INDArray iInput = Nd4j.rand(3,4);
        INDArray iWeights = Nd4j.rand(4,5);
        INDArray iBias = Nd4j.rand(1,5);

        SDVariable input = sd.var("input", iInput);
        SDVariable weights = sd.var("weights", iWeights);
        SDVariable bias = sd.var("bias", iBias);

        SDVariable mmul = sd.mmul("mmul", input, weights);
        SDVariable z = mmul.add("z", bias);
        SDVariable out = sd.sigmoid("out", z);


        INDArray outAct = sd.execAndEndResult();



        INDArray iZ = iInput.mmul(iWeights).addiRowVector(iBias);
        INDArray iOut = Transforms.sigmoid(iZ, true);

        Map<String,INDArray> values = new HashMap<>();
        values.put("input", iInput);
        values.put("weights", iWeights);
        values.put("bias", iBias);

        System.out.println();
    }


    @Test
    public void test4() {

        SameDiff sd = SameDiff.create();

        INDArray iInput = Nd4j.rand(3,4);
        INDArray iWeights = Nd4j.rand(4,5);
        INDArray iBias = Nd4j.rand(1,5);

        SDVariable input = sd.var("input", iInput);
        SDVariable weights = sd.var("weights", iWeights);
        SDVariable bias = sd.var("bias", iBias);

        SDVariable mmul = sd.mmul("mmul", input, weights);
        SDVariable z = mmul.add("z", bias);
        SDVariable out = sd.sigmoid("out", z);


        INDArray outArr = out.eval();

        INDArray exp = iInput.mmul(iWeights).addiRowVector(iBias);

        System.out.println(outArr);
        System.out.println(Arrays.toString(outArr.dup().data().asFloat()));
        System.out.println("Expected:");
        System.out.println(exp);
    }
}
