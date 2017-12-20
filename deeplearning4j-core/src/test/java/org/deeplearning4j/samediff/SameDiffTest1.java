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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;

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
        Nd4j.getRandom().setSeed(12345);

        SameDiff sd = SameDiff.create();

        INDArray iInput = Nd4j.rand(3,4);
        INDArray iWeights = Nd4j.rand(4,5);
        INDArray iBias = Nd4j.zeros(1, 5);  //Nd4j.rand(1,5);

        SDVariable input = sd.var("input", iInput);
        SDVariable weights = sd.var("weights", iWeights);
        SDVariable bias = sd.var("bias", iBias);

        SDVariable mmul = sd.mmul("mmul", input, weights);
        SDVariable z = mmul.add("z", bias);
        SDVariable out = sd.sigmoid("out", z);


//        INDArray outArr = out.eval();
        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> m = sd.exec();

        for(Map.Entry<SDVariable, DifferentialFunction> e : m.getFirst().entrySet()){
            System.out.println(e.getKey().getVarName());
            System.out.println(e.getKey().getArr());
        }

        System.out.println("------------\nAll variable values");

        List<SDVariable> variables = sd.variables();
        for(SDVariable s : variables){
            System.out.println(s.getVarName());
            System.out.println(s.getArr());
        }

        System.out.println("------------");

        INDArray exp = iInput.mmul(iWeights).addiRowVector(iBias);

        System.out.println("Input:");
        System.out.println(iInput);
        System.out.println("Weights:");
        System.out.println(iWeights);
        System.out.println("Bias:");
        System.out.println(iBias);

        System.out.println("------------");

        System.out.println("Expected:");
        System.out.println(exp);
        System.out.println("Actual:");
//        System.out.println(outArr);
//        System.out.println(Arrays.toString(outArr.dup().data().asFloat()));
    }


    @Test
    public void test5() {
        Nd4j.getRandom().setSeed(12345);

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

        System.out.println("------------\nAll variable values");

        sd.exec();

        List<SDVariable> variables = sd.variables();
        for(SDVariable s : variables){
            System.out.println(s.getVarName());
            System.out.println(s.getArr());
            System.out.println("Data buffer: " + Arrays.toString(s.getArr().data().asFloat()));
        }

        System.out.println("------------");

        List<String> varNames = variables.stream().map(SDVariable::getVarName).collect(Collectors.toList());
        System.out.println("VarNames: " + varNames);    //"z" and "out" appear twice

        INDArray expMmul = iInput.mmul(iWeights);
        INDArray expZ = expMmul.addRowVector(iBias);
        INDArray expOut = Transforms.sigmoid(expZ, true);

        assertEquals(expMmul, mmul.getArr());
        assertEquals(expZ, z.getArr());
        assertEquals(expOut, out.getArr());
    }
}
