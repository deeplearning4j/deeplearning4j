package org.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.nd4j.autodiff.OpValidationSuite;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpTestCase;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.accum.CumProd;
import org.nd4j.linalg.api.ops.impl.accum.CumSum;
import org.nd4j.linalg.api.ops.impl.accum.Mmul;
import org.nd4j.linalg.api.ops.impl.shape.DiagPart;
import org.nd4j.linalg.api.ops.impl.shape.OneHot;
import org.nd4j.linalg.api.ops.impl.shape.ZerosLike;
import org.nd4j.linalg.api.ops.impl.transforms.Fill;
import org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByNorm;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.primitives.Triple;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.*;

import static org.junit.Assert.*;
import static org.junit.Assume.assumeNotNull;

@Slf4j
public class MiscOpValidation extends BaseOpValidation {

    public MiscOpValidation(Nd4jBackend backend) {
        super(backend);
    }



    @Test
    public void testGradientAutoBroadcast1() {
        OpValidationSuite.ignoreFailing();

        Nd4j.getRandom().setSeed(12345);

        List<String> failed = new ArrayList<>();

        for (int dim_sz1 : new int[]{0, 1, 2}) {

            int[] in2Shape = {3, 4, 5};
            in2Shape[dim_sz1] = 1;

            for (int i = 0; i < 8; i++) {

                SameDiff sd = SameDiff.create();

                SDVariable in3 = sd.var("in3", Nd4j.rand(new int[]{3, 4, 5}));
                SDVariable in2 = sd.var("in2", in2Shape);

                SDVariable bcOp;
                String name;
                switch (i) {
                    case 0:
                        bcOp = in3.add(in2);
                        name = "add";
                        break;
                    case 1:
                        bcOp = in3.sub(in2);
                        name = "sub";
                        break;
                    case 2:
                        bcOp = in3.mul(in2);
                        name = "mul";
                        break;
                    case 3:
                        bcOp = in3.div(in2);
                        name = "div";
                        break;
                    case 4:
                        bcOp = in3.rsub(in2);
                        name = "rsub";
                        break;
                    case 5:
                        bcOp = in3.rdiv(in2);
                        name = "rdiv";
                        break;
                    case 6:
                        bcOp = sd.f().floorDiv(in3, in2);
                        name = "floordiv";
                        break;
                    case 7:
                        bcOp = sd.f().floorMod(in3, in2);
                        name = "floormod";
                        break;
                    default:
                        throw new RuntimeException();
                }

                SDVariable outVar = sd.sum(bcOp);

                String msg = "(test " + i + ": " + name + ", dimension=" + dim_sz1 + ")";
                log.info("*** Starting test: " + msg);

                INDArray in3Arr = Nd4j.randn(new int[]{3, 4, 5}).muli(100);
                INDArray in2Arr = Nd4j.randn(in2Shape).muli(100);

                sd.associateArrayWithVariable(in3Arr, in3);
                sd.associateArrayWithVariable(in2Arr, in2);

                TestCase tc = new TestCase(sd);

                String error = OpValidation.validate(tc);
                if(error != null){
                    failed.add(name);
                }
            }
        }

        assertEquals("Failed: " + failed, 0, failed.size());
    }

    @Test
    public void testGradientAutoBroadcast2() {
        OpValidationSuite.ignoreFailing();

        Nd4j.getRandom().setSeed(12345);

        List<String> failed = new ArrayList<>();

        for (int[] dim_sz1s : new int[][]{{0, 1}, {0, 2}, {1, 2}, {0, 1, 2}}) {

            int[] otherShape = {3, 4, 5};
            otherShape[dim_sz1s[0]] = 1;
            otherShape[dim_sz1s[1]] = 1;
            if (dim_sz1s.length == 3) {
                otherShape[dim_sz1s[2]] = 1;
            }

            for (int i = 0; i < 8; i++) {

                SameDiff sd = SameDiff.create();

                SDVariable in3 = sd.var("in3", new int[]{3, 4, 5});
                SDVariable in2 = sd.var("inToBc", otherShape);

                String name;
                SDVariable bcOp;
                switch (i) {
                    case 0:
                        bcOp = in3.add(in2);
                        name = "add";
                        break;
                    case 1:
                        bcOp = in3.sub(in2);
                        name = "sub";
                        break;
                    case 2:
                        bcOp = in3.mul(in2);
                        name = "mul";
                        break;
                    case 3:
                        bcOp = in3.div(in2);
                        name = "div";
                        break;
                    case 4:
                        bcOp = in3.rsub(in2);
                        name = "rsub";
                        break;
                    case 5:
                        bcOp = in3.rdiv(in2);
                        name = "rdiv";
                        break;
                    case 6:
                        bcOp = sd.f().floorDiv(in3, in2);
                        name = "floordiv";
                        break;
                    case 7:
                        bcOp = sd.f().floorMod(in3, in2);
                        name = "floormod";
                        break;
                    default:
                        throw new RuntimeException();
                }

                SDVariable outVar = sd.sum(bcOp);

                String msg = "(test " + i + ": " + name + ", dimensions=" + Arrays.toString(dim_sz1s) + ")";
                log.info("*** Starting test: " + msg);

                INDArray in3Arr = Nd4j.randn(new int[]{3, 4, 5}).muli(100);
                INDArray in2Arr = Nd4j.randn(otherShape).muli(100);

                sd.associateArrayWithVariable(in3Arr, in3);
                sd.associateArrayWithVariable(in2Arr, in2);

                TestCase tc = new TestCase(sd);
                String error = OpValidation.validate(tc);
                if(error != null){
                    failed.add(name);
                }
            }
        }

        assertEquals("Failed: " + failed, 0, failed.size());
    }

    @Test
    public void testGradientAutoBroadcast3() {
        OpValidationSuite.ignoreFailing();
        //These tests: output size > input sizes

        Nd4j.getRandom().setSeed(12345);

        List<String> failed = new ArrayList<>();

        //Test cases: in1Shape, in2Shape, shapeOf(op(in1,in2))
        List<Triple<long[], long[], long[]>> testCases = new ArrayList<>();
        testCases.add(new Triple<>(new long[]{3, 1}, new long[]{1, 4}, new long[]{3, 4}));
        testCases.add(new Triple<>(new long[]{3, 1}, new long[]{3, 4}, new long[]{3, 4}));
        testCases.add(new Triple<>(new long[]{3, 4}, new long[]{1, 4}, new long[]{3, 4}));
        testCases.add(new Triple<>(new long[]{3, 4, 1}, new long[]{1, 1, 5}, new long[]{3, 4, 5}));
        testCases.add(new Triple<>(new long[]{3, 4, 1}, new long[]{3, 1, 5}, new long[]{3, 4, 5}));
        testCases.add(new Triple<>(new long[]{3, 1, 5}, new long[]{1, 4, 1}, new long[]{3, 4, 5}));
        testCases.add(new Triple<>(new long[]{3, 1, 5}, new long[]{1, 4, 5}, new long[]{3, 4, 5}));
        testCases.add(new Triple<>(new long[]{3, 1, 5}, new long[]{3, 4, 5}, new long[]{3, 4, 5}));
        testCases.add(new Triple<>(new long[]{3, 1, 1, 1}, new long[]{1, 4, 5, 6}, new long[]{3, 4, 5, 6}));
        testCases.add(new Triple<>(new long[]{1, 1, 1, 6}, new long[]{3, 4, 5, 6}, new long[]{3, 4, 5, 6}));
        testCases.add(new Triple<>(new long[]{1, 4, 5, 1}, new long[]{3, 1, 1, 6}, new long[]{3, 4, 5, 6}));
        testCases.add(new Triple<>(new long[]{1, 6}, new long[]{3, 4, 5, 1}, new long[]{3, 4, 5, 6}));

        for (val p : testCases) {

            for (int i = 0; i < 8; i++) {

                SameDiff sd = SameDiff.create();

                SDVariable in3 = sd.var("in1", p.getFirst());
                SDVariable in2 = sd.var("in2", p.getSecond());

                String name;
                SDVariable bcOp;
                switch (i) {
                    case 0:
                        bcOp = in3.add(in2);
                        name = "add";
                        break;
                    case 1:
                        bcOp = in3.sub(in2);
                        name = "sub";
                        break;
                    case 2:
                        bcOp = in3.mul(in2);
                        name = "mul";
                        break;
                    case 3:
                        bcOp = in3.div(in2);
                        name = "div";
                        break;
                    case 4:
                        bcOp = in3.rsub(in2);
                        name = "rsub";
                        break;
                    case 5:
                        bcOp = in3.rdiv(in2);
                        name = "rdiv";
                        break;
                    case 6:
                        bcOp = sd.f().floorDiv(in3, in2);
                        name = "floordiv";
                        break;
                    case 7:
                        bcOp = sd.f().floorMod(in3, in2);
                        name = "floormod";
                        break;
                    default:
                        throw new RuntimeException();
                }

                SDVariable outVar = sd.sum(bcOp);

                String msg = "(test " + i + ": " + name + ", array 1 size =" + Arrays.toString(p.getFirst())
                        + ", array 2 size = " + Arrays.toString(p.getSecond()) + ")";
                log.info("*** Starting test: " + msg);

                INDArray in3Arr = Nd4j.randn(p.getFirst()).muli(100);
                INDArray in2Arr = Nd4j.randn(p.getSecond()).muli(100);

                sd.associateArrayWithVariable(in3Arr, in3);
                sd.associateArrayWithVariable(in2Arr, in2);

                TestCase tc = new TestCase(sd);
                String error = OpValidation.validate(tc);
                if(error != null){
                    failed.add(name);
                }
            }
        }

        assertEquals("Failed: " + failed, 0, failed.size());
    }



    @Test
    public void testScatterOpGradients() {
        List<String> failed = new ArrayList<>();

        for (int i = 0; i < 5; i++) {
            Nd4j.getRandom().setSeed(12345);

            SameDiff sd = SameDiff.create();

            SDVariable in = sd.var("in", new int[]{20, 10});
            SDVariable indices = sd.var("indices", new long[]{5});
            SDVariable updates = sd.var("updates", new int[]{5, 10});


            in.setArray(Nd4j.rand(20, 10));
            indices.setArray(Nd4j.create(new double[]{3, 4, 5, 10, 18}));
            updates.setArray(Nd4j.rand(5, 10).muli(2).subi(1));

            SDVariable scatter;
            String name;
            switch (i) {
                case 0:
                    scatter = sd.scatterAdd("s", in, indices, updates);
                    name = "scatterAdd";
                    break;
                case 1:
                    scatter = sd.scatterSub("s", in, indices, updates);
                    name = "scatterSub";
                    break;
                case 2:
                    scatter = sd.scatterMul("s", in, indices, updates);
                    name = "scatterMul";
                    break;
                case 3:
                    scatter = sd.scatterDiv("s", in, indices, updates);
                    name = "scatterDiv";
                    break;
                case 4:
                    scatter = sd.scatterUpdate("s", in, indices, updates);
                    name = "scatterUpdate";
                    break;
                default:
                    throw new RuntimeException();
            }

            INDArray exp = in.getArr().dup();
            int[] indicesInt = indices.getArr().dup().data().asInt();
            for( int j=0; j<indicesInt.length; j++ ){
                INDArray updateRow = updates.getArr().getRow(j);
                INDArray destinationRow = exp.getRow(indicesInt[j]);
                switch (i){
                    case 0:
                        destinationRow.addi(updateRow);
                        break;
                    case 1:
                        destinationRow.subi(updateRow);
                        break;
                    case 2:
                        destinationRow.muli(updateRow);
                        break;
                    case 3:
                        destinationRow.divi(updateRow);
                        break;
                    case 4:
                        destinationRow.assign(updateRow);
                        break;
                }
            }

            SDVariable loss = sd.sum(scatter);  //.standardDeviation(scatter, true);  //.sum(scatter);  //TODO stdev might be better here as gradients are non-symmetrical...
            sd.execAndEndResult();

            TestCase tc = new TestCase(sd)
                    .expected(scatter, exp)
                    .gradCheckSkipVariables(indices.getVarName());

            String error = OpValidation.validate(tc);
            if(error != null){
                failed.add(name);
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testGatherGradient() {
        OpValidationSuite.ignoreFailing();

        Nd4j.getRandom().setSeed(12345);

        List<String> failed = new ArrayList<>();

        for (int rank = 2; rank <= 3; rank++) {
            for (int dim = 0; dim < rank; dim++) {
                SameDiff sd = SameDiff.create();

                int[] inShape;
                if (rank == 2) {
                    inShape = new int[]{10, 10};
                } else {
                    inShape = new int[]{10, 10, 10};
                }

                SDVariable in = sd.var("in", Nd4j.rand(inShape));
                SDVariable indices = sd.var("indices", Nd4j.create(new double[]{0, 3, 7}));

                INDArray gatherExp = null;
                if(rank == 2){
                    gatherExp = Nd4j.pullRows(in.getArr(), dim, new int[]{0,3,7});
                }

                SDVariable gather = sd.gather(in, indices, dim);
                sd.execAndEndResult();  //TODO REMOVE THIS

                SDVariable loss = sd.standardDeviation("loss", gather, true, Integer.MAX_VALUE);

                String msg = "rank=" + rank + ", dim=" + dim;

                TestCase tc = new TestCase(sd).testName(msg);
                tc.testName(msg);

                if (gatherExp != null) {
                    tc.expected(gather, gatherExp);
                }

                String error = OpValidation.validate(tc);
                if(error != null){
                    failed.add(msg);
                }
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }


    @Test
    public void testTensorGradTensorMmul() {
        OpValidationSuite.ignoreFailing();

        Nd4j.getRandom().setSeed(12345);
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.rand(new long[]{2, 2, 2});
        INDArray arr2 = Nd4j.rand(new long[]{2, 2, 2});
        SDVariable x = sameDiff.var("x", arr);
        SDVariable y = sameDiff.var("y", arr2);
        SDVariable result = sameDiff.tensorMmul(x, y, new int[][]{{0}, {1}});
        assertArrayEquals(ArrayUtil.getTensorMmulShape(new long[]{2, 2, 2}, new long[]{2, 2, 2}, new int[][]{{0}, {1}}), result.getShape());
        assertEquals(32, sameDiff.numElements());

        SDVariable loss = sameDiff.standardDeviation(result, true);

        String err = OpValidation.validate(new TestCase(sameDiff));
    }

    @Test
    public void testMulGradient() {
        INDArray arr1 = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray arr2 = Nd4j.linspace(1, 4, 4).reshape(2, 2);

        INDArray gradAssertion = Nd4j.ones(arr1.shape());
        INDArray scalar = Nd4j.scalar(1.0);
        INDArray aGradAssertion = Nd4j.create(new double[][]{
                {1, 4},
                {9, 16}
        });

        INDArray cGradAssertion = Nd4j.create(new double[][]{
                {1, 2},
                {3, 4}
        });

        INDArray wGradAssertion = Nd4j.create(new double[][]{
                {2, 8},
                {18, 32}
        });

        INDArray dGradAssertion = Nd4j.ones(2, 2);

        SameDiff sameDiff = SameDiff.create();

        SDVariable sdVariable = sameDiff.var("a", arr1);
        SDVariable sdVariable1 = sameDiff.var("w", arr2);
        SDVariable varMulPre = sdVariable.mul("c", sdVariable1);
        SDVariable varMul = varMulPre.mul("d", sdVariable1);
        SDVariable sum = sameDiff.sum("ret", varMul, Integer.MAX_VALUE);

        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> mapListPair = sameDiff.execBackwards();

        SDVariable finalResult = sameDiff.grad(sum.getVarName());

        SDVariable cGrad = sameDiff.grad(varMulPre.getVarName());

        SDVariable mulGradResult = sameDiff.grad(varMul.getVarName());
        SDVariable aGrad = sameDiff.grad(sdVariable.getVarName());
        SDVariable wGrad = sameDiff.grad(sdVariable1.getVarName());
        SDVariable dGrad = sameDiff.grad(varMul.getVarName());

        INDArray scalarGradTest = finalResult.getArr();
        assertEquals(scalar, scalarGradTest);


        INDArray gradTest = mulGradResult.getArr();
        assertEquals(gradAssertion, gradTest);

        INDArray aGradTest = aGrad.getArr();
        assertEquals(aGradAssertion, aGradTest);

        INDArray cGradTest = cGrad.getArr();
        assertEquals(cGradAssertion, cGradTest);

        INDArray wGradTest = wGrad.getArr();
        assertEquals(wGradAssertion, wGradTest);

        INDArray dGradTest = dGrad.getArr();
        assertEquals(dGradAssertion, dGradTest);
    }


    @Test
    public void testMmulGradientManual() {
        SameDiff sameDiff = SameDiff.create();
        INDArray sumInput = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("x", sumInput);
        inputs.put("y", sumInput.dup());

        sameDiff.defineFunction("mmulGradient", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x", inputs.get("x"));
                SDVariable input2 = sameDiff.var("y", inputs.get("y"));
                SDVariable exp = sameDiff.mmul(input, input2);
                SDVariable sum = sameDiff.sum(exp, Integer.MAX_VALUE);
                return new SDVariable[]{sum};
            }
        }, inputs);

        List<DifferentialFunction> ops = sameDiff.getFunction("mmulGradient").execBackwards().getRight();
        String print = sameDiff.asFlatPrint();


        assumeNotNull(sameDiff.getFunction("mmulGradient").getFunction("grad"));
        assumeNotNull(sameDiff.getFunction("mmulGradient").grad("x"));
        assumeNotNull(sameDiff.getFunction("mmulGradient").grad("y"));

        SDVariable gradWrtX = sameDiff.getFunction("mmulGradient").grad("x");
        SDVariable gradWrtY = sameDiff.getFunction("mmulGradient").grad("y");
        assumeNotNull(gradWrtX.getArr());
        assumeNotNull(gradWrtY.getArr());


        INDArray xGradAssertion = Nd4j.create(new double[][]{
                {3, 7},
                {3, 7}
        });

        INDArray yGradAssertion = Nd4j.create(new double[][]{
                {4, 4},
                {6, 6}
        });

        assertEquals(xGradAssertion, gradWrtX.getArr());
        assertEquals(yGradAssertion, gradWrtY.getArr());
    }

    @Test
    public void testMmulGradients(){
        OpValidationSuite.ignoreFailing();  //https://github.com/deeplearning4j/deeplearning4j/issues/5648

        int[] aShape = new int[]{2,3};
        int[] bShape = new int[]{3,4};
        List<String> failed = new ArrayList<>();

        for( char aOrder : new char[]{'c', 'f'}) {
            for (char bOrder : new char[]{'c', 'f'}) {
                for (boolean transposeA : new boolean[]{false, true}) {
                    for (boolean transposeB : new boolean[]{false, true}) {
                        for (boolean transposeResult : new boolean[]{false, true}) {    //https://github.com/deeplearning4j/deeplearning4j/issues/5648
                            Nd4j.getRandom().setSeed(12345);

                            INDArray aArr = Nd4j.rand(t(transposeA, aShape)).dup(aOrder);
                            INDArray bArr = Nd4j.rand(t(transposeB, bShape)).dup(bOrder);

                            SameDiff sd = SameDiff.create();
                            SDVariable a = sd.var("a", aArr);
                            SDVariable b = sd.var("b", bArr);

                            MMulTranspose mt = MMulTranspose.builder()
                                    .transposeA(transposeA)
                                    .transposeB(transposeB)
                                    .transposeResult(transposeResult)
                                    .build();

                            SDVariable mmul = sd.mmul(a, b, mt);

                            INDArray exp = (transposeA ? aArr.transpose() : aArr);
                            exp = exp.mmul(transposeB ? bArr.transpose() : bArr);
                            exp = (transposeResult ? exp.transpose() : exp);

                            SDVariable loss = mmul.std(true);

                            String name = aOrder + "," + bOrder + ",tA=" + transposeA + ",tB=" + transposeB +
                                    ",tRes=" + transposeResult;
                            TestCase tc = new TestCase(sd).testName(name)
                                    .expected(mmul, exp);

                            String err = OpValidation.validate(tc, true);
                            if(err != null)
                                failed.add(err);
                        }
                    }
                }
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }

    private static int[] t(boolean transpose, int[] orig){
        if(!transpose)
            return orig;
        return new int[]{orig[1], orig[0]};
    }

    @Test
    public void testBatchMmulBasic() {
        OpValidationSuite.ignoreFailing();

        int M = 5;
        int N = 3;
        int K = 4;

        INDArray A = Nd4j.create(new int[] {M, N}, new float[]{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15});
        INDArray B = Nd4j.create(new int[] {N, K}, new float[]{1,2,3,4,5,6,7,8,9,10,11,12});

        SameDiff sd = SameDiff.create();

        SDVariable A1 = sd.var("A1", A);
        SDVariable A2 = sd.var("A2", A);
        SDVariable B1 = sd.var("B1", B);
        SDVariable B2 = sd.var("B2", B);

        SDVariable[] batchMul = sd.batchMmul(new SDVariable[] {A1, A2}, new SDVariable[] {B1, B2});
        sd.exec();

        INDArray resultingMatrix = batchMul[0].getArr();
        System.out.print(resultingMatrix);

    }


    @Test
    public void testMmulWithTranspose() {
        OpValidationSuite.ignoreFailing();

        //Here: [x,3]^T * [x,4] = [3,4]

        for (int i : new int[]{2, 1}) {
            System.out.println("i = " + i);
            INDArray first = Nd4j.linspace(1, 3 * i, 3 * i).reshape('c', i, 3);      //To [1,3] or [2,3]
            INDArray second = Nd4j.linspace(4, 4 + 4 * i, 4 * i).reshape('c', i, 4);  //To [1,4] or [2,4]

            System.out.println("Shapes: " + Arrays.toString(first.shape()) + "\t" + Arrays.toString(second.shape()));

            SameDiff sd = SameDiff.create();
            SDVariable f = sd.var("in1", first);
            SDVariable s = sd.var("in2", second);

            MMulTranspose mt = MMulTranspose.builder()
                    .transposeA(true)
                    .transposeB(false)
                    .transposeResult(false)
                    .a(first)
                    .b(second)
                    .build();
            SDVariable mmul = sd.f().mmul(f, s, mt);
            sd.updateVariableNameAndReference(mmul, "mmul");

            INDArray out = sd.execAndEndResult();

            INDArray exp = first.transpose().mmul(second);
            assertEquals(exp, out);

            SDVariable loss = sd.standardDeviation(mmul, true);
            String err = OpValidation.validate(new TestCase(sd)
                    .expected(mmul.getVarName(), exp));

            assertNull(err);
        }
    }

    @Test
    public void testFillOp(){

        INDArray ia = Nd4j.trueVector(new double[]{2,2});
        double value = 42;
        INDArray out = Nd4j.create(2,2);
        OpTestCase op = new OpTestCase(new Fill(ia, out, value));
        INDArray expOut = Nd4j.valueArrayOf(new int[]{2,2}, 42);

        op.expectedOutput(0, expOut);
        String err = OpValidation.validate(op);
        assertNull(err);
    }

    @Test
    public void testClipByNorm(){
        //Expected: if array.norm2(1) is less than 1.0, not modified
        //Otherwise: array.tad(x,1) = array.tad(x,1) * 1.0 / array.tad(x,1).norm2()

        Nd4j.getRandom().setSeed(12345);
        INDArray arr = Nd4j.rand(3,5);
        INDArray norm2_1 = arr.norm2(1);
        arr.diviColumnVector(norm2_1);

        norm2_1 = arr.norm2(1);
        assertEquals(Nd4j.ones(3), norm2_1);

        INDArray scale = Nd4j.create(new double[]{1.1, 1.0, 0.9}, new int[]{3,1});
        arr.muliColumnVector(scale);
        norm2_1 = arr.norm2(1);

        INDArray out = Nd4j.create(arr.shape());

        Nd4j.getExecutioner().exec(DynamicCustomOp.builder("clipbynorm")
                .addInputs(arr)
                .addOutputs(out)
                .addIntegerArguments(1)
                .addFloatingPointArguments(1.0)
                .build());

        INDArray norm2_1b = out.norm2(1);
        INDArray exp = Nd4j.create(new double[]{1.0, 1.0, norm2_1.getDouble(2)}, new int[]{3,1});

        assertEquals(exp, norm2_1b);
    }

    @Test
    public void testClipByNorm2(){
        //Expected: if array.norm2(1) is less than 1.0, not modified
        //Otherwise: array.tad(x,1) = array.tad(x,1) * 1.0 / array.tad(x,1).norm2()

        Nd4j.getRandom().setSeed(12345);
        INDArray arr = Nd4j.rand(3,5);
        INDArray norm2_1 = arr.norm2(1);
        arr.diviColumnVector(norm2_1);

        norm2_1 = arr.norm2(1);
        assertEquals(Nd4j.ones(3), norm2_1);

        INDArray scale = Nd4j.create(new double[]{1.1, 1.0, 0.9}, new int[]{3,1});
        arr.muliColumnVector(scale);
        norm2_1 = arr.norm2(1);

        INDArray out = Nd4j.createUninitialized(arr.shape());

        OpTestCase op = new OpTestCase(DynamicCustomOp.builder("clipbynorm")
                .addInputs(arr)
                .addOutputs(out)
                .addIntegerArguments(1)
                .addFloatingPointArguments(1.0)
                .build());

        INDArray expNorm2 = Nd4j.create(new double[]{1.0, 1.0, norm2_1.getDouble(2)}, new int[]{3,1});

        INDArray expOut = arr.divColumnVector(norm2_1).muliColumnVector(expNorm2);
        op.expectedOutput(0, expOut);

        System.out.println("Input");
        System.out.println(arr.shapeInfoToString());
        System.out.println(Arrays.toString(arr.data().asFloat()));

        System.out.println("Expected");
        System.out.println(expOut.shapeInfoToString());
        System.out.println(Arrays.toString(expOut.data().asFloat()));

        String err = OpValidation.validate(op);
        assertNull(err);
    }

    @Test
    public void testClipByNorm1(){
        //Expected: if array.norm2(1) is less than 1.0, not modified
        //Otherwise: array.tad(x,1) = array.tad(x,1) * 1.0 / array.tad(x,1).norm2()

        Nd4j.getRandom().setSeed(12345);
        INDArray arr = Nd4j.rand(3,5);
        INDArray norm2_1 = arr.norm2(1);
        arr.diviColumnVector(norm2_1);

        norm2_1 = arr.norm2(1);
        assertEquals(Nd4j.ones(3), norm2_1);

        INDArray scale = Nd4j.create(new double[]{1.1, 1.0, 0.9}, new int[]{3,1});
        arr.muliColumnVector(scale);
        norm2_1 = arr.norm2(1);

        INDArray out = Nd4j.createUninitialized(arr.shape());

        INDArray expNorm2 = Nd4j.create(new double[]{1.0, 1.0, norm2_1.getDouble(2)}, new int[]{3,1});

        INDArray expOut = arr.divColumnVector(norm2_1).muliColumnVector(expNorm2);


        OpTestCase op = new OpTestCase(
                new ClipByNorm(arr, out, 1.0, 1))
                .expectedOutput(0, expOut);

//        System.out.println("Input");
//        System.out.println(arr.shapeInfoToString());
//        System.out.println(Arrays.toString(arr.data().asFloat()));
//
//        System.out.println("Expected");
//        System.out.println(expOut.shapeInfoToString());
//        System.out.println(Arrays.toString(expOut.data().asFloat()));

        String err = OpValidation.validate(op);
        assertNull(err);
    }

    @Test
    public void testClipByNorm0(){
        //Expected: if array.norm2(0) is less than 1.0, not modified
        //Otherwise: array.tad(x,1) = array.tad(x,1) * 1.0 / array.tad(x,1).norm2()

        Nd4j.getRandom().setSeed(12345);
        INDArray arr = Nd4j.rand(5,4);
        INDArray norm2_0 = arr.norm2(0);
        arr.diviRowVector(norm2_0);

        INDArray initNorm2 = Nd4j.create(new double[]{2.2, 2.1, 2.0, 1.9}, new int[]{1,4});     //Initial norm2s along dimension 0
        arr.muliRowVector(initNorm2);
        norm2_0 = arr.norm2(0);

        assertEquals(initNorm2, norm2_0);

        INDArray out = Nd4j.create(arr.shape());

        INDArray norm2_0b = out.norm2(0);
        INDArray expNorm = Nd4j.create(new double[]{2.0, 2.0, 2.0, 1.9}, new int[]{1, 4});  //Post clip norm2s along dimension 0
        INDArray exp = arr.divRowVector(norm2_0b).muliRowVector(expNorm);

        OpTestCase op = new OpTestCase(//Clip to norm2 of 2.0, along dimension 0
                new ClipByNorm(arr, out, 2.0, 0))
                .expectedOutput(0, exp);

        assertNull(OpValidation.validate(op));
    }

    @Test
    public void testCumSum(){
        OpValidationSuite.ignoreFailing();

        List<String> failing = new ArrayList<>();
        for(char order : new char[]{'c','f'}) {

            Nd4j.getRandom().setSeed(12345);
            INDArray arr = Nd4j.linspace(1, 15, 15).reshape(3, 5).dup(order);
//            System.out.println(arr);

            INDArray expFF = Nd4j.create(new double[][]{
                    {1, 3, 6, 10, 15},
                    {6, 13, 21, 30, 40},
                    {11, 23, 36, 50, 65}
            });

            INDArray expTF = Nd4j.create(new double[][]{
                    {0, 1, 3, 6, 10},
                    {0, 6, 13, 21, 30},
                    {0, 11, 23, 36, 50}
            });

            INDArray expFT = Nd4j.create(new double[][]{
                    {15, 14, 12, 9, 5},
                    {40, 34, 27, 19, 10},
                    {65, 54, 42, 29, 15}
            });

            INDArray expTT = Nd4j.create(new double[][]{
                    {14, 12, 9, 5, 0},
                    {34, 27, 19, 10, 0},
                    {54, 42, 29, 15, 0}
            });

            INDArray axisArg = Nd4j.scalar(1);  //Along dim 1

            for (boolean exclusive : new boolean[]{false, true}) {
                for (boolean reverse : new boolean[]{false, true}) {

                    String msg = order + ", exclusive=" + exclusive + ", reverse=" + reverse;

                    INDArray out = Nd4j.create(3, 5);
                    OpTestCase op = new OpTestCase(new CumSum(arr, out, exclusive, reverse, 1));

                    if(!exclusive && !reverse){
                        op.expectedOutput(0, expFF);
                    } else if(exclusive && !reverse){
                        op.expectedOutput(0, expTF);
                    } else if(!exclusive && reverse){
                        op.expectedOutput(0, expFT);
                    } else {
                        op.expectedOutput(0, expTT);
                    }

                    String err = OpValidation.validate(op);
                    if(err != null){
//                        System.out.println(err);
                        failing.add(msg);
                    }
                }
            }
        }

        assertEquals(failing.toString(), 0, failing.size());
    }


    @Test
    public void testCumProd(){
        OpValidationSuite.ignoreFailing();

        List<String> failing = new ArrayList<>();

        for(char order : new char[]{'c','f'}) {

            Nd4j.getRandom().setSeed(12345);
            INDArray arr = Nd4j.linspace(1, 15, 15).reshape(3, 5).dup(order);
//            System.out.println(arr);

            INDArray expFF = Nd4j.create(new double[][]{
                    {1, 2, 6, 24, 120},
                    {6, 42, 336, 3024, 30240},
                    {11, 132, 1716, 24024, 360360}
            });

            INDArray expTF = Nd4j.create(new double[][]{
                    {1, 1, 2, 6, 24},
                    {1, 6, 42, 336, 3024},
                    {1, 11, 132, 1716, 24024}
            });

            INDArray expFT = Nd4j.create(new double[][]{
                    {120, 120, 60, 20, 5},
                    {30240, 5040, 720, 90, 10},
                    {360360, 32760, 2730, 210, 15}
            });

            INDArray expTT = Nd4j.create(new double[][]{
                    {120, 60, 20, 5, 1},
                    {5040, 720, 90, 10, 1},
                    {32760, 2730, 210, 15, 1}
            });

            INDArray axisArg = Nd4j.scalar(1);  //Along dim 1

            for (boolean exclusive : new boolean[]{false, true}) {
                for (boolean reverse : new boolean[]{false, true}) {

                    INDArray out = Nd4j.create(3, 5);
                    OpTestCase op = new OpTestCase(new CumProd(arr, axisArg, out, exclusive, reverse));
                    String msg = order + ", exclusive=" + exclusive + ", reverse=" + reverse;

                    if(!exclusive && !reverse){
                        op.expectedOutput(0, expFF);
                    } else if(exclusive && !reverse){
                        op.expectedOutput(0, expTF);
                    } else if(!exclusive && reverse){
                        op.expectedOutput(0, expFT);
                    } else {
                        op.expectedOutput(0, expTT);
                    }

                    String err = OpValidation.validate(op);
                    if(err != null){
//                        System.out.println(err);
                        failing.add(msg);
                    }
                }
            }
        }

        assertEquals(failing.toString(), 0, failing.size());
    }

    @Test
    public void testOneHot1(){
        OpValidationSuite.ignoreFailing();
        fail(); //JVM crash
        List<String> failed = new ArrayList<>();

        //Because it's on the diagonal, should be the same for all axis args...
        for( int i=-1; i<=0; i++ ) {
            INDArray indicesArr = Nd4j.trueVector(new double[]{0, 1, 2});
            int depth = 3;

            SameDiff sd = SameDiff.create();
            SDVariable indices = sd.var(indicesArr);
            SDVariable oneHot = sd.oneHot(indices, depth, i, 1.0, 0.0);

            INDArray exp = Nd4j.eye(3);

            String msg = "Axis: " + i;
            log.info("Test case: " + msg);

            String err = OpValidation.validate(new TestCase(sd)
                    .testName(msg)
                    .expected(oneHot, exp));

            if(err != null){
                failed.add(err);
            }
        }
        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testOneHotOp(){
        OpValidationSuite.ignoreFailing();

        //https://www.tensorflow.org/api_docs/python/tf/one_hot
        //https://github.com/deeplearning4j/deeplearning4j/blob/master/libnd4j/include/ops/declarable/generic/parity_ops/onehot.cpp

        for( int axis=-1; axis<=0; axis++ ) {
            String err = OpValidation.validate(new OpTestCase(new OneHot(Nd4j.trueVector(new double[]{0, 1, 2}),
                    Nd4j.create(3,3), 3, axis, 1.0, 0.0))
                    .expectedOutput(0, Nd4j.eye(3)));

            assertNull(err);
        }
    }

    @Test
    public void testOneHot3() {
        OpValidationSuite.ignoreFailing();

        //https://www.tensorflow.org/api_docs/python/tf/one_hot
        //indices = [[0, 2], [1, -1]]
        INDArray indicesArr = Nd4j.zeros(2, 2);
        indicesArr.put(0, 1, 2);
        indicesArr.put(1, 0, 1);
        indicesArr.put(1, 1, -1);
        INDArray expectedOut = Nd4j.zeros(new long[]{2, 2, 3});
        /*
        # output: [2 x 2 x 3]
        # [[[1.0, 0.0, 0.0],   # one_hot(0)
        #   [0.0, 0.0, 1.0]],  # one_hot(2)
        #  [[0.0, 1.0, 0.0],   # one_hot(1)
        #   [0.0, 0.0, 0.0]]]  # one_hot(-1)
        */
        expectedOut.putScalar(0, 0, 0, 1.0);
        expectedOut.putScalar(0, 1, 2, 1.0);
        expectedOut.putScalar(1, 0, 1, 1.0);

        SameDiff sd = SameDiff.create();
        SDVariable indices = sd.var("indices", new long[]{2, 2});

        int depth = 3;
        int axis = -1;
        SDVariable oneHot = sd.oneHot("oneHot", indices, depth, axis, 1.0, 0.0);

        SDVariable loss = oneHot.std(true);

        String err = OpValidation.validate(new TestCase(sd)
                .expected(oneHot, expectedOut));

        assertNull(err);
    }



    @Test
    public void testLinspace(){
        OpValidationSuite.ignoreFailing();

        SameDiff sd = SameDiff.create();
        SDVariable out = sd.linspace(1,10,10);
        SDVariable loss = out.std(true);

        String err = OpValidation.validate(new TestCase(sd)
                .expected(out, Nd4j.linspace(1,10,10)));

        assertNull(err);
    }


    @Test
    public void testShapeFn() {

        INDArray in = Nd4j.create(new long[]{1, 2});

        List<long[]> shapes = Nd4j.getExecutioner().calculateOutputShape(DynamicCustomOp.builder("shape")
                .addInputs(in)
                .build());

        assertEquals(1, shapes.size());

        assertArrayEquals(new long[]{2}, shapes.get(0));
    }

    @Test
    public void testShapeFn2() {

        INDArray i = Nd4j.create(1,3);

        SameDiff sd = SameDiff.create();
        SDVariable var = sd.var("in", i);
        SDVariable shape = sd.shape(var);
        SDVariable sum = sd.sum(shape);

        sd.execAndEndResult();
        sd.execBackwards();
    }


    @Test
    public void testMergeRank1(){
        OpValidationSuite.ignoreFailing();  //https://github.com/deeplearning4j/deeplearning4j/issues/5648
        SameDiff sd = SameDiff.create();
        SDVariable var = sd.var("in", Nd4j.create(new long[]{1}).assign(5));

        SDVariable merged = sd.mergeAvg(var);
        SDVariable sum = sd.sum(merged);

        sd.execAndEndResult();
        sd.execBackwards();

        INDArray out = merged.getArr();
        assertEquals(1, out.rank());

        INDArray inGrad = var.getGradient().getArr();
        assertEquals(1, inGrad.rank());         //Fails here, getting rank 2
    }

    @Test
    public void testDiagPart() {
        OpValidationSuite.ignoreFailing();

        INDArray i = Nd4j.create(5,5);

        SameDiff sd = SameDiff.create();
        SDVariable var = sd.var("in", i);
        SDVariable diag = sd.diagPart(var);

        INDArray out = sd.execAndEndResult();
        assertEquals(1, out.rank());
    }

    @Test
    public void testDiagShapeFn() {
        OpValidationSuite.ignoreFailing();

        INDArray i = Nd4j.create(5,5);

        CustomOp op = new DiagPart(i, null);

        List<long[]> outShape = Nd4j.getExecutioner().calculateOutputShape(op);

        assertEquals(1, outShape.size());
        assertArrayEquals(new long[]{5}, outShape.get(0));
    }


    @Test
    public void testZerosOnesLike(){
        OpValidationSuite.ignoreFailing();
        Nd4j.getRandom().setSeed(12345);

        List<int[]> shapes = Arrays.asList(new int[0], new int[]{3}, new int[]{3,4}, new int[]{3,4,5});
        List<String> failed = new ArrayList<>();

        for(boolean zeros : new boolean[]{true, false}) {
            for (int[] shape : shapes) {
                SameDiff sd = SameDiff.create();
                INDArray arr;
                if(shape.length > 0){
                    arr = Nd4j.rand(shape);
                } else {
                    arr = Nd4j.trueScalar(Nd4j.rand(new int[]{1,1}).getDouble(0));
                }
                SDVariable var = sd.var("in", arr);
                SDVariable xLike;
                if(zeros) {
                    xLike = sd.zerosLike(var);
                } else {
                    xLike = sd.onesLike(var);
                }

                SDVariable loss;
                if (shape.length > 0) {
                    loss = xLike.std(true);
                } else {
                    loss = xLike.mean();
                }

                String err = OpValidation.validate(new TestCase(sd)
                        .expected(xLike, (zeros ? Nd4j.zeros(shape) : Nd4j.ones(shape))), true);
                if(err != null){
                    failed.add(err);
                }
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testZerosLikeOp(){
        OpValidationSuite.ignoreFailing();

        INDArray arr = Nd4j.trueScalar(1.0);
        INDArray out = Nd4j.trueScalar(-1);
        INDArray exp = Nd4j.trueScalar(0);

        OpTestCase op = new OpTestCase(new ZerosLike(arr, out));
        op.expectedOutput(0, exp);

        String err = OpValidation.validate(op);
        assertNull(err);
    }


    @Test
    public void testConfusionMatrix(){

        for(boolean withMax : new boolean[]{true, false}){

            SameDiff sd = SameDiff.create();

            SDVariable labels = sd.var("labels", Nd4j.trueVector(new double[]{1, 2, 4}));
            SDVariable predictions = sd.var("predictions", Nd4j.trueVector(new double[]{2, 2, 4}));

            INDArray exp = Nd4j.create(new double[][]{
                    {0, 0, 0, 0, 0},
                    {0, 0, 1, 0, 0},
                    {0, 0, 1, 0, 0},
                    {0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 1}});

            SDVariable confMatrix;
            if(withMax){
                confMatrix = sd.confusionMatrix(labels, predictions, 5);
            } else {
                confMatrix = sd.confusionMatrix(labels, predictions);
            }

            SDVariable loss = confMatrix.std(true);


            String err = OpValidation.validate(new TestCase(sd)
                    .gradientCheck(false)   //Not gradient checkable
                    .expected(confMatrix, exp));

            assertNull(err);
        }
    }

    @Test
    public void testIsNonDecreasingIsStrictlyIncr(){
        List<long[]> shapes = Arrays.asList(null, new long[]{12}, new long[]{1,12}, new long[]{3,4}, new long[]{2,2,3});

        List<String> failed = new ArrayList<>();

        for(boolean nonDec : new boolean[]{true, false}) {
            for (long[] shape : shapes) {
                for (boolean expTrue : new boolean[]{true, false}) {
                    SameDiff sd = SameDiff.create();

                    INDArray inArr;
                    if (shape == null) {
                        inArr = Nd4j.trueScalar(1.0);
                    } else {
                        inArr = Nd4j.linspace(1, 12, 12).reshape(shape);
                    }

                    if(nonDec && !expTrue) {
                        inArr.negi();
                    }
                    if(!nonDec && !expTrue && inArr.length() > 0){
                        inArr.putScalar(inArr.length()-1, inArr.getDouble(inArr.length()-2));
                    }

                    SDVariable in = sd.var("in", inArr);
                    SDVariable out;
                    if(nonDec){
                        out = sd.isNonDecreasing(in);
                    } else {
                        out = sd.isStrictlyIncreasing(in);
                    }

                    if (shape == null) {
                        SDVariable loss = out.mean();
                    } else {
                        SDVariable loss = out.std(true);
                    }

                    INDArray exp;
                    if (expTrue || shape == null) {
                        exp = Nd4j.trueScalar(1.0);
                    } else {
                        exp = Nd4j.trueScalar(0.0);
                    }

                    String msg = (nonDec ? "isNonDecreasing" : "isStrictlyIncreasing") + " - " +  (shape == null ? "[]" : Arrays.toString(shape)) + " - expected=" + exp;
                    TestCase tc = new TestCase(sd)
                            .testName(msg)
                            .expected(out, exp);

                    String err = OpValidation.validate(tc, true);
                    if (err != null) {
                        failed.add(err);
                    }
                }
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testInplaceSubi() {
        OpValidationSuite.ignoreFailing();
        SameDiff sameDiffOuter = SameDiff.create();
        Map<String, INDArray> params = new HashMap<>();
        params.put("x", Nd4j.ones(4));
        sameDiffOuter.defineFunction("inplacesubi", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable inplace = sameDiff.var("x", inputs.get("x"));
                return new SDVariable[]{inplace.subi(1.0)};
            }
        }, params);

        sameDiffOuter.getFunction("inplacesubi").eval(params);
        assertEquals(Nd4j.zeros(4), params.get("x"));
    }
}
