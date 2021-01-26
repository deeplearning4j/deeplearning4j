/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.nd4j.OpValidationSuite;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SameDiffFunctionDefinition;
import org.nd4j.autodiff.validation.OpTestCase;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.custom.Digamma;
import org.nd4j.linalg.api.ops.custom.DivideNoNan;
import org.nd4j.linalg.api.ops.custom.Flatten;
import org.nd4j.linalg.api.ops.custom.FusedBatchNorm;
import org.nd4j.linalg.api.ops.custom.Igamma;
import org.nd4j.linalg.api.ops.custom.Igammac;
import org.nd4j.linalg.api.ops.custom.Lgamma;
import org.nd4j.linalg.api.ops.custom.Lu;
import org.nd4j.linalg.api.ops.custom.MatrixBandPart;
import org.nd4j.linalg.api.ops.custom.Polygamma;
import org.nd4j.linalg.api.ops.custom.Roll;
import org.nd4j.linalg.api.ops.custom.TriangularSolve;
import org.nd4j.linalg.api.ops.impl.broadcast.BiasAdd;
import org.nd4j.linalg.api.ops.impl.broadcast.BiasAddGrad;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.StopGradient;
import org.nd4j.linalg.api.ops.impl.reduce.Mmul;
import org.nd4j.linalg.api.ops.impl.shape.DiagPart;
import org.nd4j.linalg.api.ops.impl.shape.OneHot;
import org.nd4j.linalg.api.ops.impl.shape.ZerosLike;
import org.nd4j.linalg.api.ops.impl.transforms.CheckNumerics;
import org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByNorm;
import org.nd4j.linalg.api.ops.impl.transforms.custom.CumProd;
import org.nd4j.linalg.api.ops.impl.transforms.custom.CumSum;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Fill;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.FloorDivOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.FloorModOp;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.common.primitives.Triple;
import org.nd4j.common.util.ArrayUtil;

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
                        //bcOp = sd.scalarFloorDiv(in3, in2);
                        bcOp = new FloorDivOp(sd, in3, in2).outputVariable();
                        name = "floordiv";
                        break;
                    case 7:
                        //bcOp = sd.scalarFloorMod(in3, in2);
                        bcOp = new FloorModOp(sd, in3, in2).outputVariable();
                        name = "floormod";
                        if(OpValidationSuite.IGNORE_FAILING){
                            //https://github.com/deeplearning4j/deeplearning4j/issues/5976
                            continue;
                        }
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
        Nd4j.getRandom().setSeed(12345);

        List<String> failed = new ArrayList<>();

        for (int[] dim_sz1s : new int[][]{{0, 1}, {0, 2}, {1, 2}, {0, 1, 2}}) {

            long[] otherShape = {3, 4, 5};
            otherShape[dim_sz1s[0]] = 1;
            otherShape[dim_sz1s[1]] = 1;
            if (dim_sz1s.length == 3) {
                otherShape[dim_sz1s[2]] = 1;
            }

            for (int i = 0; i < 8; i++) {

                SameDiff sd = SameDiff.create();

                SDVariable in3 = sd.var("in3", DataType.DOUBLE, 3, 4, 5);
                SDVariable in2 = sd.var("inToBc", DataType.DOUBLE, otherShape);

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
                        //bcOp = sd.scalarFloorDiv(in3, in2);
                        bcOp = new FloorDivOp(sd, in3, in2).outputVariable();
                        name = "floordiv";
                        break;
                    case 7:
                        //bcOp = sd.scalarFloorMod(in3, in2);
                        bcOp = new FloorModOp(sd, in3, in2).outputVariable();
                        name = "floormod";
                        if(OpValidationSuite.IGNORE_FAILING){
                            //https://github.com/deeplearning4j/deeplearning4j/issues/5976
                            continue;
                        }
                        break;
                    default:
                        throw new RuntimeException();
                }

                SDVariable outVar = sd.sum(bcOp);

                String msg = "(test " + i + ": " + name + ", dimensions=" + Arrays.toString(dim_sz1s) + ")";
                log.info("*** Starting test: " + msg);

                INDArray in3Arr = Nd4j.randn(DataType.DOUBLE, 3, 4, 5).muli(100);
                INDArray in2Arr = Nd4j.randn(DataType.DOUBLE, otherShape).muli(100);

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
        if(!OpValidationSuite.IGNORE_FAILING) {
            testCases.add(new Triple<>(new long[]{1, 6}, new long[]{3, 4, 5, 1}, new long[]{3, 4, 5, 6}));
        }

        for (val p : testCases) {

            for (int i = 0; i < 8; i++) {

                SameDiff sd = SameDiff.create();

                SDVariable in3 = sd.var("in1", DataType.DOUBLE, p.getFirst());
                SDVariable in2 = sd.var("in2", DataType.DOUBLE, p.getSecond());

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
                        //bcOp = sd.scalarFloorDiv(in3, in2);
                        bcOp = new FloorDivOp(sd, in3, in2).outputVariable();
                        name = "floordiv";
                        break;
                    case 7:
                        //bcOp = sd.scalarFloorMod(in3, in2);
                        bcOp = new FloorModOp(sd, in3, in2).outputVariable();
                        name = "floormod";
                        if(OpValidationSuite.IGNORE_FAILING){
                            //https://github.com/deeplearning4j/deeplearning4j/issues/5976
                            continue;
                        }
                        break;
                    default:
                        throw new RuntimeException();
                }

                SDVariable outVar = sd.sum(bcOp);

                String msg = "(test " + i + ": " + name + ", array 1 size =" + Arrays.toString(p.getFirst())
                        + ", array 2 size = " + Arrays.toString(p.getSecond()) + ")";
                log.info("*** Starting test: " + msg);

                INDArray in3Arr = Nd4j.rand(DataType.DOUBLE, p.getFirst()).muli(100);
                INDArray in2Arr = Nd4j.rand(DataType.DOUBLE, p.getSecond()).muli(100);

                sd.associateArrayWithVariable(in3Arr, in3);
                sd.associateArrayWithVariable(in2Arr, in2);

                TestCase tc = new TestCase(sd);
                String error = OpValidation.validate(tc);
                if(error != null){
                    failed.add(name + " " + i +  " - " + error);
                }
            }
        }

        assertEquals("Failed: " + failed, 0, failed.size());
    }



    @Test
    public void testScatterOpGradients() {
        List<String> failed = new ArrayList<>();

        for (int i = 0; i < 7; i++) {
            Nd4j.getRandom().setSeed(12345);

            SameDiff sd = SameDiff.create();

            SDVariable in = sd.var("in", DataType.DOUBLE, 20, 10);
            SDVariable indices = sd.var("indices", DataType.INT, new long[]{5});
            SDVariable updates = sd.var("updates", DataType.DOUBLE, 5, 10);


            in.setArray(Nd4j.rand(DataType.DOUBLE, 20, 10));
            indices.setArray(Nd4j.create(new double[]{3, 4, 5, 10, 18}).castTo(DataType.INT));
            updates.setArray(Nd4j.rand(DataType.DOUBLE, 5, 10).muli(2).subi(1));

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
                case 5:
                    scatter = sd.scatterMax("s", in, indices, updates);
                    name = "scatterMax";
                    break;
                case 6:
                    scatter = sd.scatterMin("s", in, indices, updates);
                    name = "scatterMin";
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
                    case 5:
                        destinationRow.assign(Transforms.max(destinationRow, updateRow, true));
                        break;
                    case 6:
                        destinationRow.assign(Transforms.min(destinationRow, updateRow, true));
                        break;
                    default:
                        throw new RuntimeException();
                }
            }

            SDVariable loss = sd.sum(scatter);  //.standardDeviation(scatter, true);  //.sum(scatter);  //TODO stdev might be better here as gradients are non-symmetrical...


            TestCase tc = new TestCase(sd)
                    .expected(scatter, exp)
                    .gradCheckSkipVariables(indices.name());

            String error = OpValidation.validate(tc);
            if(error != null){
                failed.add(name);
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testScatterUpdate(){
        INDArray x = Nd4j.linspace(DataType.FLOAT, 1, 30, 1).reshape(10, 3);
        INDArray updates = Nd4j.create(new float[][]{
                {100, 101, 102},
                {200, 201, 202}});
        INDArray indices = Nd4j.createFromArray(2, 5);

        INDArray exp = x.dup();
        exp.putRow(2, updates.getRow(0));
        exp.putRow(5, updates.getRow(1));

        INDArray out = exp.ulike();
        Nd4j.exec(DynamicCustomOp.builder("scatter_upd")
                .addInputs(x, indices, updates)
                .addOutputs(out)
                .build());

        assertEquals(exp, out);
    }

    @Test
    public void testGatherGradient() {
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

                SDVariable in = sd.var("in", Nd4j.rand(DataType.DOUBLE, inShape));
                SDVariable indices = sd.constant("indices", Nd4j.createFromArray(0, 3, 7));

                INDArray gatherExp = null;
                if(rank == 2){
                    int tadDim = dim == 0 ? 1 : 0;  //Swap: pullRows dim is "tensor along dimension" vs. gather's "index is value for this dimension"
                    gatherExp = Nd4j.pullRows(in.getArr(), tadDim, new int[]{0,3,7});
                }

                SDVariable gather = sd.gather(in, indices, dim);

                SDVariable loss = sd.standardDeviation("loss", gather, true, Integer.MAX_VALUE);

                String msg = "rank=" + rank + " dim=" + dim;

                TestCase tc = new TestCase(sd)
                        .testName(msg)
                        .gradCheckSkipVariables(indices.name());

                if (gatherExp != null) {
                    tc.expected(gather, gatherExp);
                }

                String error = OpValidation.validate(tc);
                if(error != null){
                    failed.add(msg + " - " + error);
                }
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }


    @Test
    public void testTrace(){
        //TODO need to work out how to handle shape_op for scalars...
        //OpValidationSuite.ignoreFailing();
        Nd4j.getRandom().setSeed(12345);
        for( int[] inShape : new int[][]{{3,3}}){

            INDArray in = Nd4j.rand(inShape);
            SameDiff sd = SameDiff.create();
            SDVariable i = sd.var("in", in);
            SDVariable trace = sd.math().trace(i);

            double exp = Nd4j.diag(in).sumNumber().doubleValue();

            TestCase tc = new TestCase(sd)
                    .expected(trace, Nd4j.scalar(exp))
                    .testName(Arrays.toString(inShape));

            String err = OpValidation.validate(tc);

            assertNull(err);
        }
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
        SDVariable result = sameDiff.tensorMmul(x, y, new int[]{0}, new int[]{1});
        assertArrayEquals(ArrayUtil.getTensorMmulShape(new long[]{2, 2, 2}, new long[]{2, 2, 2}, new int[][]{{0}, {1}}),
                result.eval().shape());
        assertEquals(16, sameDiff.numElements());

        SDVariable loss = sameDiff.standardDeviation(result, true);
        sameDiff.addLossVariable(loss);

        String err = OpValidation.validate(new TestCase(sameDiff));
        assertNull(err);
    }

    @Test
    public void testMulGradient() {
        INDArray arr1 = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray arr2 = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);

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

        Map<String,INDArray> m = sameDiff.outputAll(null);
        Map<String,INDArray> gm = sameDiff.calculateGradients(null, m.keySet());

        SDVariable finalResult = sameDiff.grad(sum.name());

        SDVariable cGrad = sameDiff.grad(varMulPre.name());

        SDVariable mulGradResult = sameDiff.grad(varMul.name());
        SDVariable aGrad = sameDiff.grad(sdVariable.name());
        SDVariable wGrad = sameDiff.grad(sdVariable1.name());
        SDVariable dGrad = sameDiff.grad(varMul.name());

        INDArray scalarGradTest = gm.get(sum.name());
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
        INDArray sumInput = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("x", sumInput);
        inputs.put("y", sumInput.dup());

        sameDiff.defineFunction("mmulGradient", new SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x", inputs.get("x"));
                SDVariable input2 = sameDiff.var("y", inputs.get("y"));
                SDVariable exp = sameDiff.mmul(input, input2);
                SDVariable sum = sameDiff.sum(exp, Integer.MAX_VALUE);
                return new SDVariable[]{sum};
            }
        }, inputs);


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
        int[] aShape = new int[]{2,3};
        int[] bShape = new int[]{3,4};
        List<String> failed = new ArrayList<>();

        for( char aOrder : new char[]{'c', 'f'}) {
            for (char bOrder : new char[]{'c', 'f'}) {
                for (boolean transposeA : new boolean[]{false, true}) {
                    for (boolean transposeB : new boolean[]{false, true}) {
                        for (boolean transposeResult : new boolean[]{false, true}) {    //https://github.com/deeplearning4j/deeplearning4j/issues/5648
                            Nd4j.getRandom().setSeed(12345);

                            INDArray aArr = Nd4j.rand(DataType.DOUBLE, t(transposeA, aShape)).dup(aOrder);
                            INDArray bArr = Nd4j.rand(DataType.DOUBLE, t(transposeB, bShape)).dup(bOrder);

                            SameDiff sd = SameDiff.create();
                            SDVariable a = sd.var("a", aArr);
                            SDVariable b = sd.var("b", bArr);

                            SDVariable mmul = sd.mmul(a, b, transposeA, transposeB, transposeResult);

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
        OpValidationSuite.ignoreFailing();  //https://github.com/deeplearning4j/deeplearning4j/issues/6873
        int M = 5;
        int N = 3;
        int K = 4;

        INDArray A = Nd4j.create(new float[]{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}).reshape(M, N).castTo(DataType.DOUBLE);
        INDArray B = Nd4j.create(new float[]{1,2,3,4,5,6,7,8,9,10,11,12}).reshape(N, K).castTo(DataType.DOUBLE);

        SameDiff sd = SameDiff.create();

        SDVariable A1 = sd.var("A1", A);
        SDVariable A2 = sd.var("A2", A);
        SDVariable B1 = sd.var("B1", B);
        SDVariable B2 = sd.var("B2", B);

        SDVariable[] batchMul = sd.batchMmul(new SDVariable[] {A1, A2}, new SDVariable[] {B1, B2});
        Map<String,INDArray> m = sd.output(Collections.emptyMap(), sd.outputs());

        INDArray resultingMatrix = m.get(batchMul[0].name());
        //System.out.print(resultingMatrix);
    }


    @Test
    public void testMmulWithTranspose() {

        //Here: [x,3]^T * [x,4] = [3,4]

        for (int i : new int[]{2, 1}) {
            System.out.println("i = " + i);
            INDArray first = Nd4j.linspace(1, 3 * i, 3 * i, DataType.DOUBLE).reshape('c', i, 3);      //To [1,3] or [2,3]
            INDArray second = Nd4j.linspace(4, 4 + 4 * i, 4 * i, DataType.DOUBLE).reshape('c', i, 4);  //To [1,4] or [2,4]

            System.out.println("Shapes: " + Arrays.toString(first.shape()) + "\t" + Arrays.toString(second.shape()));

            SameDiff sd = SameDiff.create();
            SDVariable f = sd.var("in1", first);
            SDVariable s = sd.var("in2", second);

            MMulTranspose mt = MMulTranspose.builder()
                    .transposeA(true)
                    .transposeB(false)
                    .transposeResult(false)
                    .build();
            SDVariable mmul = sd.mmul(f, s, true, false, false);
            sd.updateVariableNameAndReference(mmul, "mmul");

            INDArray out = mmul.eval();

            INDArray exp = first.transpose().mmul(second);
            assertEquals(exp, out);

            SDVariable loss = sd.standardDeviation(mmul, true);
            String err = OpValidation.validate(new TestCase(sd)
                    .expected(mmul.name(), exp));

            assertNull(err);
        }
    }

    @Test
    public void testMmulOutputSizeCalculation(){
        //[3,2] x [2,4] with result transpose: output shape [4,3]
        INDArray a = Nd4j.create(3,2);
        INDArray b = Nd4j.create(2,4);
        INDArray z = Nd4j.create(4,3);
        Mmul m = new Mmul(a,b,z,MMulTranspose.builder()
                .transposeA(false)
                .transposeB(false)
                .transposeResult(true)
        .build());

        val outShapes = Nd4j.getExecutioner().calculateOutputShape(m);
        assertArrayEquals(new long[]{4,3}, outShapes.get(0).getShape());
        Nd4j.getExecutioner().exec(m);

        //Another case: ([3,4]*[2,4]T)T = [2,3]     -   tA=false, tB=true, tR=true
        a = Nd4j.create(3,4);
        b = Nd4j.create(2,4);
        z = Nd4j.create(2,3);
        m = new Mmul(a,b,z,MMulTranspose.builder()
                .transposeA(false)
                .transposeB(true)
                .transposeResult(true)
                .build());

        val outShapes2 = Nd4j.getExecutioner().calculateOutputShape(m);
        assertArrayEquals(new long[]{2,3}, outShapes2.get(0).getShape());
        Nd4j.getExecutioner().exec(m);

    }

    @Test
    public void testFillOp(){

        INDArray ia = Nd4j.createFromArray(new double[]{2,2}).castTo(DataType.INT);
        double value = 42;
        INDArray out = Nd4j.create(DataType.FLOAT, 2,2);
        OpTestCase op = new OpTestCase(new Fill(ia, out, value));
        INDArray expOut = Nd4j.valueArrayOf(new long[]{2,2}, 42.0f);

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

        INDArray scale = Nd4j.create(new double[]{1.1, 1.0, 0.9}, new int[]{3});
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
        INDArray exp = Nd4j.create(new double[]{1.0, 1.0, norm2_1.getDouble(2)}, new int[]{3});

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

        INDArray initNorm2 = Nd4j.create(new double[]{2.2, 2.1, 2.0, 1.9}, new int[]{4});     //Initial norm2s along dimension 0
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

        List<String> failing = new ArrayList<>();
        for(char order : new char[]{'c','f'}) {

            Nd4j.getRandom().setSeed(12345);
            INDArray arr = Nd4j.linspace(1, 15, 15, DataType.DOUBLE).reshape(3, 5).dup(order);
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
                        failing.add(msg + " (" + err + ")");
                    }
                }
            }
        }

        assertEquals(failing.toString(), 0, failing.size());
    }


    @Test
    public void testCumProd(){
        List<String> failing = new ArrayList<>();

        for(char order : new char[]{'c','f'}) {

            Nd4j.getRandom().setSeed(12345);
//            INDArray arr = Nd4j.linspace(1, 15, 15, DataType.DOUBLE).reshape('c',3, 5).dup(order);

            INDArray arr = Nd4j.create(new double[][]{
                    { 1,  2,  3,  4,  5},
                    { 6,  7,  8,  9, 10},
                    {11, 12, 13, 14, 15}});

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

                    INDArray out = Nd4j.create(DataType.DOUBLE, 3, 5);
                    OpTestCase op = new OpTestCase(new CumProd(arr, out, exclusive, reverse, 1));
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
                        failing.add(msg + " - " + err);
                    }
                }
            }
        }

        assertEquals(failing.toString(), 0, failing.size());
    }

    @Test
    public void testOneHot1(){
        List<String> failed = new ArrayList<>();

        //Because it's on the diagonal, should be the same for all axis args...
        for( int i=-1; i<=0; i++ ) {
            INDArray indicesArr = Nd4j.createFromArray(0, 1, 2);
            int depth = 3;

            SameDiff sd = SameDiff.create();
            SDVariable indices = sd.constant(indicesArr);
            SDVariable oneHot = sd.oneHot(indices, depth, i, 1.0, 0.0, DataType.DOUBLE);

            INDArray exp = Nd4j.eye(3).castTo(DataType.DOUBLE);

            String msg = "Axis: " + i;
            log.info("Test case: " + msg);

            String err = OpValidation.validate(new TestCase(sd)
                    .testName(msg)
                    .gradientCheck(false)
                    .expected(oneHot, exp));

            if(err != null){
                failed.add(err);
            }
        }
        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testOneHotOp(){
        //https://www.tensorflow.org/api_docs/python/tf/one_hot
        //https://github.com/deeplearning4j/deeplearning4j/blob/master/libnd4j/include/ops/declarable/generic/parity_ops/onehot.cpp

        for( int axis=-1; axis<=0; axis++ ) {
            String err = OpValidation.validate(new OpTestCase(new OneHot(Nd4j.create(new double[]{0, 1, 2}),
                    Nd4j.create(DataType.FLOAT,3,3), 3, axis, 1.0, 0.0))
                    .expectedOutput(0, Nd4j.eye(3).castTo(DataType.FLOAT)));

            assertNull(err);
        }
    }

    @Test
    public void testOneHot2() {

        INDArray indicesArr = Nd4j.createFromArray(0, 2, -1, 1);

        SameDiff sd = SameDiff.create();
        SDVariable indices = sd.constant("indices", indicesArr);
        int depth = 3;
        int axis = -1;
        SDVariable oneHot = sd.oneHot("oneHot", indices, depth, axis, 5.0, 0.0, DataType.DOUBLE);

        INDArray exp = Nd4j.create(new double[][]{{5, 0, 0}, {0,0,5}, {0,0,0}, {0, 5, 0}});

        String err = OpValidation.validate(new TestCase(sd)
                .expected(oneHot, exp)
                .gradientCheck(false));

        assertNull(err);
    }

    @Test
    public void testOneHot4() {

        INDArray indicesArr = Nd4j.createFromArray(0, 2, -1, 1);

        SameDiff sd = SameDiff.create();
        SDVariable indices = sd.constant("indices", indicesArr);
        int depth = 3;
        int axis = -1;
        SDVariable oneHot = sd.oneHot("oneHot", indices, depth, axis, 5.0, 0.0, DataType.INT32);

        INDArray exp = Nd4j.create(new int[][]{{5, 0, 0}, {0,0,5}, {0,0,0}, {0, 5, 0}});

        String err = OpValidation.validate(new TestCase(sd)
                .expected(oneHot, exp)
                .gradientCheck(false));

        assertNull(err);
    }

    @Test
    public void testOneHot3() {
        //https://github.com/deeplearning4j/deeplearning4j/issues/6872

        //https://www.tensorflow.org/api_docs/python/tf/one_hot
        //indices = [[0, 2], [1, -1]]
        INDArray indicesArr = Nd4j.create(new double[][]{{0, 2}, {1, -1}}).castTo(DataType.INT);
        INDArray expectedOut = Nd4j.zeros(DataType.DOUBLE, 2, 2, 3);
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
        SDVariable indices = sd.constant("indices", indicesArr);

        int depth = 3;
        int axis = -1;
        SDVariable oneHot = sd.oneHot("oneHot", indices, depth, axis, 1.0, 0.0).castTo(DataType.DOUBLE);

        SDVariable loss = oneHot.std(true);

        String err = OpValidation.validate(new TestCase(sd)
                .expected(oneHot, expectedOut)
                .gradientCheck(false));

        assertNull(err);
    }

    @Test
    public void testLinspace(){
        SameDiff sd = SameDiff.create();
        SDVariable out = sd.linspace("linspace", DataType.DOUBLE, 1,10,10);
        SDVariable loss = out.std(true);

        String err = OpValidation.validate(new TestCase(sd)
                .expected(out, Nd4j.linspace(1,10,10, DataType.DOUBLE))
                .gradientCheck(false));

        assertNull(err);
    }

    @Test
    public void testLinspace2(){
        OpValidationSuite.ignoreFailing();  //TODO 2019/01/18
        SameDiff sd = SameDiff.create();
        SDVariable out = sd.linspace("linspace", sd.constant(Nd4j.scalar(1)), sd.constant(Nd4j.scalar(10)), sd.constant(Nd4j.scalar(10)), DataType.DOUBLE);
        SDVariable loss = out.std(true);

        String err = OpValidation.validate(new TestCase(sd)
                .expected(out, Nd4j.linspace(1,10,10, DataType.DOUBLE)));

        assertNull(err);
    }


    @Test
    public void testShapeFn() {

        INDArray in = Nd4j.create(new long[]{1, 2});

        val shapes = Nd4j.getExecutioner().calculateOutputShape(DynamicCustomOp.builder("shape")
                .addInputs(in)
                .build());

        assertEquals(1, shapes.size());

        assertArrayEquals(new long[]{2}, shapes.get(0).getShape());
    }

    @Test
    public void testShapeFn2() {

        INDArray i = Nd4j.create(1,3);

        SameDiff sd = SameDiff.create();
        SDVariable var = sd.var("in", i);
        SDVariable shape = sd.shape(var);
        SDVariable sum = shape.castTo(DataType.DOUBLE).sum();
        sum.eval();
    }


    @Test
    public void testMergeRank1(){
        SameDiff sd = SameDiff.create();
        SDVariable var = sd.var("in", Nd4j.create(new long[]{1}).assign(5));

        SDVariable merged = sd.math().mergeAvg("merged", new SDVariable[]{var});
        SDVariable sum = sd.sum(merged);

        Map<String,INDArray> m = sd.output(Collections.emptyMap(), "merged");
        Map<String,INDArray> gm = sd.calculateGradients(null, "in");

        INDArray out = m.get("merged");
        assertEquals(1, out.rank());

        INDArray inGrad = gm.get("in");
        assertEquals(1, inGrad.rank());
    }

    @Test
    public void testDiagPart() {
        INDArray i = Nd4j.create(5,5);

        SameDiff sd = SameDiff.create();
        SDVariable var = sd.var("in", i);
        SDVariable diag = sd.math().diagPart(var);

        INDArray out = diag.eval();
        assertEquals(1, out.rank());
    }

    @Test
    public void testDiagShapeFn() {
        INDArray i = Nd4j.create(5,5);

        CustomOp op = new DiagPart(i, null);

        val outShape = Nd4j.getExecutioner().calculateOutputShape(op);

        assertEquals(1, outShape.size());
        assertArrayEquals(new long[]{5}, outShape.get(0).getShape());
    }


    @Test
    public void testZerosOnesLike(){
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
                    arr = Nd4j.scalar(Nd4j.rand(new int[]{1,1}).getDouble(0));
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

        INDArray arr = Nd4j.scalar(DataType.DOUBLE, 1.0);
        INDArray out = Nd4j.scalar(DataType.DOUBLE, -1);
        INDArray exp = Nd4j.scalar(DataType.DOUBLE, 0);

        OpTestCase op = new OpTestCase(new ZerosLike(arr, out));
        op.expectedOutput(0, exp);

        String err = OpValidation.validate(op);
        assertNull(err);
    }


    @Test
    public void testConfusionMatrix(){
        DataType dt = DataType.DOUBLE;

        for(boolean withMax : new boolean[]{true, false}){

            SameDiff sd = SameDiff.create();

            SDVariable labels = sd.constant("labels", Nd4j.createFromArray(1, 2, 4));
            SDVariable predictions = sd.constant("predictions", Nd4j.createFromArray(2, 2, 4));

            INDArray exp = Nd4j.create(new double[][]{
                    {0, 0, 0, 0, 0},
                    {0, 0, 1, 0, 0},
                    {0, 0, 1, 0, 0},
                    {0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 1}}).castTo(DataType.FLOAT);

            SDVariable confMatrix;
            if(withMax){
                confMatrix = sd.math().confusionMatrix(labels, predictions, 5).castTo(DataType.FLOAT);
            } else {
                confMatrix = sd.math().confusionMatrix("cm", labels, predictions, DataType.FLOAT);
            }

            SDVariable loss = confMatrix.castTo(DataType.DOUBLE).std(true);


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
                        inArr = Nd4j.scalar(1.0);
                    } else {
                        inArr = Nd4j.linspace(1, 12, 12, DataType.DOUBLE).reshape(shape);
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
                        out = sd.math().isNonDecreasing(in).castTo(DataType.DOUBLE);
                    } else {
                        out = sd.math().isStrictlyIncreasing(in).castTo(DataType.DOUBLE);
                    }

                    if (shape == null) {
                        SDVariable loss = out.mean();
                    } else {
                        SDVariable loss = out.std(true);
                    }

                    INDArray exp;
                    if (expTrue || shape == null) {
                        exp = Nd4j.scalar(1.0);
                    } else {
                        exp = Nd4j.scalar(0.0);
                    }

                    String msg = (nonDec ? "isNonDecreasing" : "isStrictlyIncreasing") + " - " +  (shape == null ? "[]" : Arrays.toString(shape)) + " - expected=" + exp;
                    TestCase tc = new TestCase(sd)
                            .testName(msg)
                            .expected(out, exp)
                            .gradientCheck(false);

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
    public void testExtractImagePatches(){
        /*
        tf.reset_default_graph()
        input = tf.reshape(tf.constant([1,2,3,4,5,6,7,8,9], dtype=tf.float32), [1,3,3,1])
        patches = tf.image.extract_image_patches(images=input, ksizes=[1,2,2,1], strides=[1,1,1,1], rates=[1,1,1,1], padding="SAME")
        linear = tf.reshape(patches, [3*3*4])
        sess = tf.Session()
        out = sess.run([patches,linear])
         */
        INDArray in = Nd4j.linspace(1,9,9, DataType.FLOAT).reshape('c', 1,3,3,1);
        INDArray out = Nd4j.create(DataType.FLOAT, 1,3,3,4);

        DynamicCustomOp op = DynamicCustomOp.builder("extract_image_patches")
                .addInputs(in)
                .addOutputs(out)
                .addIntegerArguments(
                        2,2,    //Kernel
                        1,1,    //Stride
                        1,1,    //Rates
                        1       //Same
                )
                .build();

        Nd4j.getExecutioner().exec(op);

        INDArray exp = Nd4j.create(DataType.FLOAT, 1,3,3,4);
        exp.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all())
                .assign(Nd4j.createFromArray(new double[][]{
                        {1, 2, 4, 5},
                        {2, 3, 5, 6},
                        {3, 0, 6, 0}}));

        exp.get(NDArrayIndex.point(0), NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.all())
                .assign(Nd4j.createFromArray(new double[][]{
                        {4, 5, 7, 8},
                        {5, 6, 8, 9},
                        {6, 0, 9, 0}}));

        exp.get(NDArrayIndex.point(0), NDArrayIndex.point(2), NDArrayIndex.all(), NDArrayIndex.all())
                .assign(Nd4j.createFromArray(new double[][]{
                        {7, 8, 0, 0},
                        {8, 9, 0, 0},
                        {9, 0, 0, 0}}));
        assertEquals(exp, out);
    }

    @Test
    public void testSegmentProdBpSimple(){

        INDArray segmentIdxs = Nd4j.create(new double[]{0,0,0,1,2,2,3,3}, new long[]{8}).castTo(DataType.INT);
        INDArray data = Nd4j.create(new double[]{5,1,7,2,3,4,1,3}, new long[]{8});
        INDArray grad = Nd4j.createFromArray(1.0,2.0,3.0,4.0);
        int numSegments = 4;

        INDArray gradData = data.like();
        INDArray gradIdxs = segmentIdxs.like();

        DynamicCustomOp op = DynamicCustomOp.builder("unsorted_segment_prod_bp")
                .addInputs(data,segmentIdxs,grad)
                .addIntegerArguments(numSegments)
                .addOutputs(gradData, gradIdxs)
                .build();

        Nd4j.getExecutioner().exec(op);
    }

    @Test
    public void testMmulRank4() throws Exception {
        Nd4j.getRandom().setSeed(12345);

        INDArray arr1 = Nd4j.rand(DataType.FLOAT, 32, 12, 128, 64);
        INDArray arr2 = Nd4j.rand(DataType.FLOAT, 32, 12, 128, 64);

        DynamicCustomOp op = DynamicCustomOp.builder("matmul")
                .addInputs(arr1, arr2)
                .addIntegerArguments(0, 1)      //Transpose arr2 only
                .build();

        List<LongShapeDescriptor> shapes = op.calculateOutputShape();
        assertEquals(1, shapes.size());
        long[] shape = new long[]{32,12,128,128};
        assertArrayEquals(shape, shapes.get(0).getShape());

        INDArray out = Nd4j.create(DataType.FLOAT, shape);

        INDArray outExp = out.like();
        for( int i=0; i<32; i++ ){
            for( int j=0; j<12; j++ ){
                INDArray sub1 = arr1.get(NDArrayIndex.point(i), NDArrayIndex.point(j), NDArrayIndex.all(), NDArrayIndex.all());
                INDArray sub2 = arr2.get(NDArrayIndex.point(i), NDArrayIndex.point(j), NDArrayIndex.all(), NDArrayIndex.all());
                INDArray mmul = sub1.mmul(sub2.transpose());
                outExp.get(NDArrayIndex.point(i), NDArrayIndex.point(j), NDArrayIndex.all(), NDArrayIndex.all()).assign(mmul);
            }
        }

        op.setOutputArgument(0, out);
        Nd4j.exec(op);

        assertEquals(outExp, out);
    }

    @Test
    public void testMmulRank4_simple(){

        INDArray arr1 = Nd4j.ones(DataType.FLOAT, 32, 12, 128, 64);
        INDArray arr2 = Nd4j.ones(DataType.FLOAT, 32, 12, 128, 64);

        DynamicCustomOp op = DynamicCustomOp.builder("matmul")
                .addInputs(arr1, arr2)
                .addIntegerArguments(0, 1)      //Transpose arr2 only
                .build();

        List<LongShapeDescriptor> shapes = op.calculateOutputShape();
        assertEquals(1, shapes.size());
        long[] shape = new long[]{32,12,128,128};
        assertArrayEquals(shape, shapes.get(0).getShape());

        INDArray out = Nd4j.create(DataType.FLOAT, shape);

        op.setOutputArgument(0, out);
        Nd4j.exec(op);
//        System.out.println(out);

        INDArray exp = Nd4j.valueArrayOf(shape, 64.0, DataType.FLOAT);      //Each entry in output is sum of 64 (1.0 x 1.0) multiplications
        assertEquals(exp, out);
    }

    @Test
    public void testNthElementRank1(){
        INDArray in = Nd4j.createFromArray(new double[]{0,1,2,3,4,5,6,7,8,9});
        INDArray n = Nd4j.scalar(0);
        DynamicCustomOp op = DynamicCustomOp.builder("nth_element")
                .addInputs(in,n)
                .addIntegerArguments(0) //reverse = false
                .build();

        List<LongShapeDescriptor> shapeList = op.calculateOutputShape();
        long[] shape = shapeList.get(0).getShape();
        long[] expShape = new long[0];
        assertArrayEquals(expShape, shape);

        INDArray out = Nd4j.scalar(0.0);
        op.addOutputArgument(out);

        Nd4j.getExecutioner().exec(op);
        System.out.println(out);
        assertEquals(0.0, out.getDouble(0), 1e-5);
    }

    @Test
    public void testTensorMmulShape(){
        INDArray a = Nd4j.create(new double[]{2}).reshape(1);
        INDArray b = Nd4j.create(new double[]{1, 2, 3, 4}).reshape(2, 1, 2);
        int[][] axes = new int[][]{{0},{1}};

        CustomOp op = DynamicCustomOp.builder("tensordot")
                .addInputs(a, b)
                .addIntegerArguments(axes[0].length)
                .addIntegerArguments(axes[0])
                .addIntegerArguments(axes[1].length)
                .addIntegerArguments(axes[1])
                .build();

        List<LongShapeDescriptor> l = op.calculateOutputShape();
        assertArrayEquals(new long[]{2,2}, l.get(0).getShape());         //Returning [1,2,2]
    }

    @Test
    public void testTensorMmulShape2(){
        INDArray a = Nd4j.create(new double[]{2}).reshape(1);
        INDArray b = Nd4j.create(new double[]{1, 2, 3, 4}).reshape(2, 1, 2);
        INDArray c = Nd4j.tensorMmul(a, b, new int[][]{new int[]{0}, new int[]{1}});
        assertArrayEquals(new long[]{2,2}, c.shape());
    }

    @Test
    public void testStopGradient(){

        SameDiff sd = SameDiff.create();
        SDVariable w = sd.var("w", Nd4j.rand(DataType.DOUBLE, 3, 4));
        SDVariable v = new StopGradient(sd, w).outputVariable();
        SDVariable loss = v.std(true);

        Map<String,INDArray> gm = sd.calculateGradients(null, v.name(), w.name());

        INDArray vArr = gm.get(v.name());
        INDArray wArr = gm.get(w.name());

//        System.out.println(vArr);
//        System.out.println(wArr);

        assertEquals(Nd4j.zeros(DataType.DOUBLE, 3, 4), wArr);
    }

    @Test
    public void testCheckNumerics(){
        OpValidationSuite.ignoreFailing();  //https://github.com/eclipse/deeplearning4j/issues/7927

        SameDiff sd = SameDiff.create();
        SDVariable ph = sd.placeHolder("in", DataType.DOUBLE, 3, 4);
        SDVariable msg = sd.constant("message", Nd4j.scalar("My error message!"));
        SDVariable checkNumerics = new CheckNumerics(sd, ph, msg).outputVariable();
        SDVariable loss = checkNumerics.std("loss",true);

        INDArray in = Nd4j.rand(DataType.DOUBLE, 3, 4);
        INDArray expLoss = in.std(true);

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput(checkNumerics.name(), in)
                .placeholderValue("in", in)
                .expectedOutput("loss", expLoss));
        Preconditions.checkState(err == null, err);


        //Also check that it actually does what it's supposed to:
        sd.outputAll(Collections.singletonMap("in", in));

        in.putScalar(0, Double.NaN);
        try {
            sd.outputAll(Collections.singletonMap("in", in));
            fail("Expected exception");
        } catch (Throwable t){
            //OK
        }

        in.putScalar(0, Double.POSITIVE_INFINITY);
        try {
            sd.outputAll(Collections.singletonMap("in", in));
            fail("Expected exception");
        } catch (Throwable t){
            //OK
        }

        in.putScalar(0, 0.0);
        sd.outputAll(Collections.singletonMap("in", in));
    }

    @Test
    public void testCheckNumerics2() {
        INDArray in = Nd4j.rand(DataType.DOUBLE, 3, 4);
        INDArray msg = Nd4j.scalar("My error message!");

        DynamicCustomOp op = DynamicCustomOp.builder("check_numerics")
                .addInputs(in, msg)
                .addOutputs(in.like())
                .build();

        Nd4j.getExecutioner().exec(op);
    }

    @Test
    public void testHistogramFixedWidth(){
        //Bins: [-inf, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, inf]
        INDArray in = Nd4j.createFromArray(0.0, 0.1, 0.1, 0.3, 0.5, 0.5, 0.9);
        INDArray range = Nd4j.createFromArray(0.0, 1.0);
        INDArray n = Nd4j.scalar(5);

        INDArray out = Nd4j.create(DataType.INT, 5);

        Nd4j.exec(DynamicCustomOp.builder("histogram_fixed_width")
                .addInputs(in, range, n)
                .addOutputs(out)
                .build());

        INDArray exp = Nd4j.createFromArray(3, 1, 2, 0, 1);
        assertEquals(exp, out);
    }

    @Test
    public void testDynamicPartition(){
        INDArray data = Nd4j.createFromArray(2, 1, 2, 0);
        INDArray partitions = Nd4j.createFromArray(0, 2, 1, 0);
        INDArray[] out = Nd4j.exec(DynamicCustomOp.builder("dynamic_partition")
                .addOutputs(Nd4j.createUninitialized(DataType.INT, 2), Nd4j.createUninitialized(DataType.INT, 1), Nd4j.createUninitialized(DataType.INT, 1))
                .addIntegerArguments(3) //3 partitions
                .addInputs(data, partitions).build());

        INDArray exp0 = Nd4j.createFromArray(2, 0);
        INDArray exp1 = Nd4j.createFromArray(2);
        INDArray exp2 = Nd4j.createFromArray(1);

        assertEquals(exp0, out[0]);     //Usually just gives [0,0]
        assertEquals(exp1, out[1]);
        assertEquals(exp2, out[2]);
    }

    @Test
    public void testListDiff(){
        INDArray x = Nd4j.createFromArray(0, 1, 2, 3);
        INDArray y = Nd4j.createFromArray(3, 1);

        INDArray out = Nd4j.create(DataType.INT, 2);
        INDArray outIdx = Nd4j.create(DataType.INT, 2);

        Nd4j.exec(DynamicCustomOp.builder("listdiff")
                .addInputs(x, y)
                .addOutputs(out, outIdx)
                .build());

        INDArray exp = Nd4j.createFromArray(0, 2);

        assertEquals(exp, out);         //Values in x not in y
        assertEquals(exp, outIdx);      //Indices of the values in x not in y
    }

    @Test
    public void testDivideNoNan() {
        OpValidationSuite.ignoreFailing();  //TODO: implement DivideNoNan.doDiff()

        SameDiff sameDiff = SameDiff.create();

        INDArray in1 = Nd4j.linspace(1, 12, 12).reshape(3, 4);
        INDArray in2 = Nd4j.linspace(1, 12, 12).reshape(3, 4);

        SDVariable input1 = sameDiff.var(in1);
        SDVariable input2 = sameDiff.var(in2);

        INDArray expected = Nd4j.ones(3,4);

        SDVariable output = new DivideNoNan(sameDiff, input1, input2).outputVariable();

        TestCase tc = new TestCase(sameDiff)
                .gradientCheck(true)
                .expectedOutput(output.name(), expected);

        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testDigamma() {

        INDArray in1 = Nd4j.linspace(1, 12, 12).reshape(3, 4);

        INDArray expected = Nd4j.createFromArray(new double[]{
                -0.5772157,0.42278433,0.9227843,1.2561177,1.5061177,1.7061176,1.8727844,2.0156415,2.1406415,2.2517526,2.3517525,2.4426618
        }).reshape(3,4);

        val tc = new OpTestCase(new Digamma(in1)).expectedOutput(0, expected);

        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testFlatten() {

        SameDiff sameDiff = SameDiff.create();

        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1, 27, 1).reshape(3,3,3);
        SDVariable sdx = sameDiff.var(x);

        INDArray expected = Nd4j.linspace(DataType.DOUBLE,1,27,1);

        SDVariable output = new Flatten(sameDiff, 'c', sdx).outputVariable();
        SDVariable loss = sameDiff.standardDeviation(sdx, true);
        sameDiff.addLossVariable(loss);

        TestCase tc = new TestCase(sameDiff)
                .gradientCheck(true)
                .expectedOutput(output.name(), expected);

        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testFusedBatchNorm() {
        OpValidationSuite.ignoreFailing();
        SameDiff sameDiff = SameDiff.create();

        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 2*2*3*4).reshape(2,2,3,4);
        INDArray scale = Nd4j.create(DataType.DOUBLE, 4);
        scale.assign(0.5);
        INDArray offset = Nd4j.create(DataType.DOUBLE, 4);
        offset.assign(2.0);

        SDVariable input1 = sameDiff.var(x);
        SDVariable input2 = sameDiff.var(scale);
        SDVariable input3 = sameDiff.var(offset);

        INDArray expectedY = Nd4j.createFromArray(new double[]{
                985.5258,  985.5258,  985.5258,  985.5258,
                659.7321,  659.7321,  659.7321,  659.7321,
                399.0972,  399.0972,  399.0972,  399.0972,
                203.6210,  203.6210,  203.6210,  203.6210,
                73.3036,   73.3036,   73.3036,   73.3036,
                8.1448,    8.1448,    8.1448,    8.1448,
                8.1448,    8.1448,    8.1448,    8.1448,
                73.3036,   73.3036,   73.3036,   73.3036,
                203.6210,  203.6210,  203.6210,  203.6210,
                399.0972,  399.0972,  399.0972,  399.0972,
                659.7321,  659.7321,  659.7321,  659.7321,
                985.5258,  985.5258,  985.5258,  985.5258}).reshape(x.shape());
        INDArray expectedBatchMean = Nd4j.createFromArray(new double[]{23.,  24.,  25.,  26.});
        INDArray expectedBatchVar = Nd4j.createFromArray(new double[]{208.00001526,  208.00001526,  208.00001526,  208.00001526});

        SDVariable[] outputs = new FusedBatchNorm(sameDiff, input1, input2, input3, 0, 1).outputVariables();
        SDVariable loss = sameDiff.standardDeviation(input1, true);
        sameDiff.addLossVariable(loss);

        TestCase tc = new TestCase(sameDiff)
                .gradientCheck(true)
                .expectedOutput(outputs[0].name(), expectedY)
                .expectedOutput(outputs[1].name(), expectedBatchMean)
                .expectedOutput(outputs[2].name(), expectedBatchVar);

        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testIgamma() {

        INDArray in1 = Nd4j.linspace(1, 12, 12).reshape(3, 4);
        INDArray in2 = Nd4j.linspace(1, 12, 12).reshape(3, 4);

        INDArray expected = Nd4j.createFromArray(new double[]{
                0.63212055,0.59399414,0.5768099,0.56652874,0.5595013,0.5542634,0.5501591,0.5463888,0.54329145,0.54048204,0.5378594,0.53233755
        }).reshape(3,4);

        val tc = new OpTestCase(new Igamma(in1, in2)).expectedOutput(0, expected);

        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testIgammaC() {

        INDArray in1 = Nd4j.linspace(1, 12, 12).reshape(3, 4);
        INDArray in2 = Nd4j.linspace(1, 12, 12).reshape(3, 4);


        INDArray expected = Nd4j.createFromArray(new double[]{
                0.36787945,0.40600586,0.42319012,0.43347126,0.4404987,0.44573656,0.4498409,0.45361117,0.45670855,0.459518,0.46214062,0.46766248
        }).reshape(3,4);

        val tc = new OpTestCase(new Igammac(in1, in2)).expectedOutput(0, expected);

        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testLgamma() {

        SameDiff sameDiff = SameDiff.create();

        INDArray in = Nd4j.linspace(DataType.DOUBLE, 1, 12, 1).reshape(3, 4);
        SDVariable sdInput = sameDiff.var(in);

        INDArray expected = Nd4j.createFromArray(new double[]{
                0.0,0.0,0.6931472,1.7917595,3.1780539,4.787492,6.5792513,8.525162,10.604603,12.801827,15.104413,17.502308
        }).reshape(3,4);

        SDVariable output = new Lgamma(sameDiff, sdInput).outputVariable();

        SDVariable loss = sameDiff.standardDeviation(sdInput, true);
        sameDiff.addLossVariable(loss);

        TestCase tc = new TestCase(sameDiff)
                .gradientCheck(true)
                .expectedOutput(output.name(), expected);

        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testLu() {

        SameDiff sameDiff = SameDiff.create();

        INDArray in1 = Nd4j.createFromArray(new double[]{
                1., 2., 3., 0., 2., 3., 0., 0., 7.
        }).reshape(3,3);

        SDVariable input1 = sameDiff.var(in1);

        INDArray expected = Nd4j.createFromArray(new double[]{
                1., 2., 3., 0., 2., 3., 0., 0., 7
        }).reshape(3,3);

        INDArray pexpected = Nd4j.createFromArray(new int[]{
                0, 1, 2
        });

        sameDiff.loss.l2Loss(input1);
        SDVariable[] output = new Lu(sameDiff, input1).outputVariables();

        TestCase tc = new TestCase(sameDiff)
                .gradientCheck(true)
                .expectedOutput(output[0].name(), expected)
                .expectedOutput(output[1].name(), pexpected);

        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testMatrixBandPart() {
        OpValidationSuite.ignoreFailing();
        SameDiff sameDiff = SameDiff.create();

        INDArray input = Nd4j.createFromArray(new float[]{0.7788f,0.8012f,0.7244f,0.2309f,
                0.7271f,0.1804f,0.5056f,0.8925f,
                0.5461f,0.9234f,0.0856f,0.7938f}).reshape(3,4);

        SDVariable sdInput = sameDiff.var(input);
        SDVariable sdInput1 = sameDiff.constant(1);
        SDVariable sdInput2 = sameDiff.constant(-1);

        INDArray expected = Nd4j.createFromArray(new float[]{
                0.7788f,    0.8012f,    0.7244f,    0.2309f,
                0.7271f,    0.1804f,    0.5056f,    0.8925f,
                0.f,    0.9234f,    0.0856f,    0.7938f
        }).reshape(3,4);

        sameDiff.loss.l2Loss(sdInput);
        SDVariable output = new MatrixBandPart(sameDiff, sdInput, 1, -1).outputVariable();

        TestCase tc = new TestCase(sameDiff)
                .gradientCheck(true)
                .expectedOutput(output.name(), expected);

        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testPolygamma() {

        INDArray in1 = Nd4j.linspace(1, 12, 12).reshape(3, 4);
        INDArray in2 = Nd4j.linspace(1, 12, 12).reshape(3, 4);

        INDArray expected = Nd4j.createFromArray(new double[]{
                1.644934,-0.4041138,0.1189394,-0.03750069,0.01226151,-0.0041002957,0.001392272,-4.780109E-4,1.6549716E-4,-5.7675967E-5,2.0206635E-5,-7.1101636E-6
        }).reshape(3,4);

        val tc = new OpTestCase(new Polygamma(in1, in2)).expectedOutput(0, expected);

        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testTriangularSolve() {

        INDArray a = Nd4j.createFromArray(new float[]{
                3.f,  0.f,  0.f,  0.f,
                2.f,  1.f,  0.f,  0.f,
                1.f,  0.f,  1.f,  0.f,
                1.f,  1.f,  1.f,  1.f
        }).reshape(4,4);

        INDArray b = Nd4j.createFromArray(new float[]{
                4.f, 2.f, 4.f, 2.f
        }).reshape(4,1);

        INDArray expected = Nd4j.createFromArray(new float[]{
                1.333333f,  2.0f, 4.0f, 2.0f
        }).reshape(4,1);

        val tc = new OpTestCase(new TriangularSolve(a, b, false, true)).expectedOutput(0, expected);

        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testBiasAdd() {

        SameDiff sameDiff = SameDiff.create();

        INDArray in1 = Nd4j.linspace(1, 12, 12);
        INDArray in2 = Nd4j.linspace(1, 12, 12);

        SDVariable input1 = sameDiff.var(in1);
        SDVariable input2 = sameDiff.var(in2);

        INDArray expected = Nd4j.createFromArray(new double[]{
                2.0000,    4.0000,    6.0000,    8.0000,   10.0000,   12.0000,   14.0000,   16.0000,   18.0000,   20.0000,   22.0000,   24.0000
        });

        SDVariable output = new BiasAdd(sameDiff, input1, input2, false).outputVariable();
        SDVariable loss = sameDiff.standardDeviation(input1, true);
        sameDiff.addLossVariable(loss);
        SDVariable loss2 = sameDiff.standardDeviation(input2, true);
        sameDiff.addLossVariable(loss2);

        TestCase tc = new TestCase(sameDiff)
                .gradientCheck(true)
                .expectedOutput(output.name(), expected);

        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testBiasAddGrad() {

        SameDiff sameDiff = SameDiff.create();

        INDArray x = Nd4j.linspace(DataType.FLOAT,1, 24, 24).reshape(2,2,2,3);
        INDArray grad = Nd4j.linspace(DataType.FLOAT, 0.1, 0.1, 24).reshape(2,2,2,3);

        INDArray bias = Nd4j.createFromArray(new float[]{-1.f, -2.f, -3.f});

        INDArray expected = Nd4j.createFromArray(new float[]{9.2f, 10.f , 10.8f});

        OpTestCase tc = new OpTestCase(new BiasAddGrad(x, bias, grad,false)).
                expectedOutput(0, grad).
                expectedOutput(1, expected);

        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testRoll() {

        INDArray x = Nd4j.createFromArray(new double[]{    11.11, 11.12, 11.21, 11.22, 11.31, 11.32, 11.41, 11.42,     12.11, 12.12, 12.21, 12.22, 12.31, 12.32, 12.41, 12.42,
                21.11, 21.12, 21.21, 21.22, 21.31, 21.32, 21.41, 21.42,     22.11, 22.12, 22.21, 22.22, 22.31, 22.32, 22.41, 22.42}).
                reshape(2,2,4,2);

        INDArray expected = Nd4j.createFromArray(new double[]{    22.21, 22.22, 22.31, 22.32, 22.41, 22.42, 11.11, 11.12, 11.21, 11.22, 11.31, 11.32, 11.41, 11.42,
                12.11, 12.12, 12.21, 12.22, 12.31, 12.32, 12.41, 12.42, 21.11, 21.12, 21.21, 21.22, 21.31, 21.32,
                21.41, 21.42, 22.11, 22.12
        }).reshape(x.shape());

        int shift = 6;

        val tc = new OpTestCase(new Roll(x,shift)).expectedOutput(0,expected);
        String err = OpValidation.validate(tc);

        assertNull(err);
    }

    @Test
    public void testSeqMask(){
        INDArray arr = Nd4j.createFromArray(1,2,3);
        INDArray maxLen = Nd4j.scalar(4);

        INDArray out = Nd4j.create(DataType.INT32, 3, 4);
        out.assign(Integer.MAX_VALUE);

        Nd4j.exec(DynamicCustomOp.builder("sequence_mask")
                .addInputs(arr, maxLen)
                .addOutputs(out)
                .build()
        );

        INDArray exp = Nd4j.createFromArray(new int[][]{
                {1, 0, 0, 0},
                {1, 1, 0, 0},
                {1, 1, 1, 0}});

        assertEquals(exp, out);
    }
}
