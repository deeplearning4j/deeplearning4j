package org.nd4j.autodiff.opvalidation;

import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.autodiff.validation.OpTestCase;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.accum.bp.*;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.fail;

public class ReductionOpValidationTests {

    private DataBuffer.Type initialType;

    @Before
    public void before() throws Exception {
        Nd4j.create(1);
        initialType = Nd4j.dataType();

        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
        Nd4j.getRandom().setSeed(123);
    }

    @After
    public void after() throws Exception {
        Nd4j.setDataType(initialType);
    }


    @After
    public void tearDown() throws Exception {
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableDebugMode(false);
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableVerboseMode(false);
    }


    @Test
    public void testReduceSum(){
        //Full array reduction

        //reduce_sum_bp op: has 2 inputs (original pre-reduce input, and gradient at output (epsilon))
        //out = sum_j (in_j) -> dL/dIn = dL/dOut * dOut/dIn = dL/dOut

        for(boolean keepDims : new boolean[]{false, true}) {

            INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);
            INDArray dLdOut;
            if(keepDims){
                dLdOut = Nd4j.valueArrayOf(new long[]{1,1}, 0.5);
            } else {
                dLdOut = Nd4j.trueScalar(0.5);
            }
            INDArray dLdInExpected = Nd4j.valueArrayOf(preReduceInput.shape(), 0.5);
            INDArray dLdIn = Nd4j.createUninitialized(3, 4);

            String err = OpValidation.validate(new OpTestCase(new SumBp(preReduceInput, dLdOut, dLdIn, keepDims))
                    .expectedOutput(0, dLdInExpected));

            assertNull(err);
        }
    }

    @Test
    public void testReduceSumAlongDim0(){
        //Reduction along dimension
        //Inputs/outputs as before - but note that the output is no longer a scalar

        //Note: when reducing [3,4] along dimension 0 -> 4 TADs of length 3
        //We have one epsilon/gradient for each of the 4 TADs -> dL/dOut length is 4

        for( boolean keepDims : new boolean[]{false, true}) {
            long[] reducedShape_0 = (keepDims ? new long[]{1,4} : new long[]{4});
            INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);
            INDArray dLdOut_0 = Nd4j.create(new double[]{1, 2, 3, 4}, reducedShape_0);
            INDArray dLdInExpected_0 = Nd4j.createUninitialized(preReduceInput.shape());
            for (int i = 0; i < 3; i++) {
                dLdInExpected_0.putRow(i, dLdOut_0);
            }

            INDArray dLdIn = Nd4j.createUninitialized(3, 4);

            String err = OpValidation.validate(new OpTestCase(new SumBp(preReduceInput, dLdOut_0, dLdIn, keepDims, 0))
                    .expectedOutput(0, dLdInExpected_0));

            assertNull(err);
        }
    }

    @Test
    public void testReduceSumAlongDim1(){
        //Reduction along dimension
        //Inputs/outputs as before - but note that the output is no longer a scalar

        //Note: when reducing [3,4] along dimension 1 -> 3 TADs of length 4
        //We have one epsilon/gradient for each of the 3 TADs -> dL/dOut length is 3

        for( boolean keepDims : new boolean[]{false, true}) {
            INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);

            long[] reducedShape_1 = (keepDims ? new long[]{3,1} : new long[]{3});
            INDArray dLdOut_1 = Nd4j.create(new double[]{1, 2, 3}, reducedShape_1);
            INDArray dLdInExpected_1 = Nd4j.createUninitialized(preReduceInput.shape());
            for (int i = 0; i < 4; i++) {
                dLdInExpected_1.putColumn(i, dLdOut_1);
            }

            INDArray dLdIn = Nd4j.createUninitialized(3, 4);

            String err = OpValidation.validate(new OpTestCase(new SumBp(preReduceInput, dLdOut_1, dLdIn, keepDims, 0))
                    .expectedOutput(0, dLdInExpected_1));

            assertNull(err);
        }
    }


    @Test
    public void testMean(){

        //dL/dIn_i = dL/dOut * dOut/dIn_i = dL/dOut * (1/N * sum_j (in_j))
        //         = 1/N * dL/dOut
        // i.e., same as SUM case but divided by N
        //NOTE: N = num values in array
        //But for "along dimension" case - it's the number of elements in that TAD

        //Full array reduction
        //reduce_mean_bp op: has 2 inputs (original pre-reduce input, and gradient at output (epsilon))

        for(boolean keepDims : new boolean[]{false, true}) {

            INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);
            INDArray dLdOut;
            if(keepDims){
                dLdOut = Nd4j.valueArrayOf(new long[]{1,1}, 0.5);
            } else {
                dLdOut = Nd4j.trueScalar(0.5);
            }
            INDArray dLdInExpected = Nd4j.valueArrayOf(preReduceInput.shape(), 0.5 / preReduceInput.length());
            INDArray dLdIn = Nd4j.createUninitialized(3, 4);

            String err = OpValidation.validate(new OpTestCase(new MeanBp(preReduceInput, dLdOut, dLdIn, keepDims))
                    .expectedOutput(0, dLdInExpected));

            assertNull(err);
        }
    }

    @Test
    public void testMeanAlongDim0(){
        //Reduction along dimension
        //Inputs/outputs as before - but note that the output is no longer a scalar

        //Note: when reducing [3,4] along dimension 0 -> 4 TADs of length 3 -> N=3 -> dL/dIn_i = dL/dOut * 1/3
        //We have one epsilon/gradient for each of the 4 TADs -> dL/dOut length is 4

        for( boolean keepDims : new boolean[]{false, true}) {
            long[] reducedShape_0 = (keepDims ? new long[]{1,4} : new long[]{4});
            INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);
            INDArray dLdOut_0 = Nd4j.create(new double[]{1, 2, 3, 4}, reducedShape_0);
            INDArray dLdInExpected_0 = Nd4j.createUninitialized(preReduceInput.shape());
            for (int i = 0; i < 3; i++) {
                dLdInExpected_0.putRow(i, dLdOut_0.div(3));
            }

            INDArray dLdIn = Nd4j.createUninitialized(3, 4);

//            String err = OpValidation.validate(new OpTestCase(
//                    DynamicCustomOp.builder("reduce_mean_bp")
//                            .addInputs(preReduceInput, dLdOut_0)
//                            .addOutputs(dLdIn)
//                            .addFloatingPointArguments(keepDims ? 1.0 : 0.0)
//                            .addIntegerArguments(0)
//                            .build())
//                    .expectedOutput(0, dLdInExpected_0)
//            );

            String err = OpValidation.validate(new OpTestCase(new MeanBp(preReduceInput, dLdOut_0, dLdIn, keepDims, 0))
                    .expectedOutput(0, dLdInExpected_0));

            assertNull(err);
        }
    }

    @Test
    public void testMeanAlongDim1(){
        //Reduction along dimension
        //Inputs/outputs as before - but note that the output is no longer a scalar

        //Note: when reducing [3,4] along dimension 1 -> 3 TADs of length 4 -> N=4 -> dL/dIn_i = dL/dOut * 1/4
        //We have one epsilon/gradient for each of the 3 TADs -> dL/dOut length is 3

        for( boolean keepDims : new boolean[]{false, true}) {
            INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);

            long[] reducedShape_1 = (keepDims ? new long[]{3,1} : new long[]{3});
            INDArray dLdOut_1 = Nd4j.create(new double[]{1, 2, 3}, reducedShape_1);
            INDArray dLdInExpected_1 = Nd4j.createUninitialized(preReduceInput.shape());
            for (int i = 0; i < 4; i++) {
                dLdInExpected_1.putColumn(i, dLdOut_1.div(4));
            }

            INDArray dLdIn = Nd4j.createUninitialized(3, 4);

            String err = OpValidation.validate(new OpTestCase(new MeanBp(preReduceInput, dLdOut_1, dLdIn, keepDims, 1))
                    .expectedOutput(0, dLdInExpected_1));

            assertNull(err);
        }
    }


    @Test
    public void testMin(){
        //Full array min reduction

        //dL/dIn_i  = dL/dOut * dOut/dIn_i
        //          = dL/dOut                   if in_i == out (== min(in))
        //          = 0                         otherwise

        for(boolean keepDims : new boolean[]{false, true}) {

            INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);
            preReduceInput.putScalar(new int[]{2,2}, -1);   //Minimum value at position [2,2]
            INDArray dLdOut;
            if(keepDims){
                dLdOut = Nd4j.valueArrayOf(new long[]{1,1}, 0.5);
            } else {
                dLdOut = Nd4j.trueScalar(0.5);
            }
            INDArray dLdInExpected = Nd4j.zeros(preReduceInput.shape());
            dLdInExpected.putScalar(new int[]{2,2}, 0.5);   //Minimum value: position at [2,2]
            INDArray dLdIn = Nd4j.createUninitialized(3, 4);

//            String err = OpValidation.validate(new OpTestCase(
//                    DynamicCustomOp.builder("reduce_min_bp")
//                            .addInputs(preReduceInput, dLdOut)
//                            .addOutputs(dLdIn)
//                            //First int arg: Keep dimensions. Lack of other (dimension) args: means "full array reduce"
////                            .addIntegerArguments(keepDims ? 1 : 0)
//                            .addFloatingPointArguments(keepDims ? 1.0 : 0.0)
//                            .build())
//                    .expectedOutput(0, dLdInExpected));

            String err = OpValidation.validate(new OpTestCase(new MinBp(preReduceInput, dLdOut, dLdIn, keepDims))
                    .expectedOutput(0, dLdInExpected));

            assertNull(err);
        }
    }

    @Test
    public void testMinAlongDimension(){
        //Full array min reduction

        //dL/dIn_i  = dL/dOut * dOut/dIn_i
        //          = dL/dOut                   if in_i == out (== min(in))
        //          = 0                         otherwise

        for(boolean keepDims : new boolean[]{false, true}) {

            long[] reducedShape_0 = (keepDims ? new long[]{1,4} : new long[]{4});
            INDArray preReduceInput = Nd4j.linspace(1, 16, 16).reshape(4, 4);
            preReduceInput.putScalar(0,0, -1);
            preReduceInput.putScalar(1,1, -2);
            preReduceInput.putScalar(2,2, -3);
            preReduceInput.putScalar(2,2, -4);
            INDArray dLdOut_0 = Nd4j.create(new double[]{1, 2, 3, 4}, reducedShape_0);
            INDArray dLdInExpected_0 = Nd4j.create(preReduceInput.shape()); //All 0s except along diagonal
            dLdInExpected_0.putScalar(0,0,1);
            dLdInExpected_0.putScalar(1,1,2);
            dLdInExpected_0.putScalar(2,2,3);
            dLdInExpected_0.putScalar(3,3,4);

            INDArray dLdIn = Nd4j.createUninitialized(4, 4);

            String err = OpValidation.validate(new OpTestCase(new MinBp(preReduceInput, dLdOut_0, dLdIn, keepDims, 0))
                    .expectedOutput(0, dLdInExpected_0));
            assertNull(err, err);


            long[] reducedShape_1 = (keepDims ? new long[]{4,1} : new long[]{4});
            INDArray dLdInExpected_1 = dLdInExpected_0; //Same here, only because the maximums are along the diagonal

            INDArray dLdOut_1 = Nd4j.create(new double[]{1, 2, 3, 4}, reducedShape_1);
            dLdIn = Nd4j.createUninitialized(4, 4);

            err = OpValidation.validate(new OpTestCase(new MinBp(preReduceInput, dLdOut_1, dLdIn, keepDims, 1))
                    .expectedOutput(0, dLdInExpected_1));

            assertNull(err, err);
        }
    }

    @Test
    public void testMax(){
        //Full array max reduction

        //dL/dIn_i  = dL/dOut * dOut/dIn_i
        //          = dL/dOut                   if in_i == out (== max(in))
        //          = 0                         otherwise

        for(boolean keepDims : new boolean[]{false, true}) {

            INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);
            preReduceInput.putScalar(new int[]{2,2}, 20);   //Maximum value at position [2,2]
            INDArray dLdOut;
            if(keepDims){
                dLdOut = Nd4j.valueArrayOf(new long[]{1,1}, 0.5);
            } else {
                dLdOut = Nd4j.trueScalar(0.5);
            }
            INDArray dLdInExpected = Nd4j.zeros(preReduceInput.shape());
            dLdInExpected.putScalar(new int[]{2,2}, 0.5);   //Maximum value: position at [2,2]
            INDArray dLdIn = Nd4j.createUninitialized(3, 4);

            String err = OpValidation.validate(new OpTestCase(new MaxBp(preReduceInput, dLdOut, dLdIn, keepDims))
                    .expectedOutput(0, dLdInExpected));

            assertNull(err);
        }

    }

    @Test
    public void testMaxAlongDimension(){
        //Full array min reduction

        //dL/dIn_i  = dL/dOut * dOut/dIn_i
        //          = dL/dOut                   if in_i == out (== min(in))
        //          = 0                         otherwise

        for(boolean keepDims : new boolean[]{false, true}) {

            long[] reducedShape_0 = (keepDims ? new long[]{1,4} : new long[]{4});
            INDArray preReduceInput = Nd4j.linspace(1, 16, 16).reshape(4, 4);
            preReduceInput.putScalar(0,0, 20);
            preReduceInput.putScalar(1,1, 21);
            preReduceInput.putScalar(2,2, 22);
            preReduceInput.putScalar(2,2, 23);
            INDArray dLdOut_0 = Nd4j.create(new double[]{1, 2, 3, 4}, reducedShape_0);
            INDArray dLdInExpected_0 = Nd4j.create(preReduceInput.shape());
            dLdInExpected_0.putScalar(0,0,1);
            dLdInExpected_0.putScalar(1,1,2);
            dLdInExpected_0.putScalar(2,2,3);
            dLdInExpected_0.putScalar(3,3,4);

            INDArray dLdIn = Nd4j.createUninitialized(4, 4);

            String err = OpValidation.validate(new OpTestCase(new MaxBp(preReduceInput, dLdOut_0, dLdIn, keepDims, 0))
                    .expectedOutput(0, dLdInExpected_0));
            assertNull(err, err);


            long[] reducedShape_1 = (keepDims ? new long[]{4,1} : new long[]{4});
            INDArray dLdInExpected_1 = dLdInExpected_0; //Same here, only because the maximums are along the diagonal

            INDArray dLdOut_1 = Nd4j.create(new double[]{1, 2, 3, 4}, reducedShape_1);
            dLdIn = Nd4j.createUninitialized(4, 4);

            err = OpValidation.validate(new OpTestCase(new MaxBp(preReduceInput, dLdOut_1, dLdIn, keepDims, 1))
                    .expectedOutput(0, dLdInExpected_1));

            assertNull(err, err);
        }
    }

    @Test
    public void testProd(){
        //Full array product reduction

        //dL/dIn_i  = dL/dOut * dOut/dIn_i
        //          = dL/dOut * d(prod(in))/dIn_i
        //          = dL/dOut * (prod(in) / in_i)

        for(boolean keepDims : new boolean[]{false, true}) {

            INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);
            INDArray dLdOut;
            if(keepDims){
                dLdOut = Nd4j.valueArrayOf(new long[]{1,1}, 0.5);
            } else {
                dLdOut = Nd4j.trueScalar(0.5);
            }
            double prod = preReduceInput.prodNumber().doubleValue();
            INDArray dLdInExpected = Nd4j.valueArrayOf(preReduceInput.shape(), prod).divi(preReduceInput).muli(0.5);

            INDArray dLdIn = Nd4j.createUninitialized(3, 4);

            String err = OpValidation.validate(new OpTestCase(new ProdBp(preReduceInput, dLdOut, dLdIn, keepDims))
                    .expectedOutput(0, dLdInExpected));

            assertNull(err);
        }
    }

    @Test
    public void testProdAlongDimension(){
        //dL/dIn_i  = dL/dOut * dOut/dIn_i
        //          = dL/dOut * d(prod(in))/dIn_i
        //          = dL/dOut * (prod(in) / in_i)

        for(boolean keepDims : new boolean[]{false, true}) {
            long[] reducedShape_0 = (keepDims ? new long[]{1,4} : new long[]{4});
            INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);
            INDArray prod_0 = preReduceInput.prod(0);
            INDArray dLdOut_0 = Nd4j.create(new double[]{1, 2, 3, 4}, reducedShape_0);
            INDArray dLdInExpected_0 = Nd4j.create(3,4);
            for( int i=0; i<3; i++ ){
                dLdInExpected_0.putRow(i, prod_0);
            }
            dLdInExpected_0.divi(preReduceInput);   //Currently: prod(in)/in_i (along dim 0)
            dLdInExpected_0.muliRowVector(dLdOut_0);
            //System.out.println(dLdInExpected_0);
            /*
            [[   45.0000,  120.0000,  231.0000,  384.0000],
             [    9.0000,   40.0000,   99.0000,  192.0000],
             [    5.0000,   24.0000,   63.0000,  128.0000]]
             */

            INDArray dLdIn = Nd4j.createUninitialized(4, 4);

            String err = OpValidation.validate(new OpTestCase(new ProdBp(preReduceInput, dLdOut_0, dLdIn, keepDims, 0))
                    .expectedOutput(0, dLdInExpected_0));
            assertNull(err);


            long[] reducedShape_1 = (keepDims ? new long[]{3,1} : new long[]{3});
            INDArray dLdOut_1 = Nd4j.create(new double[]{1, 2, 3}, reducedShape_1);
            INDArray prod_1 = preReduceInput.prod(1);
            INDArray dLdInExpected_1 = Nd4j.create(3,4);
            for( int i=0; i<4; i++ ){
                dLdInExpected_1.putColumn(i, prod_1);
            }
            dLdInExpected_1.divi(preReduceInput);
            dLdInExpected_1.muliColumnVector(dLdOut_1.reshape(3,1));    //Reshape is a hack around https://github.com/deeplearning4j/deeplearning4j/issues/5530
            //System.out.println(dLdInExpected_1);
            /*
            [[   24.0000,   12.0000,    8.0000,    6.0000],
             [  672.0000,  560.0000,  480.0000,  420.0000],
             [ 3960.0000, 3564.0000, 3240.0000, 2970.0000]]
             */


            dLdIn = Nd4j.createUninitialized(4, 4);
            err = OpValidation.validate(new OpTestCase(new ProdBp(preReduceInput, dLdOut_1, dLdIn, keepDims, 1))
                    .expectedOutput(0, dLdInExpected_1));

            assertNull(err, err);
        }
    }

    @Ignore
    @Test
    public void testStdev(){

        fail();
    }

    @Ignore
    @Test
    public void testVariance(){

        fail();
    }

    @Ignore
    @Test
    public void testCumSum(){

        //dL/dIn_i  = dL/dOut * dOut/dIn_i
        //          = dL/dOut * d(in_0 + ... + in_i)/dIn_i
        //          = dL/dOut

        fail();
    }

    @Ignore
    @Test
    public void testCumProd(){

        //dL/dIn_i  = dL/dOut * dOut/dIn_i
        //          = dL/dOut * d(in_0 * ... * in_i)/dIn_i
        //          = dL/dOut * prod_(j=0..i-1)(in_j)
        //          = dL/dOut * cumProd(in)/in_i
        // (note: edge case for i=0 is dL/dOut * 1

        fail();
    }

}
