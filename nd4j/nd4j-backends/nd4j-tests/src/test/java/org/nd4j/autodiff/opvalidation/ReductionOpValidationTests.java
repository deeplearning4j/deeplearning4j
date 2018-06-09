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

import java.util.Arrays;

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
    public void testReduceSumBP() {
        //Full array reduction

        //reduce_sum_bp op: has 2 inputs (original pre-reduce input, and gradient at output (epsilon))
        //out = sum_j (in_j) -> dL/dIn = dL/dOut * dOut/dIn = dL/dOut

        for (boolean keepDims : new boolean[]{false, true}) {

            INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);
            INDArray dLdOut;
            if (keepDims) {
                dLdOut = Nd4j.valueArrayOf(new long[]{1, 1}, 0.5);
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
    public void testReduceSumAlongDim0BP() {
        //Reduction along dimension
        //Inputs/outputs as before - but note that the output is no longer a scalar

        //Note: when reducing [3,4] along dimension 0 -> 4 TADs of length 3
        //We have one epsilon/gradient for each of the 4 TADs -> dL/dOut length is 4

        for (boolean keepDims : new boolean[]{false, true}) {
            long[] reducedShape_0 = (keepDims ? new long[]{1, 4} : new long[]{4});
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
    public void testReduceSumAlongDim1BP() {
        //Reduction along dimension
        //Inputs/outputs as before - but note that the output is no longer a scalar

        //Note: when reducing [3,4] along dimension 1 -> 3 TADs of length 4
        //We have one epsilon/gradient for each of the 3 TADs -> dL/dOut length is 3

        for (boolean keepDims : new boolean[]{false, true}) {
            INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);

            long[] reducedShape_1 = (keepDims ? new long[]{3, 1} : new long[]{3});
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
    public void testMeanBP() {

        //dL/dIn_i = dL/dOut * dOut/dIn_i = dL/dOut * (1/N * sum_j (in_j))
        //         = 1/N * dL/dOut
        // i.e., same as SUM case but divided by N
        //NOTE: N = num values in array
        //But for "along dimension" case - it's the number of elements in that TAD

        //Full array reduction
        //reduce_mean_bp op: has 2 inputs (original pre-reduce input, and gradient at output (epsilon))

        for (boolean keepDims : new boolean[]{false, true}) {

            INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);
            INDArray dLdOut;
            if (keepDims) {
                dLdOut = Nd4j.valueArrayOf(new long[]{1, 1}, 0.5);
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
    public void testMeanAlongDim0BP() {
        //Reduction along dimension
        //Inputs/outputs as before - but note that the output is no longer a scalar

        //Note: when reducing [3,4] along dimension 0 -> 4 TADs of length 3 -> N=3 -> dL/dIn_i = dL/dOut * 1/3
        //We have one epsilon/gradient for each of the 4 TADs -> dL/dOut length is 4

        for (boolean keepDims : new boolean[]{false, true}) {
            long[] reducedShape_0 = (keepDims ? new long[]{1, 4} : new long[]{4});
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
    public void testMeanAlongDim1BP() {
        //Reduction along dimension
        //Inputs/outputs as before - but note that the output is no longer a scalar

        //Note: when reducing [3,4] along dimension 1 -> 3 TADs of length 4 -> N=4 -> dL/dIn_i = dL/dOut * 1/4
        //We have one epsilon/gradient for each of the 3 TADs -> dL/dOut length is 3

        for (boolean keepDims : new boolean[]{false, true}) {
            INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);

            long[] reducedShape_1 = (keepDims ? new long[]{3, 1} : new long[]{3});
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
    public void testMinBP() {
        //Full array min reduction

        //dL/dIn_i  = dL/dOut * dOut/dIn_i
        //          = dL/dOut                   if in_i == out (== min(in))
        //          = 0                         otherwise

        for (boolean keepDims : new boolean[]{false, true}) {

            INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);
            preReduceInput.putScalar(new int[]{2, 2}, -1);   //Minimum value at position [2,2]
            INDArray dLdOut;
            if (keepDims) {
                dLdOut = Nd4j.valueArrayOf(new long[]{1, 1}, 0.5);
            } else {
                dLdOut = Nd4j.trueScalar(0.5);
            }
            INDArray dLdInExpected = Nd4j.zeros(preReduceInput.shape());
            dLdInExpected.putScalar(new int[]{2, 2}, 0.5);   //Minimum value: position at [2,2]
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
    public void testMinAlongDimensionBP() {
        //Full array min reduction

        //dL/dIn_i  = dL/dOut * dOut/dIn_i
        //          = dL/dOut                   if in_i == out (== min(in))
        //          = 0                         otherwise

        for (boolean keepDims : new boolean[]{false, true}) {

            long[] reducedShape_0 = (keepDims ? new long[]{1, 4} : new long[]{4});
            INDArray preReduceInput = Nd4j.linspace(1, 16, 16).reshape(4, 4);
            preReduceInput.putScalar(0, 0, -1);
            preReduceInput.putScalar(1, 1, -2);
            preReduceInput.putScalar(2, 2, -3);
            preReduceInput.putScalar(2, 2, -4);
            INDArray dLdOut_0 = Nd4j.create(new double[]{1, 2, 3, 4}, reducedShape_0);
            INDArray dLdInExpected_0 = Nd4j.create(preReduceInput.shape()); //All 0s except along diagonal
            dLdInExpected_0.putScalar(0, 0, 1);
            dLdInExpected_0.putScalar(1, 1, 2);
            dLdInExpected_0.putScalar(2, 2, 3);
            dLdInExpected_0.putScalar(3, 3, 4);

            INDArray dLdIn = Nd4j.createUninitialized(4, 4);

            String err = OpValidation.validate(new OpTestCase(new MinBp(preReduceInput, dLdOut_0, dLdIn, keepDims, 0))
                    .expectedOutput(0, dLdInExpected_0));
            assertNull(err, err);


            long[] reducedShape_1 = (keepDims ? new long[]{4, 1} : new long[]{4});
            INDArray dLdInExpected_1 = dLdInExpected_0; //Same here, only because the maximums are along the diagonal

            INDArray dLdOut_1 = Nd4j.create(new double[]{1, 2, 3, 4}, reducedShape_1);
            dLdIn = Nd4j.createUninitialized(4, 4);

            err = OpValidation.validate(new OpTestCase(new MinBp(preReduceInput, dLdOut_1, dLdIn, keepDims, 1))
                    .expectedOutput(0, dLdInExpected_1));

            assertNull(err, err);
        }
    }

    @Test
    public void testMaxBP() {
        //Full array max reduction

        //dL/dIn_i  = dL/dOut * dOut/dIn_i
        //          = dL/dOut                   if in_i == out (== max(in))
        //          = 0                         otherwise

        for (boolean keepDims : new boolean[]{false, true}) {

            INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);
            preReduceInput.putScalar(new int[]{2, 2}, 20);   //Maximum value at position [2,2]
            INDArray dLdOut;
            if (keepDims) {
                dLdOut = Nd4j.valueArrayOf(new long[]{1, 1}, 0.5);
            } else {
                dLdOut = Nd4j.trueScalar(0.5);
            }
            INDArray dLdInExpected = Nd4j.zeros(preReduceInput.shape());
            dLdInExpected.putScalar(new int[]{2, 2}, 0.5);   //Maximum value: position at [2,2]
            INDArray dLdIn = Nd4j.createUninitialized(3, 4);

            String err = OpValidation.validate(new OpTestCase(new MaxBp(preReduceInput, dLdOut, dLdIn, keepDims))
                    .expectedOutput(0, dLdInExpected));

            assertNull(err);
        }

    }

    @Test
    public void testMaxAlongDimensionBP() {
        //Full array min reduction

        //dL/dIn_i  = dL/dOut * dOut/dIn_i
        //          = dL/dOut                   if in_i == out (== min(in))
        //          = 0                         otherwise

        for (boolean keepDims : new boolean[]{false, true}) {

            long[] reducedShape_0 = (keepDims ? new long[]{1, 4} : new long[]{4});
            INDArray preReduceInput = Nd4j.linspace(1, 16, 16).reshape(4, 4);
            preReduceInput.putScalar(0, 0, 20);
            preReduceInput.putScalar(1, 1, 21);
            preReduceInput.putScalar(2, 2, 22);
            preReduceInput.putScalar(2, 2, 23);
            INDArray dLdOut_0 = Nd4j.create(new double[]{1, 2, 3, 4}, reducedShape_0);
            INDArray dLdInExpected_0 = Nd4j.create(preReduceInput.shape());
            dLdInExpected_0.putScalar(0, 0, 1);
            dLdInExpected_0.putScalar(1, 1, 2);
            dLdInExpected_0.putScalar(2, 2, 3);
            dLdInExpected_0.putScalar(3, 3, 4);

            INDArray dLdIn = Nd4j.createUninitialized(4, 4);

            String err = OpValidation.validate(new OpTestCase(new MaxBp(preReduceInput, dLdOut_0, dLdIn, keepDims, 0))
                    .expectedOutput(0, dLdInExpected_0));
            assertNull(err, err);


            long[] reducedShape_1 = (keepDims ? new long[]{4, 1} : new long[]{4});
            INDArray dLdInExpected_1 = dLdInExpected_0; //Same here, only because the maximums are along the diagonal

            INDArray dLdOut_1 = Nd4j.create(new double[]{1, 2, 3, 4}, reducedShape_1);
            dLdIn = Nd4j.createUninitialized(4, 4);

            err = OpValidation.validate(new OpTestCase(new MaxBp(preReduceInput, dLdOut_1, dLdIn, keepDims, 1))
                    .expectedOutput(0, dLdInExpected_1));

            assertNull(err, err);
        }
    }

    @Test
    public void testProdBP() {
        //Full array product reduction

        //dL/dIn_i  = dL/dOut * dOut/dIn_i
        //          = dL/dOut * d(prod(in))/dIn_i
        //          = dL/dOut * (prod(in) / in_i)

        for (boolean keepDims : new boolean[]{false, true}) {

            INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);
            INDArray dLdOut;
            if (keepDims) {
                dLdOut = Nd4j.valueArrayOf(new long[]{1, 1}, 0.5);
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
    public void testProdAlongDimensionBP() {
        //dL/dIn_i  = dL/dOut * dOut/dIn_i
        //          = dL/dOut * d(prod(in))/dIn_i
        //          = dL/dOut * (prod(in) / in_i)

        for (boolean keepDims : new boolean[]{false, true}) {
            long[] reducedShape_0 = (keepDims ? new long[]{1, 4} : new long[]{4});
            INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);
            INDArray prod_0 = preReduceInput.prod(0);
            INDArray dLdOut_0 = Nd4j.create(new double[]{1, 2, 3, 4}, reducedShape_0);
            INDArray dLdInExpected_0 = Nd4j.create(3, 4);
            for (int i = 0; i < 3; i++) {
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


            long[] reducedShape_1 = (keepDims ? new long[]{3, 1} : new long[]{3});
            INDArray dLdOut_1 = Nd4j.create(new double[]{1, 2, 3}, reducedShape_1);
            INDArray prod_1 = preReduceInput.prod(1);
            INDArray dLdInExpected_1 = Nd4j.create(3, 4);
            for (int i = 0; i < 4; i++) {
                dLdInExpected_1.putColumn(i, prod_1);
            }
            dLdInExpected_1.divi(preReduceInput);
            dLdInExpected_1.muliColumnVector(dLdOut_1.reshape(3, 1));    //Reshape is a hack around https://github.com/deeplearning4j/deeplearning4j/issues/5530
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
    public void testStdevBP() {
        //If out = stdev(in) then:
        //dL/dIn = dL/dOut * dOut/dIn
        //dOut/dIn_i = (in_i-mean)/(stdev * (n-1))
        //OR: n instead of n-1, if not bias corrected

        for (boolean biasCorrected : new boolean[]{true, false}) {
            for (boolean keepDims : new boolean[]{false, true}) {

                INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);
                INDArray dLdOut;
                if (keepDims) {
                    dLdOut = Nd4j.valueArrayOf(new long[]{1, 1}, 0.5);
                } else {
                    dLdOut = Nd4j.trueScalar(0.5);
                }

                double stdev = preReduceInput.stdNumber(biasCorrected).doubleValue();
                double mean = preReduceInput.meanNumber().doubleValue();

                long divisor = biasCorrected ? (preReduceInput.length() - 1) : preReduceInput.length();

                INDArray dLdInExp = preReduceInput.dup()
                        .subi(mean).divi(stdev * divisor)
                        .muli(0.5); //* dL/dOut
//                System.out.println("biasCorrected = " + biasCorrected + ", keepDims=" + keepDims);
//                System.out.println(dLdInExp.shapeInfoToString());
//                System.out.println(Arrays.toString(dLdInExp.data().asFloat()));
                /*
                biasCorrected = true, keepDims=false
                Rank: 2,Offset: 0
                 Order: c Shape: [3,4],  stride: [4,1]
                [-0.069337524, -0.056730703, -0.04412388, -0.031517055, -0.018910235, -0.0063034114, 0.0063034114, 0.018910235, 0.031517055, 0.04412388, 0.056730703, 0.069337524]
                biasCorrected = true, keepDims=true
                Rank: 2,Offset: 0
                 Order: c Shape: [3,4],  stride: [4,1]
                [-0.069337524, -0.056730703, -0.04412388, -0.031517055, -0.018910235, -0.0063034114, 0.0063034114, 0.018910235, 0.031517055, 0.04412388, 0.056730703, 0.069337524]
                biasCorrected = false, keepDims=false
                Rank: 2,Offset: 0
                 Order: c Shape: [3,4],  stride: [4,1]
                [-0.06638563, -0.05431551, -0.0422454, -0.030175284, -0.01810517, -0.006035057, 0.006035057, 0.01810517, 0.030175284, 0.0422454, 0.05431551, 0.06638563]
                biasCorrected = false, keepDims=true
                Rank: 2,Offset: 0
                 Order: c Shape: [3,4],  stride: [4,1]
                [-0.06638563, -0.05431551, -0.0422454, -0.030175284, -0.01810517, -0.006035057, 0.006035057, 0.01810517, 0.030175284, 0.0422454, 0.05431551, 0.06638563]
                 */

                INDArray dLdIn = Nd4j.createUninitialized(3, 4);

                String err = OpValidation.validate(new OpTestCase(new StandardDeviationBp(preReduceInput, dLdOut, dLdIn, biasCorrected, keepDims))
                        .expectedOutput(0, dLdInExp));
                assertNull(err);
            }
        }
    }

    @Test
    public void testStdevAlongDimensionBP() {
        //If out = stdev(in) then:
        //dL/dIn = dL/dOut * dOut/dIn
        //dOut/dIn_i = (in_i-mean)/(stdev * (n-1))
        //OR: n instead of n-1, if not bias corrected

        for (boolean biasCorrected : new boolean[]{false, true}) {
            for (boolean keepDims : new boolean[]{false, true}) {
                long[] reducedShape_0 = (keepDims ? new long[]{1, 4} : new long[]{4});
                INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);
                long divisor = biasCorrected ? 2 : 3;
                INDArray mean_0 = preReduceInput.mean(0);
                INDArray stdev_0 = preReduceInput.std(biasCorrected, 0);
                INDArray dLdOut_0 = Nd4j.create(new double[]{1, 2, 3, 4}, reducedShape_0);

                INDArray dLdInExpected_0 = preReduceInput.dup();
                dLdInExpected_0.subiRowVector(mean_0)
                        .diviRowVector(stdev_0.mul(divisor))
                        .muliRowVector(dLdOut_0);
//                System.out.println("biasCorrected = " + biasCorrected + ", keepDims=" + keepDims);
//                System.out.println(dLdInExpected_0.shapeInfoToString());
//                System.out.println(Arrays.toString(dLdInExpected_0.data().asFloat()));
                /*
                biasCorrected = false, keepDims=false
                Rank: 2,Offset: 0
                 Order: c Shape: [3,4],  stride: [4,1]
                [-0.4082483, -0.8164966, -1.2247449, -1.6329932, 0.0, 0.0, 0.0, 0.0, 0.4082483, 0.8164966, 1.2247449, 1.6329932]
                biasCorrected = false, keepDims=true
                Rank: 2,Offset: 0
                 Order: c Shape: [3,4],  stride: [4,1]
                [-0.4082483, -0.8164966, -1.2247449, -1.6329932, 0.0, 0.0, 0.0, 0.0, 0.4082483, 0.8164966, 1.2247449, 1.6329932]
                biasCorrected = true, keepDims=false
                Rank: 2,Offset: 0
                 Order: c Shape: [3,4],  stride: [4,1]
                [-0.5, -1.0, -1.5, -2.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0]
                biasCorrected = true, keepDims=true
                Rank: 2,Offset: 0
                 Order: c Shape: [3,4],  stride: [4,1]
                [-0.5, -1.0, -1.5, -2.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0]
                 */

                INDArray dLdIn = Nd4j.createUninitialized(3, 4);
                String err = OpValidation.validate(new OpTestCase(new StandardDeviationBp(preReduceInput, dLdOut_0, dLdIn, biasCorrected, keepDims, 0))
                        .expectedOutput(0, dLdInExpected_0));
                assertNull(err);


                divisor = biasCorrected ? 3 : 4;
                long[] reducedShape_1 = (keepDims ? new long[]{3, 1} : new long[]{3});
                INDArray dLdOut_1 = Nd4j.create(new double[]{1, 2, 3}, reducedShape_1);
                INDArray mean_1 = preReduceInput.mean(1);
                INDArray stdev_1 = preReduceInput.std(biasCorrected, 1);
                INDArray dLdInExpected_1 = preReduceInput.dup();
                dLdInExpected_1.subiColumnVector(mean_1)
                        .diviColumnVector(stdev_1.mul(divisor))
                        .muliColumnVector(dLdOut_1.reshape(3,1));
//                System.out.println("biasCorrected = " + biasCorrected + ", keepDims=" + keepDims);
//                System.out.println(dLdInExpected_1.shapeInfoToString());
//                System.out.println(Arrays.toString(dLdInExpected_1.data().asFloat()));
                /*
                biasCorrected = false, keepDims=false
                Rank: 2,Offset: 0
                 Order: c Shape: [3,4],  stride: [4,1]
                [-0.3354102, -0.1118034, 0.1118034, 0.3354102, -0.6708204, -0.2236068, 0.2236068, 0.6708204, -1.0062306, -0.3354102, 0.3354102, 1.0062306]
                biasCorrected = false, keepDims=true
                Rank: 2,Offset: 0
                 Order: c Shape: [3,4],  stride: [4,1]
                [-0.3354102, -0.1118034, 0.1118034, 0.3354102, -0.6708204, -0.2236068, 0.2236068, 0.6708204, -1.0062306, -0.3354102, 0.3354102, 1.0062306]
                biasCorrected = true, keepDims=false
                Rank: 2,Offset: 0
                 Order: c Shape: [3,4],  stride: [4,1]
                [-0.38729835, -0.12909944, 0.12909944, 0.38729835, -0.7745967, -0.2581989, 0.2581989, 0.7745967, -1.161895, -0.38729835, 0.38729835, 1.161895]
                biasCorrected = true, keepDims=true
                Rank: 2,Offset: 0
                 Order: c Shape: [3,4],  stride: [4,1]
                [-0.38729835, -0.12909944, 0.12909944, 0.38729835, -0.7745967, -0.2581989, 0.2581989, 0.7745967, -1.161895, -0.38729835, 0.38729835, 1.161895]
                 */


                dLdIn = Nd4j.createUninitialized(3, 4);
                err = OpValidation.validate(new OpTestCase(new ProdBp(preReduceInput, dLdOut_1, dLdIn, keepDims, 1))
                        .expectedOutput(0, dLdInExpected_1));
                assertNull(err, err);
            }
        }
    }

    @Ignore
    @Test
    public void testVarianceBP() {
        //If out = variance(in) then:
        //dL/dIn = dL/dOut * dOut/dIn
        //dOut/dIn_i = 2*(in_i-mean)/(n-1)
        //OR: n instead of n-1, if not bias corrected

        for (boolean biasCorrected : new boolean[]{true, false}) {
            for (boolean keepDims : new boolean[]{false, true}) {

                INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);
                INDArray dLdOut;
                if (keepDims) {
                    dLdOut = Nd4j.valueArrayOf(new long[]{1, 1}, 0.5);
                } else {
                    dLdOut = Nd4j.trueScalar(0.5);
                }

                double var = preReduceInput.var(biasCorrected).getDouble(0);
                double mean = preReduceInput.meanNumber().doubleValue();

                long divisor = biasCorrected ? (preReduceInput.length() - 1) : preReduceInput.length();

                INDArray dLdInExp = preReduceInput.dup()
                        .subi(mean).muli(2.0 / divisor)
                        .muli(0.5); //* dL/dOut
//                System.out.println("biasCorrected = " + biasCorrected + ", keepDims=" + keepDims);
//                System.out.println(dLdInExp.shapeInfoToString());
//                System.out.println(Arrays.toString(dLdInExp.data().asFloat()));
                /*
                biasCorrected = true, keepDims=false
                Rank: 2,Offset: 0
                 Order: c Shape: [3,4],  stride: [4,1]
                [-0.5, -0.4090909, -0.3181818, -0.22727273, -0.13636364, -0.045454547, 0.045454547, 0.13636364, 0.22727273, 0.3181818, 0.4090909, 0.5]
                biasCorrected = true, keepDims=true
                Rank: 2,Offset: 0
                 Order: c Shape: [3,4],  stride: [4,1]
                [-0.5, -0.4090909, -0.3181818, -0.22727273, -0.13636364, -0.045454547, 0.045454547, 0.13636364, 0.22727273, 0.3181818, 0.4090909, 0.5]
                biasCorrected = false, keepDims=false
                Rank: 2,Offset: 0
                 Order: c Shape: [3,4],  stride: [4,1]
                [-0.45833334, -0.375, -0.29166666, -0.20833333, -0.125, -0.041666668, 0.041666668, 0.125, 0.20833333, 0.29166666, 0.375, 0.45833334]
                biasCorrected = false, keepDims=true
                Rank: 2,Offset: 0
                 Order: c Shape: [3,4],  stride: [4,1]
                [-0.45833334, -0.375, -0.29166666, -0.20833333, -0.125, -0.041666668, 0.041666668, 0.125, 0.20833333, 0.29166666, 0.375, 0.45833334]
                 */

                INDArray dLdIn = Nd4j.createUninitialized(3, 4);

                String err = OpValidation.validate(new OpTestCase(new VarianceBp(preReduceInput, dLdOut, dLdIn, biasCorrected, keepDims))
                        .expectedOutput(0, dLdInExp));
                assertNull(err);
            }
        }
    }

    @Test
    public void testVarianceAlongDimensionBP() {
        //If out = variance(in) then:
        //dL/dIn = dL/dOut * dOut/dIn
        //dOut/dIn_i = 2*(in_i-mean)/(n-1)
        //OR: n instead of n-1, if not bias corrected

        for (boolean biasCorrected : new boolean[]{false, true}) {
            for (boolean keepDims : new boolean[]{false, true}) {
                long[] reducedShape_0 = (keepDims ? new long[]{1, 4} : new long[]{4});
                INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);
                long divisor = biasCorrected ? 2 : 3;
                INDArray mean_0 = preReduceInput.mean(0);
                INDArray dLdOut_0 = Nd4j.create(new double[]{1, 2, 3, 4}, reducedShape_0);

                INDArray dLdInExpected_0 = preReduceInput.dup();
                dLdInExpected_0.subiRowVector(mean_0).muli(2.0 / divisor)
                        .muliRowVector(dLdOut_0);
//                System.out.println("biasCorrected = " + biasCorrected + ", keepDims=" + keepDims);
//                System.out.println(dLdInExpected_0.shapeInfoToString());
//                System.out.println(Arrays.toString(dLdInExpected_0.data().asFloat()));
                /*
                biasCorrected = false, keepDims=false
                Rank: 2,Offset: 0
                 Order: c Shape: [3,4],  stride: [4,1]
                [-2.6666667, -5.3333335, -8.0, -10.666667, 0.0, 0.0, 0.0, 0.0, 2.6666667, 5.3333335, 8.0, 10.666667]
                biasCorrected = false, keepDims=true
                Rank: 2,Offset: 0
                 Order: c Shape: [3,4],  stride: [4,1]
                [-2.6666667, -5.3333335, -8.0, -10.666667, 0.0, 0.0, 0.0, 0.0, 2.6666667, 5.3333335, 8.0, 10.666667]
                biasCorrected = true, keepDims=false
                Rank: 2,Offset: 0
                 Order: c Shape: [3,4],  stride: [4,1]
                [-4.0, -8.0, -12.0, -16.0, 0.0, 0.0, 0.0, 0.0, 4.0, 8.0, 12.0, 16.0]
                biasCorrected = true, keepDims=true
                Rank: 2,Offset: 0
                 Order: c Shape: [3,4],  stride: [4,1]
                [-4.0, -8.0, -12.0, -16.0, 0.0, 0.0, 0.0, 0.0, 4.0, 8.0, 12.0, 16.0]
                 */

//                INDArray dLdIn = Nd4j.createUninitialized(3, 4);
//                String err = OpValidation.validate(new OpTestCase(new StandardDeviationBp(preReduceInput, dLdOut_0, dLdIn, biasCorrected, keepDims, 0))
//                        .expectedOutput(0, dLdInExpected_0));
//                assertNull(err);


                divisor = biasCorrected ? 3 : 4;
                long[] reducedShape_1 = (keepDims ? new long[]{3, 1} : new long[]{3});
                INDArray dLdOut_1 = Nd4j.create(new double[]{1, 2, 3}, reducedShape_1);
                INDArray mean_1 = preReduceInput.mean(1);
                INDArray dLdInExpected_1 = preReduceInput.dup();
                dLdInExpected_1.subiColumnVector(mean_1).muli(2.0 / divisor)
                        .muliColumnVector(dLdOut_1.reshape(3,1));
//                System.out.println("biasCorrected = " + biasCorrected + ", keepDims=" + keepDims);
//                System.out.println(dLdInExpected_1.shapeInfoToString());
//                System.out.println(Arrays.toString(dLdInExpected_1.data().asFloat()));
                /*
                biasCorrected = false, keepDims=false
                Rank: 2,Offset: 0
                 Order: c Shape: [3,4],  stride: [4,1]
                [-0.75, -0.25, 0.25, 0.75, -1.5, -0.5, 0.5, 1.5, -2.25, -0.75, 0.75, 2.25]
                biasCorrected = false, keepDims=true
                Rank: 2,Offset: 0
                 Order: c Shape: [3,4],  stride: [4,1]
                [-0.75, -0.25, 0.25, 0.75, -1.5, -0.5, 0.5, 1.5, -2.25, -0.75, 0.75, 2.25]
                biasCorrected = true, keepDims=false
                Rank: 2,Offset: 0
                 Order: c Shape: [3,4],  stride: [4,1]
                [-1.0, -0.33333334, 0.33333334, 1.0, -2.0, -0.6666667, 0.6666667, 2.0, -3.0, -1.0, 1.0, 3.0]
                biasCorrected = true, keepDims=true
                Rank: 2,Offset: 0
                 Order: c Shape: [3,4],  stride: [4,1]
                [-1.0, -0.33333334, 0.33333334, 1.0, -2.0, -0.6666667, 0.6666667, 2.0, -3.0, -1.0, 1.0, 3.0]
                 */


//                dLdIn = Nd4j.createUninitialized(3, 4);
//                err = OpValidation.validate(new OpTestCase(new ProdBp(preReduceInput, dLdOut_1, dLdIn, keepDims, 1))
//                        .expectedOutput(0, dLdInExpected_1));
//                assertNull(err, err);
            }
        }
    }


    @Ignore
    @Test
    public void testCumSumBP() {
        //CumSum is not *technically* a reduction...

        //dL/dIn_i  = dL/dOut * dOut/dIn_i
        //          = dL/dOut * d(in_0 + ... + in_i)/dIn_i
        //          = dL/dOut



//        INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);
//        INDArray dLdOut = preReduceInput.dup().addi(100);
//        INDArray dLdInExpected = Nd4j.valueArrayOf(preReduceInput.shape(), 0.5);
//        INDArray dLdIn = Nd4j.createUninitialized(3, 4);
//
//        String err = OpValidation.validate(new OpTestCase(new CumSumBp(preReduceInput, dLdOut, dLdIn, keepDims))
//                .expectedOutput(0, dLdInExpected));

//        assertNull(err);
    }

    @Ignore
    @Test
    public void testCumProdBP() {

        //dL/dIn_i  = dL/dOut * dOut/dIn_i
        //          = dL/dOut * d(in_0 * ... * in_i)/dIn_i
        //          = dL/dOut * prod_(j=0..i-1)(in_j)
        //          = dL/dOut * cumProd(in)/in_i
        // (note: edge case for i=0 is dL/dOut * 1

        fail();
    }

}
