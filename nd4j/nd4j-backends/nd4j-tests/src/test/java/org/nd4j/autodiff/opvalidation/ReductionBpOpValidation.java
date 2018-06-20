package org.nd4j.autodiff.opvalidation;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.autodiff.OpValidationSuite;
import org.nd4j.autodiff.validation.OpTestCase;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.bp.*;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.Arrays;

import static org.junit.Assert.assertNull;
import static org.junit.Assert.fail;

public class ReductionBpOpValidation extends BaseOpValidation {

    private DataBuffer.Type initialType;

    public ReductionBpOpValidation(Nd4jBackend backend) {
        super(backend);
    }

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
        OpValidationSuite.ignoreFailing();
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
    public void testMeanBP_Rank1() {
        INDArray dLdOut = Nd4j.trueScalar(0.5);
        INDArray preReduceInput = Nd4j.create(new double[]{2,3,4}, new long[]{3});
        INDArray dLdInExp = Nd4j.valueArrayOf(new long[]{3}, 0.5/3);

        INDArray dLdIn = Nd4j.createUninitialized(new long[]{3});

        String err = OpValidation.validate(new OpTestCase(new MeanBp(preReduceInput, dLdOut, dLdIn, false))
                .expectedOutput(0, dLdInExp));
        assertNull(err);
    }

    @Test
    public void testMeanAlongDim0BP() {
        OpValidationSuite.ignoreFailing();
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
            String err = OpValidation.validate(new OpTestCase(new MeanBp(preReduceInput, dLdOut_0, dLdIn, keepDims, 0))
                    .expectedOutput(0, dLdInExpected_0));

            assertNull(err);
        }
    }

    @Test
    public void testMeanAlongDim1BP() {
        OpValidationSuite.ignoreFailing();
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
        OpValidationSuite.ignoreFailing();
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
        OpValidationSuite.ignoreFailing();
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
        OpValidationSuite.ignoreFailing();
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
        OpValidationSuite.ignoreFailing();
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
        OpValidationSuite.ignoreFailing();
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

            INDArray dLdIn = Nd4j.createUninitialized(3, 4);

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


            dLdIn = Nd4j.createUninitialized(3, 4);
            err = OpValidation.validate(new OpTestCase(new ProdBp(preReduceInput, dLdOut_1, dLdIn, keepDims, 1))
                    .expectedOutput(0, dLdInExpected_1));

            assertNull(err, err);
        }
    }

    @Test
    public void testStdevBP() {
        OpValidationSuite.ignoreFailing();

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

                INDArray dLdIn = Nd4j.createUninitialized(3, 4);

                String err = OpValidation.validate(new OpTestCase(new StandardDeviationBp(preReduceInput, dLdOut, dLdIn, biasCorrected, keepDims))
                        .expectedOutput(0, dLdInExp));
                assertNull(err);
            }
        }
    }

    @Test
    public void testStdevBP_Rank1() {
        OpValidationSuite.ignoreFailing();
        //fail(); //https://github.com/deeplearning4j/deeplearning4j/issues/5582
        INDArray dLdOut = Nd4j.trueScalar(0.5);
        INDArray preReduceInput = Nd4j.create(new double[]{2,3,4}, new long[]{3});
        double stdev = preReduceInput.stdNumber(true).doubleValue();
        double mean = preReduceInput.meanNumber().doubleValue();

        INDArray dLdInExp = preReduceInput.dup()
                .subi(mean).divi(stdev * 2)
                .muli(0.5); //* dL/dOut

        System.out.println(dLdInExp.shapeInfoToString());
        System.out.println(Arrays.toString(dLdInExp.data().asFloat()));

        INDArray dLdIn = Nd4j.createUninitialized(new long[]{3});

        String err = OpValidation.validate(new OpTestCase(new StandardDeviationBp(preReduceInput, dLdOut, dLdIn, true, false))
                .expectedOutput(0, dLdInExp));
        assertNull(err);
    }

    @Test
    public void testStdevAlongDimensionBP() {
        OpValidationSuite.ignoreFailing();
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

                dLdIn = Nd4j.createUninitialized(3, 4);
                err = OpValidation.validate(new OpTestCase(new ProdBp(preReduceInput, dLdOut_1, dLdIn, keepDims, 1))
                        .expectedOutput(0, dLdInExpected_1));
                assertNull(err, err);
            }
        }
    }

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

                INDArray dLdIn = Nd4j.createUninitialized(3, 4);

                String err = OpValidation.validate(new OpTestCase(new VarianceBp(preReduceInput, dLdOut, dLdIn, biasCorrected, keepDims))
                        .expectedOutput(0, dLdInExp));
                assertNull(err);
            }
        }
    }

    @Test
    public void testVarianceAlongDimensionBP() {
        OpValidationSuite.ignoreFailing();
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

                INDArray dLdIn = Nd4j.createUninitialized(3, 4);
                String err = OpValidation.validate(new OpTestCase(new VarianceBp(preReduceInput, dLdOut_0, dLdIn, biasCorrected, keepDims, 0))
                        .expectedOutput(0, dLdInExpected_0));
                assertNull(err);

                divisor = biasCorrected ? 3 : 4;
                long[] reducedShape_1 = (keepDims ? new long[]{3, 1} : new long[]{3});
                INDArray dLdOut_1 = Nd4j.create(new double[]{1, 2, 3}, reducedShape_1);
                INDArray mean_1 = preReduceInput.mean(1);
                INDArray dLdInExpected_1 = preReduceInput.dup();
                dLdInExpected_1.subiColumnVector(mean_1).muli(2.0 / divisor)
                        .muliColumnVector(dLdOut_1.reshape(3,1));


                dLdIn = Nd4j.createUninitialized(3, 4);
                err = OpValidation.validate(new OpTestCase(new VarianceBp(preReduceInput, dLdOut_1, dLdIn, biasCorrected, keepDims, 1))
                        .expectedOutput(0, dLdInExpected_1));
                assertNull(err);
            }
        }
    }


    @Test
    public void testCumSumBP() {
        OpValidationSuite.ignoreFailing();
        //CumSum is not *technically* a reduction...

        //Standard case, non-reverse, non-exclusive
        //dL/dIn_i  = sum_j dL/dOut_j * dOut_j/dIn_i
        //          = sum_j dL/dOut_j * d(in_0 + ... + in_j)/dIn_i
        //          = reverseCumSum(dL/dOut_j)

        //Reverse case:
        //dL/dIn_i  = sum_j dL/dOut_j * dOut_j/dIn_i
        //          = sum_j dL/dOut_j * d(in_N + ... + in_j)/dIn_i
        //          = cumSum(dL/dOut_j)

        //Exclusive case:
        //dL/dIn_i  = sum_j dL/dOut_j * dOut_j/dIn_i
        //          = sum_j dL/dOut_j * d(in_0 + ... + in_{i-1})/dIn_i
        //          = reverseCumSumExclusive(dL/dOut_j)

        //Reverse exclusive case
        //dL/dIn_i  = sum_j dL/dOut_j * dOut_j/dIn_i
        //          = sum_j dL/dOut_j * d(in_N + ... + in_j)/dIn_i
        //          = cumSumExclusive(dL/dOut_j)



//        for(boolean exclusive : new boolean[]{false, true}) {
//            for(boolean reverse : new boolean[]{false, true}) {
//
//                INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);
//                INDArray dLdOut = preReduceInput.dup().addi(100);
//                INDArray dLdInExpected = Nd4j.valueArrayOf(preReduceInput.shape(), 0.5);
//                INDArray dLdIn = Nd4j.createUninitialized(3, 4);
//
//                String err = OpValidation.validate(new OpTestCase(new CumSumBp(preReduceInput, dLdOut, dLdIn, keepDims))
//                        .expectedOutput(0, dLdInExpected));
//                assertNull(err);
//            }
//        }
    }


    @Test
    public void testCumProdBP() {
        OpValidationSuite.ignoreFailing();

        //Standard case: non-reverse, non-exclusive
        //dL/dIn_i  = sum_j dL/dOut_j * dOut_j/dIn_i
        //          = sum_j dL/dOut_j * d(in_0 * ... * in_j)/dIn_i
        //          = sum_j dL/dOut_j * prod_(k=0..j)(in_j)
        //          = reverseCumSum( dL/dOut * cumProd(in)/in_i )

        //Reverse case:
        //dL/dIn_i  = sum_j dL/dOut_j * dOut_j/dIn_i
        //          = sum_j dL/dOut_j * d(in_N * ... * in_j)/dIn_i
        //          = sum_j dL/dOut_j * prod_(k=N..j)(in_j)
        //          = cumSum( dL/dOut * reverseCumProd(in)/in_i )

        //Exclusive case
        //


        fail();
    }


    @Test
    public void testNorm2Bp(){
        OpValidationSuite.ignoreFailing();
        //dL/dIn = dL/dOut * dOut/dIn
        //       = dL/dOut * x/|x|_2

        for (boolean keepDims : new boolean[]{false, true}) {

            INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);

            double norm2 = preReduceInput.norm2Number().doubleValue();

            INDArray dLdOut;
            if (keepDims) {
                dLdOut = Nd4j.valueArrayOf(new long[]{1, 1}, 0.5);
            } else {
                dLdOut = Nd4j.trueScalar(0.5);
            }
            INDArray dLdInExpected = preReduceInput.div(norm2).muli(0.5);
            INDArray dLdIn = Nd4j.createUninitialized(3, 4);

            String err = OpValidation.validate(new OpTestCase(new Norm2Bp(preReduceInput, dLdOut, dLdIn, keepDims))
                    .expectedOutput(0, dLdInExpected));

            assertNull(err);
        }
    }

    @Test
    public void testNorm2AlongDimensionBP() {
        OpValidationSuite.ignoreFailing();
        //dL/dIn = dL/dOut * dOut/dIn
        //       = dL/dOut * x/|x|_2

        for (boolean keepDims : new boolean[]{false, true}) {

            long[] reducedShape_0 = (keepDims ? new long[]{3, 4} : new long[]{4});
            INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3, 4);
            INDArray norm2_0 = preReduceInput.norm2(0);
            INDArray dLdOut_0 = Nd4j.create(new double[]{1, 2, 3, 4}, reducedShape_0);
            INDArray dLdInExpected_0 = preReduceInput.divRowVector(norm2_0).mulRowVector(dLdOut_0);

            INDArray dLdIn = Nd4j.createUninitialized(3, 4);

            String err = OpValidation.validate(new OpTestCase(new MinBp(preReduceInput, dLdOut_0, dLdIn, keepDims, 0))
                    .expectedOutput(0, dLdInExpected_0));
            assertNull(err);


            long[] reducedShape_1 = (keepDims ? new long[]{3, 1} : new long[]{3});
            INDArray norm2_1 = preReduceInput.norm2(1);
            INDArray dLdOut_1 = Nd4j.create(new double[]{1, 2, 3, 4}, reducedShape_1);
            INDArray dLdInExpected_1 = preReduceInput.divColumnVector(norm2_1).mulColumnVector(dLdOut_1);
            dLdIn = Nd4j.createUninitialized(3, 4);

            err = OpValidation.validate(new OpTestCase(new Norm2Bp(preReduceInput, dLdOut_1, dLdIn, keepDims, 1))
                    .expectedOutput(0, dLdInExpected_1));

            assertNull(err);
        }
    }

    @Test
    public void testNorm1Bp(){
        OpValidationSuite.ignoreFailing();
        //dL/dIn = dL/dOut * dOut/dIn
        //       = dL/dOut * sgn(in)

        for (boolean keepDims : new boolean[]{false, true}) {

            INDArray preReduceInput = Nd4j.linspace(-5, 6, 12).addi(0.1).reshape(3, 4);

            INDArray sgn = Transforms.sign(preReduceInput, true);

            INDArray dLdOut;
            if (keepDims) {
                dLdOut = Nd4j.valueArrayOf(new long[]{1, 1}, 0.5);
            } else {
                dLdOut = Nd4j.trueScalar(0.5);
            }
            INDArray dLdInExpected = sgn.muli(0.5);
            INDArray dLdIn = Nd4j.createUninitialized(3, 4);

            String err = OpValidation.validate(new OpTestCase(new Norm1Bp(preReduceInput, dLdOut, dLdIn, keepDims))
                    .expectedOutput(0, dLdInExpected));

            assertNull(err);
        }
    }

    @Test
    public void testNorm1AlongDimensionBP() {
        OpValidationSuite.ignoreFailing();
        //dL/dIn = dL/dOut * dOut/dIn
        //       = dL/dOut * sgn(in)

        for (boolean keepDims : new boolean[]{false, true}) {

            long[] reducedShape_0 = (keepDims ? new long[]{3, 4} : new long[]{4});
            INDArray preReduceInput = Nd4j.linspace(-5, 6, 12).reshape(3, 4);
            INDArray sgn = Transforms.sign(preReduceInput, true);
            INDArray dLdOut_0 = Nd4j.create(new double[]{1, 2, 3, 4}, reducedShape_0);
            INDArray dLdInExpected_0 = sgn.mulRowVector(dLdOut_0);

            INDArray dLdIn = Nd4j.createUninitialized(4, 4);

            String err = OpValidation.validate(new OpTestCase(new Norm1Bp(preReduceInput, dLdOut_0, dLdIn, keepDims, 0))
                    .expectedOutput(0, dLdInExpected_0));
            assertNull(err, err);


            long[] reducedShape_1 = (keepDims ? new long[]{3, 1} : new long[]{3});
            INDArray dLdOut_1 = Nd4j.create(new double[]{1, 2, 3, 4}, reducedShape_1);
            INDArray dLdInExpected_1 = sgn.mulColumnVector(dLdOut_1);
            dLdIn = Nd4j.createUninitialized(3, 4);

            err = OpValidation.validate(new OpTestCase(new Norm1Bp(preReduceInput, dLdOut_1, dLdIn, keepDims, 1))
                    .expectedOutput(0, dLdInExpected_1));

            assertNull(err, err);
        }
    }

    @Test
    public void testNormMaxBp(){
        OpValidationSuite.ignoreFailing();
        //out = max_i (|in_i|)
        //dL/dIn = dL/dOut * dOut/dIn
        //       = dL/dOut * (0 if |x_i| is not max; or sgn(x_i) otherwise)

        for (boolean keepDims : new boolean[]{false, true}) {

            INDArray preReduceInput = Nd4j.linspace(-5, 6, 12).reshape(3, 4);

            INDArray sgn = Transforms.sign(preReduceInput, true);
            INDArray max = Nd4j.create(3,4);
            max.putScalar(2,3,1.0);

            INDArray dLdOut;
            if (keepDims) {
                dLdOut = Nd4j.valueArrayOf(new long[]{1, 1}, 0.5);
            } else {
                dLdOut = Nd4j.trueScalar(0.5);
            }
            INDArray dLdInExpected = sgn.mul(max).mul(0.5);
            INDArray dLdIn = Nd4j.createUninitialized(3, 4);

            String err = OpValidation.validate(new OpTestCase(new NormMaxBp(preReduceInput, dLdOut, dLdIn, keepDims))
                    .expectedOutput(0, dLdInExpected));

            assertNull(err);
        }
    }

    @Test
    public void testNormMaxAlongDimensionBP() {
        OpValidationSuite.ignoreFailing();
        //out = max_i (|in_i|)
        //dL/dIn = dL/dOut * dOut/dIn
        //       = dL/dOut * (0 if |x_i| is not max; or sgn(x_i) otherwise)

        for (boolean keepDims : new boolean[]{false, true}) {

            long[] reducedShape_0 = (keepDims ? new long[]{3, 4} : new long[]{4});
            INDArray preReduceInput = Nd4j.linspace(-5, 6, 12).reshape(3, 4);
            INDArray sgn = Transforms.sign(preReduceInput, true);
            INDArray max_0 = Nd4j.create(3,4);
            max_0.getRow(2).assign(1.0);
            INDArray dLdOut_0 = Nd4j.create(new double[]{1, 2, 3, 4}, reducedShape_0);
            INDArray dLdInExpected_0 = sgn.mul(max_0).mulRowVector(dLdOut_0);

            INDArray dLdIn = Nd4j.createUninitialized(3, 4);

            String err = OpValidation.validate(new OpTestCase(new NormMaxBp(preReduceInput, dLdOut_0, dLdIn, keepDims, 0))
                    .expectedOutput(0, dLdInExpected_0));
            assertNull(err);


            long[] reducedShape_1 = (keepDims ? new long[]{3, 1} : new long[]{3});
            INDArray dLdOut_1 = Nd4j.create(new double[]{1, 2, 3, 4}, reducedShape_1);
            INDArray max_1 = Nd4j.create(3,4);
            max_1.getColumn(3).assign(1.0);
            INDArray dLdInExpected_1 = sgn.mul(max_1).mulColumnVector(dLdOut_1);
            dLdIn = Nd4j.createUninitialized(3, 4);

            err = OpValidation.validate(new OpTestCase(new NormMaxBp(preReduceInput, dLdOut_1, dLdIn, keepDims, 1))
                    .expectedOutput(0, dLdInExpected_1));

            assertNull(err, err);
        }
    }
}
