package org.nd4j.autodiff.opvalidation;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.autodiff.validation.OpTestCase;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;

import static org.junit.Assert.assertEquals;

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

        //sum_bp op: has 2 inputs (original pre-reduce input, and gradient at output (epsilon))
        //out = sum_j (in_j) -> dL/dIn = dL/dOut * dOut/dIn = dL/dOut

        INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3,4);
        INDArray dLdOut = Nd4j.trueScalar(0.5);
        INDArray dLdInExpected = Nd4j.valueArrayOf(preReduceInput.shape(), 0.5);
        INDArray dLdIn = Nd4j.createUninitialized(3,4);

        OpValidation.validate(new OpTestCase(
                DynamicCustomOp.builder("sum_bp")
                .addInputs(preReduceInput, dLdOut)
                .addOutputs(dLdIn)
                //TODO Not sure if we're going with "no reduction dimension == full array reduce" or using Integer.MAX_VALUE to signify this
                //.addIntegerArguments(Integer.MAX_VALUE)
                .build())
                .expectedOutput(0, dLdInExpected)
        );
    }

    @Test
    public void testReduceSumAlongDim(){
        //Reduction along dimension

        //Inputs/outputs as before - but note that the output is no longer a scalar

        INDArray preReduceInput = Nd4j.linspace(1, 12, 12).reshape(3,4);
        INDArray dLdOut_0 = Nd4j.create(new double[]{1,2,3,4}, new long[]{4});      //Rank 1, 4 elements
        INDArray dLdInExpected_0 = Nd4j.createUninitialized(preReduceInput.shape());
        for( int i=0; i<3; i++ ){

        }

        INDArray dLdIn = Nd4j.createUninitialized(3,4);

        OpValidation.validate(new OpTestCase(
                DynamicCustomOp.builder("sum_bp")
                        .addInputs(preReduceInput, dLdOut_0)
                        .addOutputs(dLdIn)
                        .addIntegerArguments(0) //Reduction along dimension 0
                        .build())
                .expectedOutput(0, dLdInExpected_0)
        );
    }



}
