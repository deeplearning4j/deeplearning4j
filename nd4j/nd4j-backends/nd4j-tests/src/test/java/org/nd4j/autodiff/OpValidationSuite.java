package org.nd4j.autodiff;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.nd4j.autodiff.gradcheck.GradCheckLoss;
import org.nd4j.autodiff.gradcheck.GradCheckMisc;
import org.nd4j.autodiff.gradcheck.GradCheckReductions;
import org.nd4j.autodiff.gradcheck.GradCheckTransforms;
import org.nd4j.autodiff.opvalidation.ReductionOpValidationTests;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.function.Function;

/**
 * Op validation test suite.
 * Should include all classes using the {@link OpValidation} test framework, so test coverage can be calculated and reported.
 *
 * NOTE: For the op coverage information to work properly, we need the op validation to be run via this suite.
 * Otherwise, we could report coverage information before all test have run - underestimating coverage.
 *
 * For op coverage information to be collected, you need to execute a test like:<br>
 * SINGLE OP TEST: OpValidation.validate(new OpTestCase(op).expectedOutputs(0, <INDArray here>))
 *     - OpTestCase checks the output values of a single op, no backprop/gradients<br>
 *     - Returns an error message if op failed, or NULL if op passed<br>
 * SAMEDIFF TEST:  OpValidation.validate(new TestCase(sameDiff).gradientCheck(true).expectedOutput("someVar", <INDArray>))<br>
 *     - These tests can be used to check both gradients AND expected output, collecting coverage as required<br>
 *     - Returns an error message if op failed, or NULL if op passed<br>
 *     - Note gradientCheck(true) is the default<br>
 *     - Expected outputs are optional<br>
 *     - You can specify a function for validating the correctness of each output using {@link org.nd4j.autodiff.validation.TestCase#expected(String, Function)}<br>
 *
 */
@RunWith(Suite.class)
@Suite.SuiteClasses({
        //Note: these will be run as part of the suite only, and will NOT be run again separately
        ReductionOpValidationTests.class,
        GradCheckLoss.class,
        GradCheckMisc.class,
        GradCheckReductions.class,
        GradCheckTransforms.class
})
public class OpValidationSuite {

    private static DataBuffer.Type initialType;

    @BeforeClass
    public static void beforeClass() throws Exception {
        Nd4j.create(1);
        initialType = Nd4j.dataType();

        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
        Nd4j.getRandom().setSeed(123);
    }

    @AfterClass
    public static void afterClass() throws Exception {
        Nd4j.setDataType(initialType);

        // Log coverage information
        OpValidation.logCoverageInformation(true, true);
    }






}
