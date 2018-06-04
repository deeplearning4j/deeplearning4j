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

/**
 * Op validation test suite
 * Should include all classes using the {@link OpValidation} test framework, so test coverage can be calculated and reported
 */
@RunWith(Suite.class)
@Suite.SuiteClasses({
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

        OpValidation.logCoverageInformation(true, true);
    }






}
