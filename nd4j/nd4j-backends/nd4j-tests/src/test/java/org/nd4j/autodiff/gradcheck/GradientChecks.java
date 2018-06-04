package org.nd4j.autodiff.gradcheck;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;

@RunWith(Suite.class)
@Suite.SuiteClasses({
        GradCheckLoss.class,
        GradCheckMisc.class,
        GradCheckReductions.class,
        GradCheckTransforms.class
})
public class GradientChecks {

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

        GradCheckUtil.logCoverageInformation(true, true, true);
    }






}
