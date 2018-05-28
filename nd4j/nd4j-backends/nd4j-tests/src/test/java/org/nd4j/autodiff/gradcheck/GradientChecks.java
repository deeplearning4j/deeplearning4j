package org.nd4j.autodiff.gradcheck;

import org.junit.After;
import org.junit.Before;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;

@RunWith(Suite.class)
@Suite.SuiteClasses({
        GradCheckMisc.class,
        GradCheckReductions.class
})
public class GradientChecks {

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






}
