package org.nd4j.autodiff.samediff.impl;

import org.junit.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SameDiffOpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.factory.Nd4j;

public class SameDiffOpExecutionerTest {

    @Test
    public void testupdateGraphFromProfiler() {
        SameDiffOpExecutioner sameDiffOpExecutioner = new SameDiffOpExecutioner();
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.ALL);
        Nd4j.getExecutioner().exec(new Sigmoid(Nd4j.scalar(1.0)));
        SameDiff sameDiff = sameDiffOpExecutioner.getSameDiff();

    }

}
