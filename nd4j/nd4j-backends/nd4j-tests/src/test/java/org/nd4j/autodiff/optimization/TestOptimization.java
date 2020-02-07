package org.nd4j.autodiff.optimization;

import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Collections;

import static org.junit.Assert.assertEquals;

public class TestOptimization extends BaseNd4jTest {

    public TestOptimization(Nd4jBackend backend) {
        super(backend);
    }

    @Override
    public char ordering() {
        return 'c';
    }


    @Test
    public void testConstantOpFolding(){

        SameDiff sd = SameDiff.create();
        SDVariable c = sd.constant("c", Nd4j.scalar(1.0));
        SDVariable v = c.add("add", 1);

        SameDiff optimized = GraphOptimizer.optimize(sd);
        assertEquals(2, optimized.getVariables().size());
        assertEquals(VariableType.CONSTANT, optimized.getVariable("add").getVariableType());
        assertEquals(0, optimized.getOps().size());

        assertEquals(sd.outputSingle(Collections.emptyMap(), "add"), optimized.outputSingle(Collections.emptyMap(), "add"));
    }
}
