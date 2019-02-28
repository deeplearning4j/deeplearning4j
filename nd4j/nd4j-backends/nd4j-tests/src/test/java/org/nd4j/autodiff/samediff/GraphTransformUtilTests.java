package org.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.nd4j.autodiff.samediff.transform.OpPredicate;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

@Slf4j
public class GraphTransformUtilTests {

    @Test
    public void testBasic(){

        SameDiff sd = SameDiff.create();
        SDVariable ph1 = sd.placeHolder("ph1", DataType.FLOAT, -1, 32);
        SDVariable ph2 = sd.placeHolder("ph2", DataType.FLOAT, -1, 32);

        SDVariable add = ph1.add(ph2);
        SDVariable add2 = add.add(ph1);

        SDVariable sub = add.sub(add2);

        assertTrue(OpPredicate.classEquals(AddOp.class).matches(sd, sd.getVariableOutputFunction(add.getVarName())));
        assertTrue(OpPredicate.classEquals(AddOp.class).matches(sd, sd.getVariableOutputFunction(add2.getVarName())));
        assertFalse(OpPredicate.classEquals(AddOp.class).matches(sd, sd.getVariableOutputFunction(sub.getVarName())));

        assertTrue(OpPredicate.opNameEquals(AddOp.OP_NAME).matches(sd, sd.getVariableOutputFunction(add.getVarName())));
        assertTrue(OpPredicate.opNameEquals(AddOp.OP_NAME).matches(sd, sd.getVariableOutputFunction(add2.getVarName())));
        assertFalse(OpPredicate.opNameEquals(AddOp.OP_NAME).matches(sd, sd.getVariableOutputFunction(sub.getVarName())));

        assertTrue(OpPredicate.opNameMatches(".*dd").matches(sd, sd.getVariableOutputFunction(add.getVarName())));
        assertTrue(OpPredicate.opNameMatches("ad.*").matches(sd, sd.getVariableOutputFunction(add2.getVarName())));
        assertFalse(OpPredicate.opNameMatches(".*dd").matches(sd, sd.getVariableOutputFunction(sub.getVarName())));



    }

}
