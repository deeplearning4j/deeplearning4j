package org.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.nd4j.autodiff.samediff.transform.*;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
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


        SubGraphPredicate p = SubGraphPredicate.withRoot(OpPredicate.classEquals(AddOp.class));

        List<SubGraph> l = GraphTransformUtil.getSubgraphsMatching(sd, p);
        assertEquals(2, l.size());

        SubGraph sg1 = l.get(0);
        assertTrue(sg1.getRootNode() == sd.getVariableOutputFunction(add.getVarName()));
        assertEquals(0, sg1.getChildNodes().size());

        SubGraph sg2 = l.get(1);
        assertTrue(sg2.getRootNode() == sd.getVariableOutputFunction(add2.getVarName()));
        assertEquals(0, sg2.getChildNodes().size());
    }

    @Test
    public void testSubgraphReplace1(){

        SameDiff sd = SameDiff.create();
        SDVariable ph1 = sd.placeHolder("ph1", DataType.FLOAT, -1, 4);
        SDVariable ph2 = sd.placeHolder("ph2", DataType.FLOAT, -1, 4);

        INDArray p1 = Nd4j.ones(DataType.FLOAT, 1, 4);
        INDArray p2 = Nd4j.ones(DataType.FLOAT, 1, 4).mul(3);
        ph1.setArray(p1);
        ph2.setArray(p2);

        SDVariable add = ph1.add(ph2);
        SDVariable sub = ph1.sub(ph2);
        SDVariable mul = add.mul(sub);

//        INDArray out = mul.eval();
//        INDArray exp = p1.add(p2).mul(p1.sub(p2));
//        assertEquals(exp, out);

        SubGraphPredicate p = SubGraphPredicate.withRoot(OpPredicate.classEquals(AddOp.class));

        SameDiff sd2 = GraphTransformUtil.replaceSubgraphsMatching(sd, p, new SubGraphProcessor() {
            @Override
            public List<SDVariable> processSubgraph(SameDiff sd, SubGraph subGraph) {
                //Let's replace add op with div op
                assertTrue(subGraph.getChildNodes() == null || subGraph.getChildNodes().isEmpty());
                List<SDVariable> inputs = subGraph.inputs();
                SDVariable out = inputs.get(0).div(inputs.get(1));
                return Collections.singletonList(out);
            }
        });

        INDArray exp2 = p1.div(p2).mul(p1.sub(p2));
        INDArray out2 = sd2.getVariable(mul.getVarName()).eval();
        assertEquals(exp2, out2);


    }

}
