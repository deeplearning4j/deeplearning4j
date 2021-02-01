/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.nd4j.autodiff.samediff.transform.GraphTransformUtil;
import org.nd4j.autodiff.samediff.transform.OpPredicate;
import org.nd4j.autodiff.samediff.transform.SubGraph;
import org.nd4j.autodiff.samediff.transform.SubGraphPredicate;
import org.nd4j.autodiff.samediff.transform.SubGraphProcessor;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

@Slf4j
public class GraphTransformUtilTests extends BaseNd4jTest {

    public GraphTransformUtilTests(Nd4jBackend b){
        super(b);
    }

    @Override
    public char ordering(){
        return 'c';
    }

    @Test
    public void testBasic(){

        SameDiff sd = SameDiff.create();
        SDVariable ph1 = sd.placeHolder("ph1", DataType.FLOAT, -1, 32);
        SDVariable ph2 = sd.placeHolder("ph2", DataType.FLOAT, -1, 32);

        SDVariable add = ph1.add(ph2);
        SDVariable add2 = add.add(ph1);

        SDVariable sub = add.sub(add2);

        assertTrue(OpPredicate.classEquals(AddOp.class).matches(sd, sd.getVariableOutputOp(add.name())));
        assertTrue(OpPredicate.classEquals(AddOp.class).matches(sd, sd.getVariableOutputOp(add2.name())));
        assertFalse(OpPredicate.classEquals(AddOp.class).matches(sd, sd.getVariableOutputOp(sub.name())));

        assertTrue(OpPredicate.opNameEquals(AddOp.OP_NAME).matches(sd, sd.getVariableOutputOp(add.name())));
        assertTrue(OpPredicate.opNameEquals(AddOp.OP_NAME).matches(sd, sd.getVariableOutputOp(add2.name())));
        assertFalse(OpPredicate.opNameEquals(AddOp.OP_NAME).matches(sd, sd.getVariableOutputOp(sub.name())));

        assertTrue(OpPredicate.opNameMatches(".*dd").matches(sd, sd.getVariableOutputOp(add.name())));
        assertTrue(OpPredicate.opNameMatches("ad.*").matches(sd, sd.getVariableOutputOp(add2.name())));
        assertFalse(OpPredicate.opNameMatches(".*dd").matches(sd, sd.getVariableOutputOp(sub.name())));


        SubGraphPredicate p = SubGraphPredicate.withRoot(OpPredicate.classEquals(AddOp.class));

        List<SubGraph> l = GraphTransformUtil.getSubgraphsMatching(sd, p);
        assertEquals(2, l.size());

        SubGraph sg1 = l.get(0);
        assertTrue(sg1.getRootNode() == sd.getVariableOutputOp(add.name()));
        assertEquals(0, sg1.getChildNodes().size());

        SubGraph sg2 = l.get(1);
        assertTrue(sg2.getRootNode() == sd.getVariableOutputOp(add2.name()));
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
        INDArray out2 = sd2.getVariable(mul.name()).eval();
        assertEquals(exp2, out2);


    }

}
