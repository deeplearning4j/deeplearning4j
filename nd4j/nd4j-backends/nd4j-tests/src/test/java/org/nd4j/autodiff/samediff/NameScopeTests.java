/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
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

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class NameScopeTests extends BaseNd4jTestWithBackends {


    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testVariableNameScopesBasic(Nd4jBackend backend) {

        SameDiff sd = SameDiff.create();
        SDVariable v = sd.var("x");
        try(NameScope ns = sd.withNameScope("nameScope")){
            SDVariable v2 = sd.var("x2");
            assertEquals("nameScope/x2", v2.name());
            assertTrue(sd.getVariables().containsKey("nameScope/x2"));
            assertEquals("nameScope", sd.currentNameScope());

            SDVariable v3 = sd.var("x");
            assertEquals("nameScope/x", v3.name());
            assertTrue(sd.getVariables().containsKey("nameScope/x"));

            try(NameScope ns2 = sd.withNameScope("scope2")){
                assertEquals("nameScope/scope2", sd.currentNameScope());
                SDVariable v4 = sd.var("x");
                assertEquals("nameScope/scope2/x", v4.name());
                assertTrue(sd.getVariables().containsKey("nameScope/scope2/x"));
            }

            assertEquals("nameScope", sd.currentNameScope());
        }
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testOpFieldsAndNames(Nd4jBackend backend) {

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.var("x", DataType.FLOAT, 1);
        SDVariable y;
        SDVariable z;

        SDVariable add;
        SDVariable addWithName;
        SDVariable merge;
        SDVariable mergeWithName;
        try(NameScope ns = sd.withNameScope("s1")){
            y = sd.var("y", DataType.FLOAT, 1);
            add = x.add(y);
            addWithName = x.add("addxy", y);
            try(NameScope ns2 = sd.withNameScope("s2")){
                z = sd.var("z", DataType.FLOAT, 1);
                merge = sd.math().mergeMax(new SDVariable[]{y, z});
                mergeWithName = sd.math.mergeMax("mmax", new SDVariable[]{y, z});
            }
        }
        SDVariable a = sd.var("a", DataType.FLOAT, 1);

        assertEquals("x", x.name());
        assertEquals("s1/y", y.name());
        assertEquals("s1/s2/z", z.name());
        assertEquals("a", a.name());

        assertTrue(add.name().startsWith("s1/"),add.name());
        assertEquals("s1/addxy", addWithName.name());

        assertTrue(merge.name().startsWith("s1/s2/"),merge.name());
        assertEquals("s1/s2/mmax", mergeWithName.name());

        Set<String> allowedVarNames = new HashSet<>(Arrays.asList("x", "s1/y", "s1/s2/z", "a",
                add.name(), addWithName.name(), merge.name(), mergeWithName.name()));
        Set<String> allowedOpNames = new HashSet<>();

        //Check op names:
        Map<String, SameDiffOp> ops = sd.getOps();
        System.out.println(ops.keySet());

        for(String s : ops.keySet()){
            assertTrue(s.startsWith("s1") || s.startsWith("s1/s2"),s);
            allowedOpNames.add(s);
        }

        //Check fields - Variable, SDOp, etc
        for(Variable v : sd.getVariables().values()){
            assertTrue( allowedVarNames.contains(v.getVariable().name()),v.getVariable().name());
            assertEquals(v.getName(), v.getVariable().name());
            if(v.getInputsForOp() != null){
                for(String s : v.getInputsForOp()){
                    assertTrue(allowedOpNames.contains(s),s);
                }
            }

            if(v.getOutputOfOp() != null){
                assertTrue(allowedOpNames.contains(v.getOutputOfOp()));
            }
        }

        assertTrue(allowedOpNames.containsAll(sd.getOps().keySet()));

        for(SameDiffOp op : sd.getOps().values()){
            assertTrue(allowedOpNames.contains(op.getName()));
            assertEquals(op.getName(), op.getOp().getOwnName());
            if(op.getInputsToOp() != null){
                assertTrue(allowedVarNames.containsAll(op.getInputsToOp()));
            }

            if(op.getOutputsOfOp() != null){
                assertTrue(allowedVarNames.containsAll(op.getOutputsOfOp()));
            }
        }
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testNoNesting(Nd4jBackend backend) {
        SameDiff SD = SameDiff.create();

        SDVariable a = SD.constant(4);

        NameScope scope = SD.withNameScope("test");

        SDVariable out = SD.argmax(a);

        out.add(45);

        scope.close();

        assertTrue(SD.variableMap().containsKey("test/argmax"),"Var with name test/argmax exists");
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testNoTesting2(Nd4jBackend backend) {
        SameDiff SD = SameDiff.create();

        SDVariable a = SD.constant(4);
        SDVariable b = SD.constant(5).lt(4);

        NameScope scope = SD.withNameScope("test");

        SDVariable out = SD.switchOp(a, b)[0];

        out.add(45);

        scope.close();

        assertTrue( SD.variableMap().containsKey("test/switch:1"),"Var with name test/switch:1 exists");
    }
}
