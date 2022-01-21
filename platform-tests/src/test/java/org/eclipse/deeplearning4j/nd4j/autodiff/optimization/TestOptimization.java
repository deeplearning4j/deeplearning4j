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
package org.eclipse.deeplearning4j.nd4j.autodiff.optimization;

import org.eclipse.deeplearning4j.nd4j.autodiff.optimization.util.OptTestConfig;
import org.eclipse.deeplearning4j.nd4j.autodiff.optimization.util.OptimizationTestUtil;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.autodiff.samediff.optimize.optimizations.ConstantFunctionOptimizations;
import org.nd4j.autodiff.samediff.optimize.optimizations.IdentityFunctionOptimizations;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.File;
import java.nio.file.Path;
import java.util.Collections;

import static org.junit.Assert.*;

@Tag(TagNames.DL4J_OLD_API)
public class TestOptimization extends BaseNd4jTestWithBackends {
    @TempDir
    Path tempDir;

    @Override
    public char ordering() {
        return 'c';
    }

    @Override
    public long getTimeoutMilliseconds() {
        return 1_000_000_000L;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConstantOpFolding(Nd4jBackend nd4jBackend) {
        //We expect 2 things in this test:
        //(a) the output of  add(constant, constant) is pre-calculated and itself becomes a constant
        //(b) the


        SameDiff sd = SameDiff.create();
        SDVariable c = sd.constant("c", Nd4j.scalar(1.0));
        SDVariable c2 = c.add("add", 1);
        SDVariable v = sd.var("variable", Nd4j.scalar(1.0));
        SDVariable out = v.sub("out", c2);

        SameDiff copy = sd.dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "out");
        assertEquals(3, optimized.getVariables().size());       //"add", "variable", "out" -> "c" should be removed
        assertEquals(VariableType.CONSTANT, optimized.getVariable("add").getVariableType());
        assertEquals(1, optimized.getOps().size());
        assertEquals("subtract", optimized.getOps().values().iterator().next().getName());

        assertFalse(optimized.hasVariable("c"));

        assertEquals(sd.outputSingle(Collections.emptyMap(), "out"), optimized.outputSingle(Collections.emptyMap(), "out"));

        //Check the

        //Check that the original can be saved and loaded, and still gives the same results

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConstantOpFolding2(Nd4jBackend nd4jBackend) {
        //We expect 2 things in this test:
        //(a) the output of  add(constant, constant) is pre-calculated and itself becomes a constant
        //(b) the


        SameDiff sd = SameDiff.create();
        SDVariable c = sd.constant("c", Nd4j.scalar(1.0));
        SDVariable c2 = c.add("add", 1);
        SDVariable v = sd.var("variable", Nd4j.scalar(1.0));
        SDVariable out = v.sub("out", c2);

        File subDir = tempDir.resolve("op-folding").toFile();
        assertTrue(subDir.mkdirs());
        OptTestConfig conf = OptTestConfig.builder()
                .original(sd)
                .tempFolder(subDir)
                .outputs(Collections.singletonList("out"))
                .mustApply(sd.getVariables().get("add").getOutputOfOp(), ConstantFunctionOptimizations.FoldConstantFunctions.class)
                .build();

        SameDiff optimized = OptimizationTestUtil.testOptimization(conf);
        assertEquals(3, optimized.getVariables().size());       //"add", "variable", "out" -> "c" should be removed
        assertEquals(VariableType.CONSTANT, optimized.getVariable("add").getVariableType());
        assertEquals(1, optimized.getOps().size());
        assertEquals("subtract", optimized.getOps().values().iterator().next().getName());

        assertFalse(optimized.hasVariable("c"));

        assertEquals(sd.outputSingle(Collections.emptyMap(), "out"), optimized.outputSingle(Collections.emptyMap(), "out"));

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIdentityRemoval(Nd4jBackend nd4jBackend) {

        //Ensure that optimizer is actually used when calling output methods:
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 4);
        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 4, 3));
        SDVariable b = sd.var("b", Nd4j.rand(DataType.FLOAT, 3));
        SDVariable i1 = sd.identity(in);
        SDVariable i2 = sd.identity(w);
        SDVariable i3 = sd.identity(b);
        SDVariable out = sd.nn.softmax("out", sd.identity(i1.mmul(i2).add(i3)));


        File subDir = tempDir.resolve("new-dir-identity-removal").toFile();
        assertTrue(subDir.mkdirs());

        OptTestConfig conf = OptTestConfig.builder()
                .original(sd)
                .tempFolder(subDir)
                .outputs(Collections.singletonList("out"))
                .placeholder("in", Nd4j.rand(DataType.FLOAT, 5, 4))
                .mustApply(sd.getVariables().get(i1.name()).getOutputOfOp(), IdentityFunctionOptimizations.RemoveIdentityOps.class)
                .mustApply(sd.getVariables().get(i2.name()).getOutputOfOp(), IdentityFunctionOptimizations.RemoveIdentityOps.class)
                .mustApply(sd.getVariables().get(i3.name()).getOutputOfOp(), IdentityFunctionOptimizations.RemoveIdentityOps.class)
                .build();

        SameDiff optimized = OptimizationTestUtil.testOptimization(conf);
        assertEquals(3, optimized.getOps().size());
        assertFalse(optimized.hasVariable(i1.name()));
        assertFalse(optimized.hasVariable(i2.name()));
        assertFalse(optimized.hasVariable(i3.name()));
        assertTrue(optimized.hasVariable("out"));
    }
}