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

import lombok.Data;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;

import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.impl.reduce.Mmul;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.same.Identity;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.nio.file.Path;
import java.util.*;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

public class TestSeamlessOptimization extends BaseNd4jTestWithBackends {

    @TempDir
    Path tempDir;


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testOutput(Nd4jBackend nd4jBackend) {

        //Ensure that optimizer is actually used when calling output methods:
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 4);
        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 4, 3));
        SDVariable b = sd.var("b", Nd4j.rand(DataType.FLOAT, 3));

        SDVariable i1 = sd.identity(in);
        SDVariable i2 = sd.identity(w);
        SDVariable i3 = sd.identity(b);

        SDVariable out = sd.nn.softmax("out", sd.identity(i1.mmul(i2).add(i3)));

        sd = GraphOptimizer.optimize(sd,"out");

        RecordOpsListener l = new RecordOpsListener();
        sd.setListeners(new AssertNoOpsOfTypeListener(Identity.class), l);

        Map<String, INDArray> ph = Collections.singletonMap("in", Nd4j.rand(DataType.FLOAT, 10, 4));

        for( int i = 0; i < 3; i++) {
            l.ops.clear();

            switch (i){
                case 0:
                    sd.outputSingle(ph, "out");
                    break;
                case 1:
                    sd.output(ph, "out");
                    break;
                case 2:
                    sd.batchOutput().output("out")
                            .input("in", ph.get("in"))
                            .outputSingle();
                    break;
            }


            List<Class<?>> expClasses = Arrays.asList(Mmul.class, AddOp.class, SoftMax.class);
            assertEquals(3, l.ops.size());
            for (int j = 0; j < 3; j++) {
                assertEquals(expClasses.get(j), l.ops.get(j).getOp().getClass());
            }

        }
    }


    public static class AssertNoOpsOfTypeListener extends BaseListener {
        private List<Class<?>> list;

        public AssertNoOpsOfTypeListener(Class<? extends DifferentialFunction>... c) {
            Preconditions.checkState(c != null && c.length > 0, "No classes provided");
            this.list = Arrays.asList(c);
        }

        @Override
        public boolean isActive(Operation operation) {
            return true;
        }

        @Override
        public void opExecution(SameDiff sd, At at, MultiDataSet batch, SameDiffOp op, OpContext opContext, INDArray[] outputs) {
            if(list.contains(op.getOp().getClass())) {
                throw new IllegalStateException("Encountered unexpected class: " + op.getOp().getClass().getName());
            }
        }
    }

    @Data
    public static class RecordOpsListener extends BaseListener {

        private List<SameDiffOp> ops = new ArrayList<>();

        @Override
        public boolean isActive(Operation operation) {
            return true;
        }

        @Override
        public void opExecution(SameDiff sd, At at, MultiDataSet batch, SameDiffOp op, OpContext opContext, INDArray[] outputs) {
            ops.add(op);

        }
    }


    @Override
    public char ordering() {
        return 'c';
    }
}