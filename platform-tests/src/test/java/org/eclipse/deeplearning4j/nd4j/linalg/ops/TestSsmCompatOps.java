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

package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Regression tests for the P5 SSM compat ops: ssm_conv (adapter over
 * causal_conv1d) and ssm_scan (inline ZOH discretization + selective_scan).
 * Validated native-vs-native against the ops they delegate to.
 *
 * <h2>Running</h2>
 * <pre>
 * cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-native \
 *   -Dtest=TestSsmCompatOps 2>&1 | tee /tmp/test-ssm.log
 * </pre>
 */
@Slf4j
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class TestSsmCompatOps {

    @Test
    public void testSsmConvMatchesCausalConv1d() {
        Nd4j.getRandom().setSeed(31);
        int B = 2, L = 6, D = 4, K = 4;
        INDArray x = Nd4j.rand(DataType.FLOAT, B, L, D).subi(0.5);
        INDArray weight = Nd4j.rand(DataType.FLOAT, D, K).subi(0.5);

        INDArray viaCompat = Nd4j.exec(DynamicCustomOp.builder("ssm_conv")
                .addInputs(x, weight).build())[0];

        // causal_conv1d, activation=none, wFormat=[D,K]; take output[0]
        INDArray viaNative = Nd4j.exec(DynamicCustomOp.builder("causal_conv1d")
                .addInputs(x, weight)
                .addIntegerArguments(0, 0)
                .build())[0];

        assertArrayEquals(x.shape(), viaCompat.shape());
        assertEquals(viaNative, viaCompat, "ssm_conv must equal causal_conv1d output[0]");
    }

    @Test
    public void testSsmScanMatchesDiscretizedSelectiveScan() {
        Nd4j.getRandom().setSeed(32);
        int B = 2, L = 5, dim = 4, state = 3;
        INDArray x = Nd4j.rand(DataType.FLOAT, B, L, dim).subi(0.5);
        INDArray dt = Nd4j.rand(DataType.FLOAT, B, L, state).muli(0.1);         // small positive steps
        INDArray a = Nd4j.rand(DataType.FLOAT, B, L, state).muli(-1);           // continuous A (negative)
        INDArray b = Nd4j.rand(DataType.FLOAT, B, L, state).subi(0.5);
        INDArray c = Nd4j.rand(DataType.FLOAT, B, L, state).subi(0.5);

        INDArray viaCompat = Nd4j.exec(DynamicCustomOp.builder("ssm_scan")
                .addInputs(x, dt, a, b, c).build())[0];

        // reference: pre-discretize, then selective_scan(x, A_bar, B_bar, C, D=0)
        INDArray aBar = Transforms.exp(dt.mul(a));
        INDArray bBar = dt.mul(b);
        INDArray dZero = Nd4j.zeros(DataType.FLOAT, dim);
        INDArray viaNative = Nd4j.exec(DynamicCustomOp.builder("selective_scan")
                .addInputs(x, aBar, bBar, c, dZero).build())[0];

        assertArrayEquals(x.shape(), viaCompat.shape());
        assertEquals(viaNative, viaCompat, "ssm_scan must equal discretized selective_scan");
    }

    @Test
    public void testSsmScanWithInitialState() {
        Nd4j.getRandom().setSeed(33);
        int B = 1, L = 4, dim = 3, state = 2;
        INDArray x = Nd4j.rand(DataType.FLOAT, B, L, dim).subi(0.5);
        INDArray dt = Nd4j.rand(DataType.FLOAT, B, L, state).muli(0.1);
        INDArray a = Nd4j.rand(DataType.FLOAT, B, L, state).muli(-1);
        INDArray b = Nd4j.rand(DataType.FLOAT, B, L, state).subi(0.5);
        INDArray c = Nd4j.rand(DataType.FLOAT, B, L, state).subi(0.5);
        INDArray s = Nd4j.rand(DataType.FLOAT, B, dim, state).subi(0.5);

        INDArray viaCompat = Nd4j.exec(DynamicCustomOp.builder("ssm_scan")
                .addInputs(x, dt, a, b, c, s).build())[0];

        INDArray aBar = Transforms.exp(dt.mul(a));
        INDArray bBar = dt.mul(b);
        INDArray dZero = Nd4j.zeros(DataType.FLOAT, dim);
        INDArray viaNative = Nd4j.exec(DynamicCustomOp.builder("selective_scan")
                .addInputs(x, aBar, bBar, c, dZero, s).build())[0];

        assertArrayEquals(x.shape(), viaCompat.shape());
        assertEquals(viaNative, viaCompat, "ssm_scan with initial state must match selective_scan(h0=s)");
    }
}
