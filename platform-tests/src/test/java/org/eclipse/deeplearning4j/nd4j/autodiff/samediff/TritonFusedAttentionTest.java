/*
 *  ******************************************************************************
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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for fused attention patterns with Triton GPU backend.
 * Tests manual attention: Q@Kt -> scale -> softmax -> @V for decode, prefill, and cross-attn scenarios.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class TritonFusedAttentionTest extends BaseNd4jTestWithBackends {

    private static final double ATTENTION_TOLERANCE = 5e-3;

    @AfterEach
    public void cleanup() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.invalidateTritonCache();
        nativeOps.resetTritonCounters();
        Nd4j.getMemoryManager().purgeCaches();
        System.gc();
        nativeOps.trimMemoryPool(0);
    }

    @Test @DisplayName("Attention Decode: Q=[1,1,64] @ K=[1,1,64] -> scale -> softmax @ V=[1,1,64]")
    public void testAttentionDecode() {
        SameDiff sd = SameDiff.create();
        SDVariable Q = sd.placeHolder("Q", DataType.FLOAT, -1, 1, 64);
        SDVariable K = sd.constant("K", Nd4j.randn(DataType.FLOAT, 1, 1, 64));
        SDVariable V = sd.constant("V", Nd4j.randn(DataType.FLOAT, 1, 1, 64));

        float scale = 1.0f / (float) Math.sqrt(64);

        // Q @ K^T -> [1, 1, 1]
        SDVariable Kt = sd.permute("Kt", K, 0, 2, 1);
        SDVariable scores = sd.mmul("scores", Q, Kt);

        // Scale
        SDVariable scaled = scores.mul("scaled", sd.constant("scale", Nd4j.scalar(DataType.FLOAT, scale)));

        // Softmax along last dim
        SDVariable attn = sd.nn().softmax("attn", scaled, -1);

        // @ V -> [1, 1, 64]
        sd.mmul("result", attn, V);

        TritonTestUtils.runOpTest("testAttentionDecode", sd,
            Map.of("Q", Nd4j.randn(DataType.FLOAT, 1, 1, 64)), "result");
        sd.close();
    }

    @Test @DisplayName("Attention Prefill: Q=[1,8,64] @ K=[1,8,64] -> scale -> softmax @ V=[1,8,64]")
    public void testAttentionPrefill() {
        SameDiff sd = SameDiff.create();
        SDVariable Q = sd.placeHolder("Q", DataType.FLOAT, -1, 8, 64);
        SDVariable K = sd.constant("K", Nd4j.randn(DataType.FLOAT, 1, 8, 64));
        SDVariable V = sd.constant("V", Nd4j.randn(DataType.FLOAT, 1, 8, 64));

        float scale = 1.0f / (float) Math.sqrt(64);

        SDVariable Kt = sd.permute("Kt", K, 0, 2, 1);
        SDVariable scores = sd.mmul("scores", Q, Kt);
        SDVariable scaled = scores.mul("scaled", sd.constant("scale", Nd4j.scalar(DataType.FLOAT, scale)));
        SDVariable attn = sd.nn().softmax("attn", scaled, -1);
        sd.mmul("result", attn, V);

        TritonTestUtils.runOpTest("testAttentionPrefill", sd,
            Map.of("Q", Nd4j.randn(DataType.FLOAT, 1, 8, 64)), "result");
        sd.close();
    }

    @Test @DisplayName("Cross Attention: Q=[1,4,64] @ K=[1,8,64] -> scale -> softmax @ V=[1,8,64]")
    public void testCrossAttention() {
        SameDiff sd = SameDiff.create();
        SDVariable Q = sd.placeHolder("Q", DataType.FLOAT, -1, 4, 64);
        SDVariable K = sd.constant("K", Nd4j.randn(DataType.FLOAT, 1, 8, 64));
        SDVariable V = sd.constant("V", Nd4j.randn(DataType.FLOAT, 1, 8, 64));

        float scale = 1.0f / (float) Math.sqrt(64);

        SDVariable Kt = sd.permute("Kt", K, 0, 2, 1);
        SDVariable scores = sd.mmul("scores", Q, Kt);
        SDVariable scaled = scores.mul("scaled", sd.constant("scale", Nd4j.scalar(DataType.FLOAT, scale)));
        SDVariable attn = sd.nn().softmax("attn", scaled, -1);
        sd.mmul("result", attn, V);

        TritonTestUtils.runOpTest("testCrossAttention", sd,
            Map.of("Q", Nd4j.randn(DataType.FLOAT, 1, 4, 64)), "result");
        sd.close();
    }

    @Test @DisplayName("Attention + Output Projection: Q@Kt->scale->softmax->@V->@Wo [1,4,64]")
    public void testAttentionWithOutputProjection() {
        SameDiff sd = SameDiff.create();
        SDVariable Q = sd.placeHolder("Q", DataType.FLOAT, -1, 4, 64);
        SDVariable K = sd.constant("K", Nd4j.randn(DataType.FLOAT, 1, 4, 64));
        SDVariable V = sd.constant("V", Nd4j.randn(DataType.FLOAT, 1, 4, 64));
        SDVariable Wo = sd.constant("Wo", Nd4j.randn(DataType.FLOAT, 64, 64));

        float scale = 1.0f / (float) Math.sqrt(64);

        SDVariable Kt = sd.permute("Kt", K, 0, 2, 1);
        SDVariable scores = sd.mmul("scores", Q, Kt);
        SDVariable scaled = scores.mul("scaled", sd.constant("scale", Nd4j.scalar(DataType.FLOAT, scale)));
        SDVariable attn = sd.nn().softmax("attn", scaled, -1);
        SDVariable context = sd.mmul("context", attn, V);
        sd.mmul("result", context, Wo);

        TritonTestUtils.runOpTest("testAttentionWithOutputProjection", sd,
            Map.of("Q", Nd4j.randn(DataType.FLOAT, 1, 4, 64)), "result");
        sd.close();
    }
}
