/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.llm.generation;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Contract tests for {@code GenerationPipeline.resolveFixedBufferMaxKvLen}: the single source of
 * truth for the physical KV/mask envelope of a fixed-buffer plan.
 *
 * <p>Regression context (r35): the one-shot retain-and-rebind guard computed its expected envelope
 * inline as {@code min(promptLen + maxNewTokens, kvCap)} while the allocator uses the FULL
 * configured envelope — expected ~4700 vs actual 8192 meant the signature never matched and reuse
 * never fired. These tests pin the shared resolver so the guard and the allocator can never
 * diverge silently again.</p>
 */
class FixedBufferMaxKvLenResolverTest {

    private GenerationPipelineConfig config(int maxPrefill, int kvCap) {
        return GenerationPipelineConfig.builder()
                .maxPrefillLength(maxPrefill)
                .maxKvCacheLength(kvCap)
                .build();
    }

    @Test
    void configuredEnvelopeWinsOverRequestedDecodeCapacity() {
        // The FP&A capture profile: prefill 4352, KV envelope 8192. A request with a 4159-token
        // prompt and 256 new tokens must expect the FULL 8192 envelope, not 4159+256.
        assertEquals(8192L, GenerationPipeline.resolveFixedBufferMaxKvLen(config(4352, 8192), 256));
        assertEquals(8192L, GenerationPipeline.resolveFixedBufferMaxKvLen(config(4352, 8192), 4096));
        assertEquals(8192L, GenerationPipeline.resolveFixedBufferMaxKvLen(config(4352, 8192), 3833));
    }

    @Test
    void withoutConfiguredCapEnvelopeIsPrefillPlusRequestedCapacity() {
        assertEquals(4352L + 512L, GenerationPipeline.resolveFixedBufferMaxKvLen(config(4352, 0), 512));
        assertEquals(4096L + 128L, GenerationPipeline.resolveFixedBufferMaxKvLen(config(4096, 0), 128));
    }

    @Test
    void zeroDecodeCapacityStillUsesFullConfiguredEnvelope() {
        assertEquals(8192L, GenerationPipeline.resolveFixedBufferMaxKvLen(config(4352, 8192), 0));
    }

    @Test
    void rejectsNonFixedBufferConfig() {
        assertThrows(IllegalArgumentException.class,
                () -> GenerationPipeline.resolveFixedBufferMaxKvLen(config(0, 8192), 256));
        assertThrows(IllegalArgumentException.class,
                () -> GenerationPipeline.resolveFixedBufferMaxKvLen(config(-1, 8192), 256));
    }

    @Test
    void rejectsNegativeDecodeCapacity() {
        assertThrows(IllegalArgumentException.class,
                () -> GenerationPipeline.resolveFixedBufferMaxKvLen(config(4352, 8192), -1));
    }

    @Test
    void rejectsNullConfig() {
        assertThrows(NullPointerException.class,
                () -> GenerationPipeline.resolveFixedBufferMaxKvLen(null, 256));
    }
}
