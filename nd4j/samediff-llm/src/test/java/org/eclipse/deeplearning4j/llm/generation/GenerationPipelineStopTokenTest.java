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

import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.junit.jupiter.api.Test;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;

class GenerationPipelineStopTokenTest {

    @Test
    void generatedTextExcludesTrailingStopsWithoutDroppingProtocolTokenIds() {
        int[] tokenIds = {10, 11, 7};

        assertEquals(2, GenerationPipeline.contentTokenLength(tokenIds, Set.of(7)));
        assertEquals(3, tokenIds.length);
    }

    @Test
    void configuredEosOverridesTokenizerMetadata() {
        SamplingConfig sampling = SamplingConfig.builder()
                .eosTokenId(11)
                .build();

        assertEquals(11, GenerationPipeline.selectEosTokenId(sampling, 7));
    }

    @Test
    void unsetEosUsesTokenizerMetadata() {
        SamplingConfig sampling = SamplingConfig.builder().build();

        assertEquals(7, GenerationPipeline.selectEosTokenId(sampling, 7));
    }

    @Test
    void zeroIsAnExplicitEosToken() {
        SamplingConfig sampling = SamplingConfig.builder()
                .eosTokenId(0)
                .build();

        assertEquals(0, GenerationPipeline.selectEosTokenId(sampling, 7));
    }

    @Test
    void absentSamplingUsesTokenizerMetadata() {
        assertEquals(7, GenerationPipeline.selectEosTokenId(null, 7));
    }

    @Test
    void importedMetadataRestoresContainerEosWhenTokenizerOmitsIt() {
        SamplingConfig sampling = SamplingConfig.builder().build();

        assertEquals(7, GenerationPipeline.selectEosTokenId(sampling, 7, -1));
    }

    @Test
    void requestOverrideStillWinsOverImportedMetadata() {
        SamplingConfig sampling = SamplingConfig.builder()
                .eosTokenId(11)
                .build();

        assertEquals(11, GenerationPipeline.selectEosTokenId(sampling, 7, 2));
    }

    @Test
    void tokenizerRemainsFallbackForNativeArtifactsWithoutImporterMetadata() {
        SamplingConfig sampling = SamplingConfig.builder().build();

        assertEquals(2, GenerationPipeline.selectEosTokenId(sampling, -1, 2));
    }

    @Test
    void tokenizerDeclaredNativeMarkersSelectModelProtocolWithoutTemplateMarkers() {
        assertEquals(ChatTemplate.ToolCallFormat.NATIVE,
                GenerationPipeline.selectModelToolCallFormat(
                        ChatTemplate.ToolCallFormat.JSON, 10, 11, Set.of(7, 10, 11)));
    }

    @Test
    void decodedSpecialMarkersSelectNativeWhenAddedTokensLackReverseLookup() {
        assertEquals(ChatTemplate.ToolCallFormat.NATIVE,
                GenerationPipeline.selectModelToolCallFormat(
                        ChatTemplate.ToolCallFormat.JSON, null, null, Set.of(7, 10, 11),
                        Set.of(ChatTemplate.NATIVE_TOOL_CALL_START,
                                ChatTemplate.NATIVE_TOOL_CALL_END, "<|im_end|>")));
    }

    @Test
    void absentNativeMarkerPairKeepsModelJsonProtocol() {
        assertEquals(ChatTemplate.ToolCallFormat.JSON,
                GenerationPipeline.selectModelToolCallFormat(
                        ChatTemplate.ToolCallFormat.JSON, 10, null, Set.of(7, 10)));
    }
}
