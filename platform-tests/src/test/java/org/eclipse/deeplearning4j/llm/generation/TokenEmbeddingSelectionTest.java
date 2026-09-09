/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.llm.generation;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.*;

class TokenEmbeddingSelectionTest {
    @Test
    void declaredTokenTableWinsOverLargerPerLayerEmbedding() {
        for (String name : new String[]{"token_embd.weight", "model.embed_tokens.weight"}) {
            try (SameDiff graph = SameDiff.create()) {
                var tokens = Nd4j.ones(DataType.FLOAT, 8, 4);
                graph.var(name, tokens);
                graph.var("per_layer_token_embd.weight", Nd4j.ones(DataType.FLOAT, 8, 32));
                assertSame(tokens, GenerationPipeline.extractEmbeddingTable(graph));
            }
        }
    }

    @Test
    void malformedDeclaredTableDoesNotSelectUnrelatedMatrix() {
        try (SameDiff graph = SameDiff.create()) {
            graph.var("token_embd.weight", Nd4j.ones(DataType.FLOAT, 4));
            graph.var("unrelated.weight", Nd4j.ones(DataType.FLOAT, 8, 8));
            assertThrows(IllegalArgumentException.class, () -> GenerationPipeline.extractEmbeddingTable(graph));
        }
    }
}
