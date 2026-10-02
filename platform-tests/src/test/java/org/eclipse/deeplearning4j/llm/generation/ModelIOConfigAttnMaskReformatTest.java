/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.llm.generation;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.NameScope;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * ModelIOConfig.discover() names the attention bias the attn_mask_reformat subgraph hands to the
 * layers. Optimized graphs no longer have the canonical Tile output, and the subgraph's internal
 * tensors carry its name and an "output" suffix too. One of those taken for the mask is overridden
 * at prefill and written into by the native decode loop every step (SmolDocling's decoder handed it
 * a folded ConstantOfShape shape vector).
 */
class ModelIOConfigAttnMaskReformatTest {
    private static final String SCOPE = "/model/attn_mask_reformat";

    @Test
    void discoversTheBiasTheLayersRead() {
        SameDiff sd = SameDiff.create();
        SDVariable embeds = sd.placeHolder("inputs_embeds", DataType.FLOAT, 1, 4, 8);
        SDVariable bias;
        try (NameScope ignored = sd.withNameScope(SCOPE)) {
            SDVariable shape = sd.constant("ConstantOfShape/output_0", Nd4j.createFromArray(1L, 1L, 4L, 4L));
            bias = shape.castTo("Expand/output_0", DataType.FLOAT);
        }
        embeds.add("layer_input", bias);

        assertEquals(bias.name(), ModelIOConfig.discover(sd).getAttnMaskReformatOutput());
    }

    @Test
    void findsNoBiasWhenOnlyTheSubgraphsShapeConstantRemains() {
        // A decoder optimized for causal_mask keeps only a folded shape vector of the subgraph.
        SameDiff sd = SameDiff.create();
        SDVariable embeds = sd.placeHolder("inputs_embeds", DataType.FLOAT, 1, 4, 8);
        SDVariable shape;
        try (NameScope ignored = sd.withNameScope(SCOPE)) {
            shape = sd.constant("attn_mask_subgraph/ConstantOfShape/output_0",
                    Nd4j.createFromArray(1L, 1L, 1L, 1L));
        }
        embeds.add("layer_input", shape.castTo("shape_as_float", DataType.FLOAT));

        assertNull(ModelIOConfig.discover(sd).getAttnMaskReformatOutput());
    }
}
