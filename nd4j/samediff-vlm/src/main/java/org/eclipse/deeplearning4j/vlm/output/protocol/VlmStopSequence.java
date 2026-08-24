/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.vlm.output.protocol;

import lombok.Builder;
import lombok.Data;

/** Typed terminator rule resolved to tokenizer IDs for one model package. */
@Data
@Builder
public class VlmStopSequence {
    public enum Kind { MODEL_EOS, END_OF_TURN, STOP_SEQUENCE, STRUCTURAL_COMPLETE, QUALITY_STOP }
    public enum Retention { KEEP_MATCH, DROP_MATCH }

    private final String id;
    @Builder.Default private final Kind kind = Kind.STOP_SEQUENCE;
    @Builder.Default private final Retention retention = Retention.DROP_MATCH;
    private final int[] tokenIds;

    public int[] getTokenIds() { return tokenIdsCopy(); }
    public int[] tokenIdsCopy() { return tokenIds == null ? null : tokenIds.clone(); }
}
