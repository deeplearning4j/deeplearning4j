/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.vlm.output.protocol;

import lombok.Builder;
import lombok.Data;

import java.util.Collections;
import java.util.Map;

/** Protocol-aware raw, structured, and rendered forms of one generation. */
@Data
@Builder
public class VlmProtocolOutput {
    private final String protocolId;
    private final String nativeFormat;
    private final String rawText;
    private final String renderedText;
    private final Object structured;
    private final VlmCompletion completion;
    @Builder.Default private final Map<String, Object> metadata = Collections.emptyMap();
}
