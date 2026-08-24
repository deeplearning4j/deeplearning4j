/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.vlm.output.protocol;

import lombok.Builder;
import lombok.Data;

import java.util.Collections;
import java.util.Map;

/** Per-generation selection layered over the model package's protocol manifest. */
@Data
@Builder
public class VlmProtocolRequest {
    private final String protocolId;
    private final String task;
    private final String promptOverride;
    @Builder.Default private final VlmRenderFormat renderFormat = VlmRenderFormat.RAW;
    @Builder.Default private final Map<String, Object> options = Collections.emptyMap();

    public static VlmProtocolRequest raw(String prompt) {
        return builder().promptOverride(prompt).renderFormat(VlmRenderFormat.RAW).build();
    }
}
