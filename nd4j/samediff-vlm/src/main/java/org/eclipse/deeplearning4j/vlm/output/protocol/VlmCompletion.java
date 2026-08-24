/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.vlm.output.protocol;

import lombok.Builder;
import lombok.Data;

/** Structural assessment kept separate from token-level generation finish reason. */
@Data
@Builder
public class VlmCompletion {
    private final boolean complete;
    private final boolean usable;
    private final String diagnostic;
}
