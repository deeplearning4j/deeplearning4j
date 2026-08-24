/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.vlm.output.protocol;

import lombok.Builder;
import lombok.Data;

import java.util.Collections;
import java.util.List;

/** Immutable generation plan resolved from package metadata plus one request. */
@Data
@Builder
public class VlmProtocolPlan {
    private final String protocolId;
    private final String task;
    private final String prompt;
    @Builder.Default private final boolean applyChatTemplate = true;
    @Builder.Default private final List<VlmStopSequence> stops = Collections.emptyList();
    @Builder.Default private final boolean inheritModelEos = true;
    @Builder.Default private final boolean inheritChatTemplateStops = true;
    private final String nativeFormat;
    private final boolean structuralCompletionRequired;

    public List<VlmStopSequence> getStops() {
        return java.util.Collections.unmodifiableList(new java.util.ArrayList<>(stops));
    }

    public String cacheKey() {
        StringBuilder key = new StringBuilder(protocolId).append('|').append(task)
                .append('|').append(applyChatTemplate);
        for (VlmStopSequence stop : stops) {
            key.append('|').append(stop.getId()).append(':').append(stop.getKind())
                    .append(':').append(stop.getRetention())
                    .append(':').append(java.util.Arrays.toString(stop.getTokenIds()));
        }
        key.append('|').append(inheritModelEos).append('|').append(inheritChatTemplateStops);
        return key.toString();
    }

    public List<int[]> tokenSequences() {
        List<int[]> sequences = new java.util.ArrayList<>();
        for (VlmStopSequence stop : stops) sequences.add(stop.tokenIdsCopy());
        return java.util.Collections.unmodifiableList(sequences);
    }
}
