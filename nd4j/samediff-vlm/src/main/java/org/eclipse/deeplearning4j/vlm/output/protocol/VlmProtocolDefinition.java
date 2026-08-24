/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.vlm.output.protocol;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Declarative contents of {@code vlm-output-protocol.json}. */
@Data
@NoArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
public class VlmProtocolDefinition {
    private int schemaVersion = 1;
    private String defaultProtocol;
    private Map<String, Protocol> protocols = new LinkedHashMap<>();

    @Data @NoArgsConstructor @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Protocol {
        private String provider = "builtin.plain";
        private String defaultTask = "default";
        private Map<String, Task> tasks = new LinkedHashMap<>();
        private Termination termination = new Termination();
        private Completion completion = new Completion();
        private Output output = new Output();
    }

    @Data @NoArgsConstructor @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Task {
        private List<String> aliases;
        private String prompt;
        private String framing = "CHAT_TEMPLATE";
    }

    @Data @NoArgsConstructor @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Termination {
        private boolean inheritModelEos = true;
        private boolean inheritChatTemplateStops = true;
        private List<Stop> sequences;
    }

    @Data @NoArgsConstructor @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Stop {
        private String id;
        private String kind = "STOP_SEQUENCE";
        private String text;
        private int[] tokenIds;
        private String retention = "DROP_MATCH";
    }

    @Data @NoArgsConstructor @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Completion {
        private String mode = "NON_EMPTY";
        private boolean required;
        private String rootElement;
    }

    @Data @NoArgsConstructor @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Output {
        private String nativeFormat = "PLAIN_TEXT";
        private Map<String, String> renderers = new LinkedHashMap<>();
    }
}
