package org.nd4j.linalg.dataset.curation.format;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Formats raw instruction/response data into training-ready sequences
 * with loss mask generation for selective training.
 */
public class InstructionDataFormatter implements Serializable {
    private static final long serialVersionUID = 1L;

    private final ChatTemplate template;
    private final Map<String, String[]> customPrefixSuffix; // role -> [prefix, suffix]

    private InstructionDataFormatter(Builder builder) {
        this.template = builder.template;
        this.customPrefixSuffix = builder.customPrefixSuffix;
    }

    /**
     * Format a conversation into a training example with loss mask.
     *
     * @param conversation list of conversation turns
     * @return formatted example with text and character-level loss mask
     */
    public FormattedExample format(List<ConversationTurn> conversation) {
        StringBuilder sb = new StringBuilder();
        int[] mask = new int[0];

        for (ConversationTurn turn : conversation) {
            String prefix = getPrefix(turn.getRole());
            String suffix = getSuffix(turn.getRole());
            String content = turn.getContent();

            String segment = prefix + content + suffix;
            int oldLen = mask.length;
            mask = Arrays.copyOf(mask, oldLen + segment.length());

            if ("assistant".equals(turn.getRole())) {
                // Train on assistant content only (not prefix/suffix formatting)
                int contentStart = oldLen + prefix.length();
                int contentEnd = contentStart + content.length();
                Arrays.fill(mask, contentStart, contentEnd, 1);
            }
            // System and user turns: mask stays 0

            sb.append(segment);
        }

        return new FormattedExample(sb.toString(), mask);
    }

    /**
     * Format a simple instruction-response pair.
     */
    public FormattedExample format(String instruction, String response) {
        return format(Arrays.asList(
            ConversationTurn.user(instruction),
            ConversationTurn.assistant(response)
        ));
    }

    /**
     * Format with system prompt.
     */
    public FormattedExample format(String systemPrompt, String instruction, String response) {
        return format(Arrays.asList(
            ConversationTurn.system(systemPrompt),
            ConversationTurn.user(instruction),
            ConversationTurn.assistant(response)
        ));
    }

    private String getPrefix(String role) {
        if (customPrefixSuffix != null && customPrefixSuffix.containsKey(role)) {
            return customPrefixSuffix.get(role)[0];
        }
        return template.getPrefix(role);
    }

    private String getSuffix(String role) {
        if (customPrefixSuffix != null && customPrefixSuffix.containsKey(role)) {
            return customPrefixSuffix.get(role)[1];
        }
        return template.getSuffix(role);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private ChatTemplate template = ChatTemplate.CHATML;
        private Map<String, String[]> customPrefixSuffix;

        public Builder template(ChatTemplate template) {
            this.template = template;
            return this;
        }

        public Builder customPrefixSuffix(Map<String, String[]> custom) {
            this.customPrefixSuffix = custom;
            return this;
        }

        public InstructionDataFormatter build() {
            return new InstructionDataFormatter(this);
        }
    }
}
