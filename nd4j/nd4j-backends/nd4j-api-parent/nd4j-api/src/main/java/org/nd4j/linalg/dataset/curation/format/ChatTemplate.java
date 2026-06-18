package org.nd4j.linalg.dataset.curation.format;

/**
 * Built-in chat templates for instruction formatting.
 * Each template defines role prefixes/suffixes for formatting conversations.
 */
public enum ChatTemplate {

    CHATML(
        "<|im_start|>system\n", "<|im_end|>\n",
        "<|im_start|>user\n", "<|im_end|>\n",
        "<|im_start|>assistant\n", "<|im_end|>\n"
    ),

    ALPACA(
        "### Instruction:\n", "\n\n",
        "### Input:\n", "\n\n",
        "### Response:\n", "\n\n"
    ),

    SHAREGPT(
        "<|system|>\n", "</s>\n",
        "<|user|>\n", "</s>\n",
        "<|assistant|>\n", "</s>\n"
    ),

    LLAMA3(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n", "<|eot_id|>",
        "<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|>",
        "<|start_header_id|>assistant<|end_header_id|>\n\n", "<|eot_id|>"
    ),

    PHI(
        "<|system|>\n", "<|end|>\n",
        "<|user|>\n", "<|end|>\n",
        "<|assistant|>\n", "<|end|>\n"
    ),

    RAW("", "\n", "", "\n", "", "\n");

    private final String systemPrefix;
    private final String systemSuffix;
    private final String userPrefix;
    private final String userSuffix;
    private final String assistantPrefix;
    private final String assistantSuffix;

    ChatTemplate(String systemPrefix, String systemSuffix,
                 String userPrefix, String userSuffix,
                 String assistantPrefix, String assistantSuffix) {
        this.systemPrefix = systemPrefix;
        this.systemSuffix = systemSuffix;
        this.userPrefix = userPrefix;
        this.userSuffix = userSuffix;
        this.assistantPrefix = assistantPrefix;
        this.assistantSuffix = assistantSuffix;
    }

    public String getPrefix(String role) {
        switch (role) {
            case "system": return systemPrefix;
            case "user": return userPrefix;
            case "assistant": return assistantPrefix;
            default: return userPrefix;
        }
    }

    public String getSuffix(String role) {
        switch (role) {
            case "system": return systemSuffix;
            case "user": return userSuffix;
            case "assistant": return assistantSuffix;
            default: return userSuffix;
        }
    }

    public String getSystemPrefix() { return systemPrefix; }
    public String getSystemSuffix() { return systemSuffix; }
    public String getUserPrefix() { return userPrefix; }
    public String getUserSuffix() { return userSuffix; }
    public String getAssistantPrefix() { return assistantPrefix; }
    public String getAssistantSuffix() { return assistantSuffix; }
}
