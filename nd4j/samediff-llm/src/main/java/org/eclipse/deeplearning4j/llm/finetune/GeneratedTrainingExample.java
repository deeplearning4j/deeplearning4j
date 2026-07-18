package org.eclipse.deeplearning4j.llm.finetune;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Persisted result of offline teacher generation, ready for masked supervised fine-tuning. */
public class GeneratedTrainingExample {
    public static final String CURRENT_SCHEMA = "samediff-generated-example-v1";

    private String schemaVersion = CURRENT_SCHEMA;
    private String id;
    private String systemPrompt;
    private String prompt;
    private String response;
    private List<FineTuneMessage> messages = new ArrayList<>();
    private Map<String, Object> context = new LinkedHashMap<>();
    private Map<String, Object> metadata = new LinkedHashMap<>();

    public void validate() {
        if (!CURRENT_SCHEMA.equals(schemaVersion)) throw new IllegalArgumentException("Unsupported generated example schema: " + schemaVersion);
        if (id == null || id.trim().isEmpty()) throw new IllegalArgumentException("Generated example id is required");
        if (messages != null && !messages.isEmpty()) {
            boolean hasAssistant = false;
            for (FineTuneMessage message : messages) {
                if (message == null) throw new IllegalArgumentException("Generated example contains a null message");
                message.validate();
                hasAssistant |= "assistant".equals(message.getRole());
            }
            if (!hasAssistant) throw new IllegalArgumentException("Generated example needs at least one assistant message");
        } else {
            if (prompt == null || prompt.trim().isEmpty()) throw new IllegalArgumentException("Generated example prompt is required");
            if (response == null || response.trim().isEmpty()) throw new IllegalArgumentException("Generated example response is required");
        }
    }

    /** Returns ordered messages, adapting legacy single-turn prompt/response records when needed. */
    public List<FineTuneMessage> effectiveMessages() {
        if (messages != null && !messages.isEmpty()) return new ArrayList<>(messages);
        List<FineTuneMessage> result = new ArrayList<>();
        if (systemPrompt != null && !systemPrompt.trim().isEmpty()) {
            result.add(new FineTuneMessage("system", systemPrompt));
        }
        result.add(new FineTuneMessage("user", prompt));
        result.add(new FineTuneMessage("assistant", response));
        return result;
    }

    public String getSchemaVersion() { return schemaVersion; }
    public void setSchemaVersion(String schemaVersion) { this.schemaVersion = schemaVersion; }
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getSystemPrompt() { return systemPrompt; }
    public void setSystemPrompt(String systemPrompt) { this.systemPrompt = systemPrompt; }
    public String getPrompt() { return prompt; }
    public void setPrompt(String prompt) { this.prompt = prompt; }
    public String getResponse() { return response; }
    public void setResponse(String response) { this.response = response; }
    public List<FineTuneMessage> getMessages() { return messages; }
    public void setMessages(List<FineTuneMessage> messages) { this.messages = messages; }
    public Map<String, Object> getContext() { return context; }
    public void setContext(Map<String, Object> context) { this.context = context; }
    public Map<String, Object> getMetadata() { return metadata; }
    public void setMetadata(Map<String, Object> metadata) { this.metadata = metadata; }
}
