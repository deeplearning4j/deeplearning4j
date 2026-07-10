package org.eclipse.deeplearning4j.llm.finetune;

import java.util.LinkedHashMap;
import java.util.Map;

/** Persisted result of offline teacher generation, ready for masked supervised fine-tuning. */
public class GeneratedTrainingExample {
    public static final String CURRENT_SCHEMA = "samediff-generated-example-v1";

    private String schemaVersion = CURRENT_SCHEMA;
    private String id;
    private String systemPrompt;
    private String prompt;
    private String response;
    private Map<String, Object> context = new LinkedHashMap<>();
    private Map<String, Object> metadata = new LinkedHashMap<>();

    public void validate() {
        if (!CURRENT_SCHEMA.equals(schemaVersion)) throw new IllegalArgumentException("Unsupported generated example schema: " + schemaVersion);
        if (id == null || id.trim().isEmpty()) throw new IllegalArgumentException("Generated example id is required");
        if (prompt == null || prompt.trim().isEmpty()) throw new IllegalArgumentException("Generated example prompt is required");
        if (response == null || response.trim().isEmpty()) throw new IllegalArgumentException("Generated example response is required");
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
    public Map<String, Object> getContext() { return context; }
    public void setContext(Map<String, Object> context) { this.context = context; }
    public Map<String, Object> getMetadata() { return metadata; }
    public void setMetadata(Map<String, Object> metadata) { this.metadata = metadata; }
}
