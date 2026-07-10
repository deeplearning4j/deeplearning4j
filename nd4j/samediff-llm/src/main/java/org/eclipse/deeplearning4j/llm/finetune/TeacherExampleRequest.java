package org.eclipse.deeplearning4j.llm.finetune;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/** A domain-neutral structured request from which a teacher produces one target response. */
public class TeacherExampleRequest {
    public static final String CURRENT_SCHEMA = "samediff-teacher-example-v1";

    private String schemaVersion = CURRENT_SCHEMA;
    private String id;
    private String systemPrompt;
    private Map<String, Object> context = new LinkedHashMap<>();
    private Map<String, Object> metadata = new LinkedHashMap<>();

    public void validate() {
        if (!CURRENT_SCHEMA.equals(schemaVersion)) {
            throw new IllegalArgumentException("Unsupported teacher example schema: " + schemaVersion);
        }
        if (id == null || id.trim().isEmpty()) throw new IllegalArgumentException("Teacher example id is required");
        if (context == null) throw new IllegalArgumentException("Teacher example context is required");
    }

    public String getSchemaVersion() { return schemaVersion; }
    public void setSchemaVersion(String schemaVersion) { this.schemaVersion = schemaVersion; }
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getSystemPrompt() { return systemPrompt; }
    public void setSystemPrompt(String systemPrompt) { this.systemPrompt = systemPrompt; }
    public Map<String, Object> getContext() { return context; }
    public void setContext(Map<String, Object> context) { this.context = context; }
    public Map<String, Object> getMetadata() { return metadata; }
    public void setMetadata(Map<String, Object> metadata) { this.metadata = metadata; }

    @Override public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof TeacherExampleRequest)) return false;
        TeacherExampleRequest that = (TeacherExampleRequest) o;
        return Objects.equals(schemaVersion, that.schemaVersion) && Objects.equals(id, that.id)
                && Objects.equals(systemPrompt, that.systemPrompt) && Objects.equals(context, that.context)
                && Objects.equals(metadata, that.metadata);
    }
    @Override public int hashCode() { return Objects.hash(schemaVersion, id, systemPrompt, context, metadata); }
}
