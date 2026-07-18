package org.eclipse.deeplearning4j.llm.finetune;

import java.util.Objects;

/** One ordered chat message in a generated fine-tuning example. */
public class FineTuneMessage {
    private String role;
    private String content;

    public FineTuneMessage() {}

    public FineTuneMessage(String role, String content) {
        this.role = role;
        this.content = content;
    }

    public void validate() {
        if (!"system".equals(role) && !"user".equals(role) && !"assistant".equals(role)) {
            throw new IllegalArgumentException("Unsupported message role: " + role);
        }
        if (content == null || content.trim().isEmpty()) {
            throw new IllegalArgumentException("Message content is required");
        }
    }

    public String getRole() { return role; }
    public void setRole(String role) { this.role = role; }
    public String getContent() { return content; }
    public void setContent(String content) { this.content = content; }

    @Override public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof FineTuneMessage)) return false;
        FineTuneMessage that = (FineTuneMessage) o;
        return Objects.equals(role, that.role) && Objects.equals(content, that.content);
    }

    @Override public int hashCode() { return Objects.hash(role, content); }
}
