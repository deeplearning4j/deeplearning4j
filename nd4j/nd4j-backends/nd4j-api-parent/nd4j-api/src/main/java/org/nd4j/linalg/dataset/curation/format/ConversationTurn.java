package org.nd4j.linalg.dataset.curation.format;

import java.io.Serializable;
import java.util.Objects;

/**
 * Represents a single turn in a conversation with a role and content.
 */
public class ConversationTurn implements Serializable {
    private static final long serialVersionUID = 1L;

    private final String role;
    private final String content;

    public ConversationTurn(String role, String content) {
        this.role = Objects.requireNonNull(role, "role must not be null");
        this.content = Objects.requireNonNull(content, "content must not be null");
    }

    public static ConversationTurn system(String content) {
        return new ConversationTurn("system", content);
    }

    public static ConversationTurn user(String content) {
        return new ConversationTurn("user", content);
    }

    public static ConversationTurn assistant(String content) {
        return new ConversationTurn("assistant", content);
    }

    public String getRole() {
        return role;
    }

    public String getContent() {
        return content;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ConversationTurn)) return false;
        ConversationTurn that = (ConversationTurn) o;
        return role.equals(that.role) && content.equals(that.content);
    }

    @Override
    public int hashCode() {
        return Objects.hash(role, content);
    }

    @Override
    public String toString() {
        return "ConversationTurn{role='" + role + "', content='" + content + "'}";
    }
}
