package org.eclipse.deeplearning4j.llm.finetune;

/** A rejected or failed request from an offline teacher generation job. */
public final class TeacherGenerationFailure {
    private final String requestId;
    private final String reason;

    public TeacherGenerationFailure(String requestId, String reason) {
        this.requestId = requestId;
        this.reason = reason;
    }

    public String getRequestId() { return requestId; }
    public String getReason() { return reason; }
}
