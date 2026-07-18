package org.eclipse.deeplearning4j.llm.finetune;

/** Backend-independent accepted or rejected token-preparation outcome. */
public final class TokenPreparationResult {
    public enum Status { ACCEPTED, TRUNCATED, REJECTED }

    private final String exampleId;
    private final Status status;
    private final ResponseMaskedTokens tokens;
    private final String reason;

    private TokenPreparationResult(String id, Status status, ResponseMaskedTokens tokens, String reason) {
        this.exampleId = id; this.status = status; this.tokens = tokens; this.reason = reason;
    }

    public static TokenPreparationResult accepted(String id, ResponseMaskedTokens tokens) {
        return new TokenPreparationResult(id, tokens.isTruncated() ? Status.TRUNCATED : Status.ACCEPTED,
                tokens, null);
    }

    public static TokenPreparationResult rejected(String id, String reason) {
        return new TokenPreparationResult(id, Status.REJECTED, null, reason);
    }

    public String getExampleId() { return exampleId; }
    public Status getStatus() { return status; }
    public ResponseMaskedTokens getTokens() { return tokens; }
    public String getReason() { return reason; }
}
