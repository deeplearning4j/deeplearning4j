package org.eclipse.deeplearning4j.llm.finetune;

/** Persisted audit record for a teacher request that could not produce an accepted example. */
public class TeacherGenerationRejection {
    private String requestId;
    private String reason;
    private int attempt;
    private String runId;
    private TeacherExampleRequest request;

    public TeacherGenerationRejection() {}

    public TeacherGenerationRejection(String requestId, String reason, int attempt,
                                      String runId, TeacherExampleRequest request) {
        this.requestId = requestId;
        this.reason = reason;
        this.attempt = attempt;
        this.runId = runId;
        this.request = request;
    }

    public String getRequestId() { return requestId; }
    public void setRequestId(String value) { requestId = value; }
    public String getReason() { return reason; }
    public void setReason(String value) { reason = value; }
    public int getAttempt() { return attempt; }
    public void setAttempt(int value) { attempt = value; }
    public String getRunId() { return runId; }
    public void setRunId(String value) { runId = value; }
    public TeacherExampleRequest getRequest() { return request; }
    public void setRequest(TeacherExampleRequest value) { request = value; }
}
