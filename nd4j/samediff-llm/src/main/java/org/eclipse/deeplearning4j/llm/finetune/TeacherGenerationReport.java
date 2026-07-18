package org.eclipse.deeplearning4j.llm.finetune;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Summary of a resumable offline teacher generation job. */
public final class TeacherGenerationReport {
    private final List<GeneratedTrainingExample> accepted = new ArrayList<>();
    private final List<TeacherGenerationFailure> rejected = new ArrayList<>();
    private int skipped;
    private int fallbackCount;

    void accept(GeneratedTrainingExample example) { accepted.add(example); }
    void acceptFallback(GeneratedTrainingExample example) { accepted.add(example); fallbackCount++; }
    void reject(String id, String reason) { rejected.add(new TeacherGenerationFailure(id, reason)); }
    void skip() { skipped++; }

    public List<GeneratedTrainingExample> getAccepted() {
        return Collections.unmodifiableList(accepted);
    }

    public List<TeacherGenerationFailure> getRejected() {
        return Collections.unmodifiableList(rejected);
    }

    public int getSkipped() { return skipped; }
    public int getFallbackCount() { return fallbackCount; }
    public int getTeacherAcceptedCount() { return accepted.size() - fallbackCount; }
}
