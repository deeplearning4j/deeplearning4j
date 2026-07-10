package org.eclipse.deeplearning4j.llm.finetune;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Aggregated result from a teacher-output validator. */
public final class TeacherValidationResult {
    private final boolean accepted;
    private final List<String> reasons;

    private TeacherValidationResult(boolean accepted, List<String> reasons) {
        this.accepted = accepted;
        this.reasons = Collections.unmodifiableList(new ArrayList<>(reasons));
    }

    public static TeacherValidationResult accept() { return new TeacherValidationResult(true, Collections.emptyList()); }
    public static TeacherValidationResult reject(String reason) { return new TeacherValidationResult(false, Collections.singletonList(reason)); }
    public static TeacherValidationResult reject(List<String> reasons) { return new TeacherValidationResult(false, reasons); }
    public boolean isAccepted() { return accepted; }
    public List<String> getReasons() { return reasons; }
}
