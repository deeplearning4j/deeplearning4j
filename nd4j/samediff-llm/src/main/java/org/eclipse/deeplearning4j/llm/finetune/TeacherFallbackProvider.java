package org.eclipse.deeplearning4j.llm.finetune;

/** Produces a deterministic fallback example after teacher generation or validation fails. */
@FunctionalInterface
public interface TeacherFallbackProvider {
    GeneratedTrainingExample fallback(TeacherExampleRequest request, String rejectionReason);
}
