package org.eclipse.deeplearning4j.llm.finetune;

/** Domain hook for grounding, safety, format, and quality validation of teacher output. */
@FunctionalInterface
public interface TeacherOutputValidator {
    TeacherValidationResult validate(TeacherExampleRequest request, String response);
}
