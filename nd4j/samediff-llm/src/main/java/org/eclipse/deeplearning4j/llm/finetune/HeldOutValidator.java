package org.eclipse.deeplearning4j.llm.finetune;

/**
 * Pluggable validator for held-out generation evaluation.
 *
 * <p>Analogous to {@link TeacherOutputValidator} but operates on a
 * {@link GeneratedTrainingExample} and the model-generated text rather than
 * a {@link TeacherExampleRequest}, avoiding coupling to domain schemas.</p>
 */
@FunctionalInterface
public interface HeldOutValidator {

    TeacherValidationResult validate(GeneratedTrainingExample example, String generatedText);
}
