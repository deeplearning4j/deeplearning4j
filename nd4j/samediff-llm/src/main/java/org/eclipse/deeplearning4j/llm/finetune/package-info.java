/**
 * Generic offline sequence-distillation and supervised fine-tuning support.
 *
 * <p>The workflow has two deliberately independent phases:</p>
 * <ol>
 *     <li>Create {@link org.eclipse.deeplearning4j.llm.finetune.TeacherExampleRequest} records,
 *     generate and validate teacher responses with
 *     {@link org.eclipse.deeplearning4j.llm.finetune.OfflineTeacherDataGenerator}, and run
 *     append-only resumable jobs with
 *     {@link org.eclipse.deeplearning4j.llm.finetune.TeacherGenerationJob}.</li>
 *     <li>Convert those examples to exact tokenizer-derived response masks with
 *     {@link org.eclipse.deeplearning4j.llm.finetune.ResponseMaskedDatasetBuilder}, then train
 *     a caller-supplied trainable SameDiff causal-LM graph using
 *     {@link org.eclipse.deeplearning4j.llm.finetune.StudentFineTuningWorkflow}.</li>
 * </ol>
 *
 * <p>Generated examples support both legacy single-turn prompt/response fields and ordered
 * multi-turn {@link org.eclipse.deeplearning4j.llm.finetune.FineTuneMessage} records. Domain
 * state remains structured in the request context map. Applications own the schema, prompt
 * renderer, output validators, and student architecture. The package provides deterministic,
 * group-aware dataset splits, reproducibility manifests, explicit truncation policies,
 * persistent rejection records with optional fallbacks, and held-out generation evaluation.
 * The API does not require the teacher to be a SameDiff model; a local generation pipeline,
 * remote service, replay fixture, or another runtime can implement the teacher function.</p>
 *
 * <p>This package implements hard-target sequence distillation. SameDiff's logit, feature, and
 * attention distillation APIs remain separate for online knowledge distillation.</p>
 */
package org.eclipse.deeplearning4j.llm.finetune;
