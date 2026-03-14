/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.samediff;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.config.DistillationConfig;
import org.nd4j.autodiff.samediff.config.DistillationConfig.DistillationType;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.loss.AttentionDistillationLoss;
import org.nd4j.linalg.api.ops.impl.loss.DistillationKLLoss;
import org.nd4j.linalg.api.ops.impl.loss.FeatureDistillationLoss;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

/**
 * Knowledge Distillation Trainer for SameDiff models.
 * <p>
 * Orchestrates teacher-student training with support for:
 * <ul>
 *   <li>Standard Hinton-style KD (logit-level KL divergence)</li>
 *   <li>Feature-level distillation (intermediate layer matching)</li>
 *   <li>Attention distillation (attention map matching)</li>
 *   <li>Combined loss weighting</li>
 *   <li>Progressive distillation (temperature annealing)</li>
 *   <li>Online self-distillation</li>
 * </ul>
 *
 * <p>Example usage:</p>
 * <pre>
 * DistillationConfig config = DistillationConfig.logitKD("student_logits", "teacher_logits", 4.0, 0.5);
 *
 * DistillationTrainer trainer = new DistillationTrainer(studentModel, teacherModel, config);
 *
 * // Single step distillation
 * Map&lt;String, INDArray&gt; inputs = new HashMap&lt;&gt;();
 * inputs.put("input", inputData);
 * inputs.put("labels", labelData);
 * double loss = trainer.trainStep(inputs);
 * </pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class DistillationTrainer {

    @Getter
    private final SameDiff studentModel;

    @Getter
    private final SameDiff teacherModel;

    @Getter
    private final DistillationConfig config;

    private int totalSteps = 0;
    private int totalEpochs = 0;

    /**
     * Create a distillation trainer.
     *
     * @param studentModel The student model (will be trained)
     * @param teacherModel The teacher model (frozen, used for generating soft targets)
     * @param config       The distillation configuration
     */
    public DistillationTrainer(SameDiff studentModel, SameDiff teacherModel, DistillationConfig config) {
        Preconditions.checkNotNull(studentModel, "Student model must not be null");
        Preconditions.checkNotNull(teacherModel, "Teacher model must not be null");
        Preconditions.checkNotNull(config, "Distillation config must not be null");

        this.studentModel = studentModel;
        this.teacherModel = teacherModel;
        this.config = config;
    }

    /**
     * Perform a single distillation training step.
     *
     * @param inputs Input data (will be fed to both teacher and student)
     * @return The total distillation loss
     */
    public double trainStep(Map<String, INDArray> inputs) {
        totalSteps++;

        // 1. Teacher forward pass (no gradient tracking needed)
        Map<String, INDArray> teacherOutputs = getTeacherOutputs(inputs);

        // 2. Compute distillation loss based on configuration
        double loss = 0.0;

        switch (config.getDistillationType()) {
            case LOGIT_KD:
                loss = computeLogitKDLoss(inputs, teacherOutputs);
                break;

            case FEATURE_KD:
                loss = computeFeatureKDLoss(inputs, teacherOutputs);
                break;

            case ATTENTION_KD:
                loss = computeAttentionKDLoss(inputs, teacherOutputs);
                break;

            case COMBINED:
                loss = computeCombinedLoss(inputs, teacherOutputs);
                break;
        }

        return loss;
    }

    /**
     * Get teacher outputs for the given inputs.
     */
    private Map<String, INDArray> getTeacherOutputs(Map<String, INDArray> inputs) {
        // Collect all teacher variable names we need
        Set<String> teacherVarNames = new LinkedHashSet<>();

        if (config.getTeacherLogitVariable() != null) {
            teacherVarNames.add(config.getTeacherLogitVariable());
        }
        if (config.getFeatureLayerMappings() != null) {
            teacherVarNames.addAll(config.getFeatureLayerMappings().values());
        }
        if (config.getAttentionLayerMappings() != null) {
            teacherVarNames.addAll(config.getAttentionLayerMappings().values());
        }

        if (teacherVarNames.isEmpty()) {
            return Collections.emptyMap();
        }

        return teacherModel.output(inputs, teacherVarNames.toArray(new String[0]));
    }

    /**
     * Compute logit-level KL divergence distillation loss.
     */
    private double computeLogitKDLoss(Map<String, INDArray> inputs, Map<String, INDArray> teacherOutputs) {
        String studentLogitVar = config.getStudentLogitVariable();
        String teacherLogitVar = config.getTeacherLogitVariable();

        // Get student output
        Map<String, INDArray> studentOutputs = studentModel.output(inputs, studentLogitVar);
        INDArray studentLogits = studentOutputs.get(studentLogitVar);
        INDArray teacherLogits = teacherOutputs.get(teacherLogitVar);

        double effectiveTemp = config.getEffectiveTemperature(getProgress());

        // Compute KL distillation loss
        DistillationKLLoss klLoss = new DistillationKLLoss(studentLogits, teacherLogits,
            effectiveTemp, config.getAlpha());
        INDArray lossResult = Nd4j.exec(klLoss)[0];

        return lossResult.getDouble(0);
    }

    /**
     * Compute feature-level distillation loss.
     */
    private double computeFeatureKDLoss(Map<String, INDArray> inputs, Map<String, INDArray> teacherOutputs) {
        Map<String, String> featureMappings = config.getFeatureLayerMappings();
        double totalLoss = 0.0;

        // Get student feature outputs
        String[] studentVars = featureMappings.keySet().toArray(new String[0]);
        Map<String, INDArray> studentOutputs = studentModel.output(inputs, studentVars);

        for (Map.Entry<String, String> entry : featureMappings.entrySet()) {
            String studentVar = entry.getKey();
            String teacherVar = entry.getValue();

            INDArray studentFeatures = studentOutputs.get(studentVar);
            INDArray teacherFeatures = teacherOutputs.get(teacherVar);

            FeatureDistillationLoss featureLoss = new FeatureDistillationLoss(studentFeatures, teacherFeatures);
            INDArray lossResult = Nd4j.exec(featureLoss)[0];
            totalLoss += lossResult.getDouble(0);
        }

        return totalLoss / featureMappings.size();
    }

    /**
     * Compute attention-level distillation loss.
     */
    private double computeAttentionKDLoss(Map<String, INDArray> inputs, Map<String, INDArray> teacherOutputs) {
        Map<String, String> attentionMappings = config.getAttentionLayerMappings();
        double totalLoss = 0.0;

        String[] studentVars = attentionMappings.keySet().toArray(new String[0]);
        Map<String, INDArray> studentOutputs = studentModel.output(inputs, studentVars);

        for (Map.Entry<String, String> entry : attentionMappings.entrySet()) {
            String studentVar = entry.getKey();
            String teacherVar = entry.getValue();

            INDArray studentAttn = studentOutputs.get(studentVar);
            INDArray teacherAttn = teacherOutputs.get(teacherVar);

            AttentionDistillationLoss attnLoss = new AttentionDistillationLoss(studentAttn, teacherAttn);
            INDArray lossResult = Nd4j.exec(attnLoss)[0];
            totalLoss += lossResult.getDouble(0);
        }

        return totalLoss / attentionMappings.size();
    }

    /**
     * Compute combined distillation loss with weighted components.
     */
    private double computeCombinedLoss(Map<String, INDArray> inputs, Map<String, INDArray> teacherOutputs) {
        double totalLoss = 0.0;

        // Logit KD component
        if (config.getStudentLogitVariable() != null && config.getTeacherLogitVariable() != null) {
            totalLoss += config.getLogitLossWeight() * computeLogitKDLoss(inputs, teacherOutputs);
        }

        // Feature KD component
        if (config.getFeatureLayerMappings() != null && !config.getFeatureLayerMappings().isEmpty()) {
            totalLoss += config.getFeatureLossWeight() * computeFeatureKDLoss(inputs, teacherOutputs);
        }

        // Attention KD component
        if (config.getAttentionLayerMappings() != null && !config.getAttentionLayerMappings().isEmpty()) {
            totalLoss += config.getAttentionLossWeight() * computeAttentionKDLoss(inputs, teacherOutputs);
        }

        return totalLoss;
    }

    /**
     * Get the training progress as a fraction from 0.0 to 1.0.
     */
    private double getProgress() {
        // Simple step-based progress for temperature annealing
        // Could be made configurable with total steps
        return Math.min(1.0, totalSteps / 10000.0);
    }

    /**
     * Create a self-distillation trainer where the teacher is a snapshot of the student.
     * The teacher is refreshed periodically.
     *
     * @param model  The model (serves as both student and initial teacher)
     * @param config Distillation configuration
     * @return A DistillationTrainer for self-distillation
     */
    public static DistillationTrainer selfDistillation(SameDiff model, DistillationConfig config) {
        SameDiff teacherSnapshot = model.dup();
        return new DistillationTrainer(model, teacherSnapshot, config);
    }

    /**
     * Refresh the teacher model from the current student state (for self-distillation).
     * Uses exponential moving average (EMA) of student weights.
     *
     * @param emaDecay EMA decay factor (e.g., 0.999)
     */
    public void refreshTeacherEMA(double emaDecay) {
        for (SDVariable studentVar : studentModel.variables()) {
            if (studentVar.getVariableType() == VariableType.VARIABLE) {
                SDVariable teacherVar = teacherModel.getVariable(studentVar.name());
                if (teacherVar != null && teacherVar.getArr() != null && studentVar.getArr() != null) {
                    // teacher = decay * teacher + (1 - decay) * student
                    INDArray teacherArr = teacherVar.getArr();
                    INDArray studentArr = studentVar.getArr();
                    teacherArr.muli(emaDecay).addi(studentArr.mul(1.0 - emaDecay));
                }
            }
        }
    }
}
