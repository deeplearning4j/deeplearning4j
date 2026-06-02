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

package org.eclipse.deeplearning4j.llm.editing;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Svd;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Finds refusal directions in a language model's residual stream by comparing
 * activations on harmful vs harmless prompts.
 *
 * <p>The core algorithm:</p>
 * <ol>
 *   <li>Run forward passes on harmful and harmless prompt activations</li>
 *   <li>Compute mean activation per class at each probed layer</li>
 *   <li>Compute refusal direction as normalized difference of means</li>
 *   <li>Rank directions by score (magnitude of mean projection)</li>
 * </ol>
 */
@Slf4j
public class RefusalDirectionFinder {

    /**
     * Find refusal directions in the model by comparing activations on
     * harmful vs harmless inputs.
     *
     * @param model the SameDiff model to analyze
     * @param harmfulActivations activations from harmful prompts at each probe point.
     *        Map from variable name to INDArray of shape [numSamples, hiddenDim].
     * @param harmlessActivations activations from harmless prompts at each probe point.
     *        Map from variable name to INDArray of shape [numSamples, hiddenDim].
     * @param config abliteration configuration
     * @return ranked list of refusal directions, highest score first
     */
    public List<RefusalDirection> findRefusalDirections(
            SameDiff model,
            Map<String, INDArray> harmfulActivations,
            Map<String, INDArray> harmlessActivations,
            AbliterationConfig config) {

        Preconditions.checkArgument(!harmfulActivations.isEmpty(),
                "Harmful activations map must not be empty");
        Preconditions.checkArgument(!harmlessActivations.isEmpty(),
                "Harmless activations map must not be empty");

        List<RefusalDirection> candidates = new ArrayList<>();

        for (Map.Entry<String, INDArray> entry : harmfulActivations.entrySet()) {
            String varName = entry.getKey();
            INDArray harmfulActs = entry.getValue();
            INDArray harmlessActs = harmlessActivations.get(varName);

            if (harmlessActs == null) {
                log.warn("No harmless activations for variable {}, skipping", varName);
                continue;
            }

            INDArray harmfulCast = harmfulActs.castTo(config.getComputeDataType());
            INDArray harmlessCast = harmlessActs.castTo(config.getComputeDataType());

            if (config.isWinsorize()) {
                harmfulCast = winsorize(harmfulCast, config.getWinsorizePercentile());
                harmlessCast = winsorize(harmlessCast, config.getWinsorizePercentile());
            }

            INDArray direction;
            switch (config.getMethod()) {
                case DIFF_IN_MEANS:
                    direction = computeDiffInMeans(harmfulCast, harmlessCast);
                    break;
                case PCA:
                    direction = computePCA(harmfulCast, harmlessCast);
                    break;
                case PROJECTED:
                    direction = computeProjected(harmfulCast, harmlessCast);
                    break;
                default:
                    throw new IllegalArgumentException("Unknown method: " + config.getMethod());
            }

            double score = computeScore(harmfulCast, harmlessCast, direction);
            int layerIndex = extractLayerIndex(varName);
            AbliterationConfig.ResidualStreamPosition position = inferPosition(varName);

            candidates.add(new RefusalDirection(varName, position, direction, score, layerIndex));
        }

        candidates.sort(Comparator.comparingDouble(RefusalDirection::getScore).reversed());
        return candidates;
    }

    /**
     * Identify variable names in the SameDiff graph that correspond to
     * residual stream probe points, based on weight name patterns.
     *
     * @param model the SameDiff model
     * @param config configuration with target weight patterns
     * @return list of variable names suitable for activation collection
     */
    public List<String> identifyProbePoints(SameDiff model, AbliterationConfig config) {
        List<String> probePoints = new ArrayList<>();
        List<Pattern> patterns = new ArrayList<>();
        for (String p : config.getTargetWeightPatterns()) {
            patterns.add(Pattern.compile(p));
        }

        for (SDVariable var : model.variables()) {
            String name = var.name();
            for (Pattern pattern : patterns) {
                if (pattern.matcher(name).matches()) {
                    probePoints.add(name);
                    break;
                }
            }
        }

        log.info("Identified {} probe points in model", probePoints.size());
        return probePoints;
    }

    /**
     * Collect activations from the model for a set of input arrays at specified
     * probe points. Each probe point should be a named variable in the SameDiff graph.
     *
     * @param model the SameDiff model
     * @param inputs map of input variable names to input arrays
     * @param probePointNames variable names to collect activations from
     * @return map from variable name to activation array
     */
    public Map<String, INDArray> collectActivations(
            SameDiff model,
            Map<String, INDArray> inputs,
            List<String> probePointNames) {

        String[] outputNames = probePointNames.toArray(new String[0]);
        Map<String, INDArray> outputs = model.output(inputs, outputNames);
        Map<String, INDArray> result = new LinkedHashMap<>();

        for (String name : probePointNames) {
            INDArray activation = outputs.get(name);
            if (activation != null) {
                // Extract last token position: activation shape is typically [batch, seqLen, hidden]
                if (activation.rank() == 3) {
                    long lastPos = activation.size(1) - 1;
                    activation = activation.get(NDArrayIndex.all(),
                            NDArrayIndex.point(lastPos),
                            NDArrayIndex.all());
                }
                result.put(name, activation.dup());
            }
        }

        return result;
    }

    /**
     * Compute refusal direction via difference-in-means (Arditi et al.).
     * direction = normalize(mean_harmful - mean_harmless)
     */
    private INDArray computeDiffInMeans(INDArray harmful, INDArray harmless) {
        INDArray meanHarmful = harmful.mean(0);
        INDArray meanHarmless = harmless.mean(0);
        INDArray diff = meanHarmful.sub(meanHarmless);
        double norm = diff.norm2Number().doubleValue();
        if (norm > 1e-10) {
            diff.divi(norm);
        }
        return diff;
    }

    /**
     * Compute refusal direction via PCA — top principal component of the
     * centered difference matrix.
     */
    private INDArray computePCA(INDArray harmful, INDArray harmless) {
        // Stack harmful and harmless with labels: harmful centered around +1, harmless around -1
        INDArray combined = Nd4j.concat(0, harmful, harmless);
        INDArray mean = combined.mean(0);
        INDArray centered = combined.subRowVector(mean);

        // SVD of centered data matrix to find top principal component
        // Svd returns [S, U, V] when computeUv=true
        INDArray[] svdResult = Nd4j.exec(new Svd(centered, false, true,
                Svd.DEFAULT_SWITCHNUM));
        INDArray v = svdResult[2]; // V matrix — right singular vectors

        // First column of V is the top principal component
        INDArray topComponent;
        if (v.rank() == 2) {
            topComponent = v.getColumn(0, false).dup();
        } else {
            topComponent = v.dup();
        }

        double norm = topComponent.norm2Number().doubleValue();
        if (norm > 1e-10) {
            topComponent.divi(norm);
        }

        // Ensure direction points from harmless toward harmful
        INDArray meanHarmful = harmful.mean(0);
        INDArray meanHarmless = harmless.mean(0);
        INDArray diff = meanHarmful.sub(meanHarmless);
        double dot = Nd4j.getBlasWrapper().dot(topComponent, diff);
        if (dot < 0) {
            topComponent.negi();
        }

        return topComponent;
    }

    /**
     * Compute refusal direction via projected method (grimjim variant).
     * Decomposes the diff-in-means into components parallel and orthogonal
     * to the harmless mean, keeping only the orthogonal component.
     * This reduces collateral damage to general model capabilities.
     */
    private INDArray computeProjected(INDArray harmful, INDArray harmless) {
        INDArray meanHarmful = harmful.mean(0);
        INDArray meanHarmless = harmless.mean(0);
        INDArray diff = meanHarmful.sub(meanHarmless);

        // Normalize harmless mean to get projection axis
        INDArray harmlessDir = meanHarmless.dup();
        double harmlessNorm = harmlessDir.norm2Number().doubleValue();
        if (harmlessNorm > 1e-10) {
            harmlessDir.divi(harmlessNorm);
        }

        // Project diff onto harmless direction
        double projection = Nd4j.getBlasWrapper().dot(diff, harmlessDir);
        INDArray parallelComponent = harmlessDir.mul(projection);

        // Orthogonal component = diff - parallel
        INDArray orthogonal = diff.sub(parallelComponent);
        double orthNorm = orthogonal.norm2Number().doubleValue();
        if (orthNorm > 1e-10) {
            orthogonal.divi(orthNorm);
        }

        return orthogonal;
    }

    /**
     * Compute ranking score for a direction: mean absolute projection of the
     * difference between harmful and harmless activations onto the direction.
     */
    private double computeScore(INDArray harmful, INDArray harmless, INDArray direction) {
        INDArray meanHarmful = harmful.mean(0);
        INDArray meanHarmless = harmless.mean(0);
        INDArray diff = meanHarmful.sub(meanHarmless);
        return Math.abs(Nd4j.getBlasWrapper().dot(diff, direction));
    }

    /**
     * Extract layer index from a variable name.
     * Looks for patterns like ".layers.5." or ".h.12." or ".model.layers.3."
     */
    public static int extractLayerIndex(String varName) {
        java.util.regex.Matcher m = java.util.regex.Pattern
                .compile("\\.(layers|h|blocks?)\\.(\\d+)\\.")
                .matcher(varName);
        if (m.find()) {
            return Integer.parseInt(m.group(2));
        }
        return -1;
    }

    /**
     * Infer residual stream position from variable name patterns.
     */
    public static AbliterationConfig.ResidualStreamPosition inferPosition(String varName) {
        String lower = varName.toLowerCase();
        if (lower.contains("attn") || lower.contains("attention") ||
                lower.contains("o_proj") || lower.contains("out_proj")) {
            return AbliterationConfig.ResidualStreamPosition.POST_ATTENTION;
        } else if (lower.contains("mlp") || lower.contains("down_proj") ||
                lower.contains("fc_out") || lower.contains("fc2")) {
            return AbliterationConfig.ResidualStreamPosition.POST_MLP;
        } else if (lower.contains("ln") || lower.contains("norm") ||
                lower.contains("layernorm")) {
            return AbliterationConfig.ResidualStreamPosition.PRE_ATTENTION;
        }
        return AbliterationConfig.ResidualStreamPosition.POST_MLP;
    }

    /**
     * Winsorize activations by clipping values beyond the specified percentile.
     */
    private INDArray winsorize(INDArray activations, double percentile) {
        INDArray flat = activations.reshape(-1);
        INDArray sorted = Nd4j.sort(flat, true);
        long n = sorted.length();

        long lowerIdx = Math.round(n * (1.0 - percentile));
        long upperIdx = Math.round(n * percentile);
        lowerIdx = Math.max(0, Math.min(lowerIdx, n - 1));
        upperIdx = Math.max(0, Math.min(upperIdx, n - 1));

        double lowerBound = sorted.getDouble(lowerIdx);
        double upperBound = sorted.getDouble(upperIdx);

        INDArray result = activations.dup();
        // Clip to [lowerBound, upperBound]
        result = Nd4j.math().min(result, Nd4j.scalar(result.dataType(), upperBound));
        result = Nd4j.math().max(result, Nd4j.scalar(result.dataType(), lowerBound));
        return result;
    }
}
