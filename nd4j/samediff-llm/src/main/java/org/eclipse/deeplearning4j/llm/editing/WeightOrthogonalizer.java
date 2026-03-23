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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Applies permanent weight modification by orthogonalizing weight matrices
 * against the identified refusal direction(s).
 *
 * <p>For each direction d and target weight matrix W:</p>
 * <pre>
 *   projection = (W @ d) * d^T
 *   W' = W - alpha * projection
 * </pre>
 *
 * <p>This removes the component of W that projects onto the refusal direction,
 * effectively preventing the model from representing refusal behavior.</p>
 */
@Slf4j
public class WeightOrthogonalizer {

    /**
     * Orthogonalize target weight matrices in the model against the given
     * refusal directions.
     *
     * @param model the SameDiff model to modify in-place
     * @param directions the refusal direction(s) to remove
     * @param config abliteration configuration
     * @return list of weight variable names that were modified
     */
    public List<String> orthogonalizeWeights(
            SameDiff model,
            List<RefusalDirection> directions,
            AbliterationConfig config) {

        Preconditions.checkArgument(!directions.isEmpty(),
                "Must provide at least one refusal direction");

        List<Pattern> patterns = new ArrayList<>();
        for (String p : config.getTargetWeightPatterns()) {
            patterns.add(Pattern.compile(p));
        }

        List<String> modifiedNames = new ArrayList<>();
        double alpha = config.getAblationStrength();

        for (SDVariable var : model.variables()) {
            if (var.getVariableType() != VariableType.CONSTANT &&
                    var.getVariableType() != VariableType.VARIABLE) {
                continue;
            }

            String name = var.name();
            if (!matchesAnyPattern(name, patterns)) {
                continue;
            }

            INDArray weight = var.getArr();
            if (weight == null || weight.rank() < 1) {
                continue;
            }

            INDArray modified = weight.dup();
            boolean wasModified = false;

            for (RefusalDirection dir : directions) {
                INDArray d = dir.getDirection().castTo(modified.dataType());

                if (modified.rank() == 2) {
                    // W is [outDim, inDim], d is [hiddenDim]
                    // Need to match d's dimension to either rows or columns
                    modified = orthogonalizeMatrix(modified, d, alpha);
                    wasModified = true;
                } else if (modified.rank() == 1) {
                    // Bias or 1D embedding — project out the direction component
                    if (modified.length() == d.length()) {
                        double proj = Nd4j.getBlasWrapper().dot(modified, d);
                        modified.subi(d.mul(alpha * proj));
                        wasModified = true;
                    }
                }
            }

            if (wasModified) {
                model.setArrayForVariable(name, modified);
                modifiedNames.add(name);
                log.debug("Orthogonalized weight: {}", name);
            }
        }

        log.info("Orthogonalized {} weight matrices against {} direction(s)",
                modifiedNames.size(), directions.size());
        return modifiedNames;
    }

    /**
     * Orthogonalize a 2D weight matrix against a direction vector.
     *
     * <p>For W of shape [outDim, inDim] and d of shape [dim]:</p>
     * <ul>
     *   <li>If dim == inDim: projection = (W @ d) outer d → W' = W - alpha * projection</li>
     *   <li>If dim == outDim: projection = d outer (d^T @ W) → W' = W - alpha * projection</li>
     * </ul>
     */
    private INDArray orthogonalizeMatrix(INDArray weight, INDArray direction, double alpha) {
        long outDim = weight.size(0);
        long inDim = weight.size(1);
        long dirDim = direction.length();

        if (dirDim == inDim) {
            // W @ d gives [outDim], then outer product with d gives [outDim, inDim]
            INDArray dCol = direction.reshape(inDim, 1);
            INDArray wDotD = weight.mmul(dCol); // [outDim, 1]
            INDArray dRow = direction.reshape(1, inDim);
            INDArray projection = wDotD.mmul(dRow); // [outDim, inDim]
            return weight.sub(projection.mul(alpha));
        } else if (dirDim == outDim) {
            // d^T @ W gives [inDim], then outer product with d gives [outDim, inDim]
            INDArray dRow = direction.reshape(1, outDim);
            INDArray dTW = dRow.mmul(weight); // [1, inDim]
            INDArray dCol = direction.reshape(outDim, 1);
            INDArray projection = dCol.mmul(dTW); // [outDim, inDim]
            return weight.sub(projection.mul(alpha));
        } else {
            log.warn("Direction dim {} doesn't match weight dims [{}, {}], skipping",
                    dirDim, outDim, inDim);
            return weight;
        }
    }

    private boolean matchesAnyPattern(String name, List<Pattern> patterns) {
        for (Pattern p : patterns) {
            if (p.matcher(name).matches()) {
                return true;
            }
        }
        return false;
    }
}
