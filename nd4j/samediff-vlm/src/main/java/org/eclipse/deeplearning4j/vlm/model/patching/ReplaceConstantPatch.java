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

package org.eclipse.deeplearning4j.vlm.model.patching;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.function.Supplier;
import java.util.regex.Pattern;

/**
 * A patch that replaces a named constant/initializer with a new value.
 *
 * <p>The variable can be matched by exact name or by regex pattern.
 * The replacement value is provided by a {@link Supplier} so it can
 * be lazily created only when the patch is actually applied.</p>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * // Replace a constant by exact name
 * SameDiffGraphPatch patch = ReplaceConstantPatch.forExactName(
 *     "position_ids",
 *     () -> Nd4j.arange(1024).reshape(1, 1024).castTo(DataType.INT64)
 * );
 *
 * // Replace constants matching a pattern
 * SameDiffGraphPatch patch = ReplaceConstantPatch.forPattern(
 *     ".*position_ids.*",
 *     () -> Nd4j.arange(1024).reshape(1, 1024).castTo(DataType.INT64)
 * );
 * }</pre>
 *
 * Adam Gibson
 */
@Slf4j
public class ReplaceConstantPatch implements SameDiffGraphPatch {

    private final String patchName;
    private final String variableNamePattern;
    private final boolean isRegex;
    private final Supplier<INDArray> replacementSupplier;

    /**
     * Create a patch that replaces a constant variable.
     *
     * @param patchName human-readable name for this patch
     * @param variableNamePattern the variable name (exact) or regex pattern to match
     * @param isRegex if true, variableNamePattern is treated as a regex
     * @param replacementSupplier supplier that creates the replacement INDArray
     */
    public ReplaceConstantPatch(String patchName, String variableNamePattern,
                                boolean isRegex, Supplier<INDArray> replacementSupplier) {
        this.patchName = patchName;
        this.variableNamePattern = variableNamePattern;
        this.isRegex = isRegex;
        this.replacementSupplier = replacementSupplier;
    }

    /**
     * Create a patch for an exact variable name.
     *
     * @param variableName the exact variable name to replace
     * @param replacementSupplier supplier for the replacement value
     * @return the patch
     */
    public static ReplaceConstantPatch forExactName(String variableName,
                                                    Supplier<INDArray> replacementSupplier) {
        return new ReplaceConstantPatch(
                "replace-constant-" + variableName,
                variableName,
                false,
                replacementSupplier
        );
    }

    /**
     * Create a patch for variables matching a regex pattern.
     *
     * @param pattern the regex pattern to match variable names
     * @param replacementSupplier supplier for the replacement value
     * @return the patch
     */
    public static ReplaceConstantPatch forPattern(String pattern,
                                                  Supplier<INDArray> replacementSupplier) {
        return new ReplaceConstantPatch(
                "replace-constant-pattern-" + pattern,
                pattern,
                true,
                replacementSupplier
        );
    }

    @Override
    public String name() {
        return patchName;
    }

    @Override
    public String description() {
        if (isRegex) {
            return "Replace constant(s) matching pattern '" + variableNamePattern + "' with new value";
        }
        return "Replace constant '" + variableNamePattern + "' with new value";
    }

    @Override
    public boolean appliesTo(SameDiff graph) {
        if (isRegex) {
            Pattern pattern = Pattern.compile(variableNamePattern);
            for (SDVariable var : graph.variables()) {
                if (var.isConstant() && pattern.matcher(var.name()).matches()) {
                    return true;
                }
            }
            return false;
        } else {
            SDVariable var = graph.getVariable(variableNamePattern);
            return var != null && var.isConstant();
        }
    }

    @Override
    public boolean apply(SameDiff graph) {
        INDArray replacement = replacementSupplier.get();
        boolean anyApplied = false;

        if (isRegex) {
            Pattern pattern = Pattern.compile(variableNamePattern);
            for (SDVariable var : graph.variables()) {
                if (var.isConstant() && pattern.matcher(var.name()).matches()) {
                    INDArray oldArr = var.getArr();
                    log.info("Replacing constant '{}': shape {} -> shape {}",
                            var.name(),
                            oldArr != null ? Arrays.toString(oldArr.shape()) : "null",
                            Arrays.toString(replacement.shape()));
                    graph.setArrayForVariable(var.name(), replacement.dup());
                    anyApplied = true;
                }
            }
        } else {
            SDVariable var = graph.getVariable(variableNamePattern);
            if (var != null && var.isConstant()) {
                INDArray oldArr = var.getArr();
                log.info("Replacing constant '{}': shape {} -> shape {}",
                        var.name(),
                        oldArr != null ? Arrays.toString(oldArr.shape()) : "null",
                        Arrays.toString(replacement.shape()));
                graph.setArrayForVariable(var.name(), replacement);
                anyApplied = true;
            } else {
                log.warn("Variable '{}' not found or not a constant", variableNamePattern);
            }
        }

        return anyApplied;
    }
}
