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
package org.eclipse.deeplearning4j.tests.extensions;

import org.junit.jupiter.api.extension.ConditionEvaluationResult;
import org.junit.jupiter.api.extension.ExecutionCondition;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.nd4j.common.tests.tags.TagNames;

import java.lang.reflect.Field;
import java.util.HashSet;
import java.util.List;
import java.util.Set;


/**
 * This extension disables any tests for gpu that are large resources
 * or long. GPU tests should only need to test execution on the gpu.
 *
 * @author Adam Gibson
 */
public class TFGraphCheckerExtension implements ExecutionCondition {

    public final static Set<String> invalidResourcesTags = new HashSet<>(){{
        add(TagNames.LARGE_RESOURCES);
        add(TagNames.DOWNLOADS);
        add(TagNames.LONG_TEST);
        add(TagNames.MULTI_THREADED);
        add(TagNames.SPARK);
        add(TagNames.PYTHON);
    }};

    private static volatile List<String> executeOnlyModels;
    private static volatile boolean loadAttempted;

    @SuppressWarnings("unchecked")
    private static List<String> getExecuteOnlyModels() {
        if (!loadAttempted) {
            synchronized (TFGraphCheckerExtension.class) {
                if (!loadAttempted) {
                    try {
                        Class<?> baseClass = Class.forName(
                                "org.eclipse.deeplearning4j.frameworkimport.tensorflow.models.TestTFGraphAllSameDiffPartitionedBase");
                        Field field = baseClass.getField("EXECUTE_ONLY_MODELS");
                        executeOnlyModels = (List<String>) field.get(null);
                    } catch (ClassNotFoundException | NoClassDefFoundError | NoSuchFieldException | IllegalAccessException e) {
                        executeOnlyModels = null;
                    }
                    loadAttempted = true;
                }
            }
        }
        return executeOnlyModels;
    }

    @Override
    public ConditionEvaluationResult evaluateExecutionCondition(ExtensionContext context) {
        List<String> models = getExecuteOnlyModels();
        if (models == null) {
            return ConditionEvaluationResult.enabled("TFGraphCheckerExtension - TF classes not available");
        }

        // When EXECUTE_ONLY_MODELS is non-empty, filter to only those models
        if (!models.isEmpty() && context.getTestClass().isPresent()
                && context.getTestClass().get().getName().contains("TFGraph")) {
            String displayName = context.getDisplayName();
            for (String model : models) {
                if (displayName.contains(model)) {
                    return ConditionEvaluationResult.enabled("TFGraphCheckerExtension - model in EXECUTE_ONLY_MODELS");
                }
            }
            return ConditionEvaluationResult.disabled("TFGraphCheckerExtension - model not in EXECUTE_ONLY_MODELS list");
        }

        return ConditionEvaluationResult.enabled("TFGraphCheckerExtension");
    }
}
