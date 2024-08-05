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

import org.eclipse.deeplearning4j.frameworkimport.tensorflow.models.TestTFGraphAllSameDiffPartitioned0;
import org.junit.jupiter.api.extension.ConditionEvaluationResult;
import org.junit.jupiter.api.extension.ExecutionCondition;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.nd4j.common.tests.tags.TagNames;

import java.util.HashSet;
import java.util.Set;

import static org.eclipse.deeplearning4j.frameworkimport.tensorflow.models.TestTFGraphAllSameDiffPartitionedBase.EXECUTE_ONLY_MODELS;


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



    @Override
    public ConditionEvaluationResult evaluateExecutionCondition(ExtensionContext context) {
        if (EXECUTE_ONLY_MODELS.isEmpty() && context.getTestClass().get().getName().contains("TFGraph")
                && !context.getDisplayName().contains("TestTFGraphAllSameDiff")
                && !context.getDisplayName().equals("runTest(Map, Map, String, File)")) {
            if(!EXECUTE_ONLY_MODELS.isEmpty()) {
                if(EXECUTE_ONLY_MODELS.contains(context.getDisplayName()))
                    return ConditionEvaluationResult.enabled("TFGraphCheckerExtension");
                else
                    return ConditionEvaluationResult.disabled("TFGraphCheckerExtension");
            }
        }

        return ConditionEvaluationResult.enabled("TFGraphCheckerExtension");
    }
}
