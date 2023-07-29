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
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashSet;
import java.util.Set;

/**
 * This extension disables any tests for gpu that are large resources
 * or long. GPU tests should only need to test execution on the gpu.
 *
 * @author Adam Gibson
 */
public class BackendCheckerExtension  implements ExecutionCondition {

    public final static Set<String> invalidResourcesTags = new HashSet<>(){{
        add(TagNames.LARGE_RESOURCES);
        add(TagNames.DOWNLOADS);
        add(TagNames.LONG_TEST);
        add(TagNames.MULTI_THREADED);
        add(TagNames.SPARK);
        add(TagNames.PYTHON);
    }};

    private boolean hasAny(Set<String> tags, Set<String> invalid) {
        for(String s : invalid) {
            if(tags.contains(s)) {
                return true;
            }
        }
        return false;
    }

    @Override
    public ConditionEvaluationResult evaluateExecutionCondition(ExtensionContext context) {
        if(!Nd4j.getEnvironment().isCPU() && hasAny(invalidResourcesTags,context.getTags())) {
            return ConditionEvaluationResult.disabled("BackendCheckerExtension");
        }
        return ConditionEvaluationResult.enabled("BackendCheckerExtension");
    }
}
