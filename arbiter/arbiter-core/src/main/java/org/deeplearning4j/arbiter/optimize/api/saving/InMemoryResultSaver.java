/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.arbiter.optimize.api.saving;

import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;

import java.io.IOException;
import java.util.Collections;
import java.util.List;

/**
 * A simple class to store optimization results in-memory.
 * Not recommended for large (or a large number of) models.
 */
@NoArgsConstructor
public class InMemoryResultSaver implements ResultSaver {
    @Override
    public ResultReference saveModel(OptimizationResult result, Object modelResult) throws IOException {
        return new InMemoryResult(result, modelResult);
    }

    @Override
    public List<Class<?>> getSupportedCandidateTypes() {
        return Collections.<Class<?>>singletonList(Object.class);
    }

    @Override
    public List<Class<?>> getSupportedModelTypes() {
        return Collections.<Class<?>>singletonList(Object.class);
    }

    @AllArgsConstructor
    private static class InMemoryResult implements ResultReference {
        private OptimizationResult result;
        private Object modelResult;

        @Override
        public OptimizationResult getResult() throws IOException {
            return result;
        }

        @Override
        public Object getResultModel() throws IOException {
            return modelResult;
        }
    }
}
