/*-
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.deeplearning4j.arbiter.optimize.api.saving;

import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;

import java.io.IOException;

/**
 * A simple class to store optimization results in-memory.
 * Not recommended for large (or a large number of) models.
 */
@NoArgsConstructor
public class InMemoryResultSaver<T, M, A> implements ResultSaver<T, M, A> {
    @Override
    public ResultReference<T, M, A> saveModel(OptimizationResult<T, M, A> result) throws IOException {
        return new InMemoryResult<>(result);
    }

    @AllArgsConstructor
    private static class InMemoryResult<T, M, A> implements ResultReference<T, M, A> {
        private OptimizationResult<T, M, A> result;

        @Override
        public OptimizationResult<T, M, A> getResult() throws IOException {
            return result;
        }
    }
}
