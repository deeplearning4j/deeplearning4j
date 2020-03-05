/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
package org.deeplearning4j.rl4j.observation.transform;

/**
 * The {@link TransformProcess TransformProcess} will call reset() (at the start of an episode) of any step that implement this interface.
 */
public interface ResettableOperation {
    /**
     * Called by TransformProcess when an episode starts. See {@link TransformProcess#reset() TransformProcess.reset()}
     */
    void reset();
}
