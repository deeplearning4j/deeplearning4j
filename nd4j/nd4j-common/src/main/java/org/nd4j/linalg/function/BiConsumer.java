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

package org.nd4j.linalg.function;

/**
 * BiConsumer is an operation that accepts two arguments and returns no result.
 *
 * @param <T> Type of first argument
 * @param <U> Type of second argument
 */
public interface BiConsumer<T, U> {

    /**
     * Perform the operation on the given arguments
     *
     * @param t First input
     * @param u Second input
     */
    void accept(T t, U u);

}
