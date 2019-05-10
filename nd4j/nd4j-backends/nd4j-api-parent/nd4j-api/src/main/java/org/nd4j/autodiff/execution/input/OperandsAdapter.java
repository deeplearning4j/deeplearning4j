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

package org.nd4j.autodiff.execution.input;

/**
 * This interface describes adapter that wraps Operands for custom datatypes.
 * I.e. Image in and class as integer value as output
 *
 * @author raver119@gmail.com
 */
public interface OperandsAdapter<T> {

    /**
     * This method must return collection of graph inputs as Operands
     * @return
     */
    Operands input(T input);

    /**
     * This method returns adopted result of graph execution
     * @return
     */
    T output(Operands operands);
}
