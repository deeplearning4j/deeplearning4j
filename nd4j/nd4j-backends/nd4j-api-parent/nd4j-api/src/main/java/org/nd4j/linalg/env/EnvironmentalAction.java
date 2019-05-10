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

package org.nd4j.linalg.env;

/**
 *  This interface describes action applied to a given environment variable
 *
 * @author raver119@protonmail.com
 */
public interface EnvironmentalAction {
    /**
     * This method returns target environemt variable for this action
     * @return
     */
    String targetVariable();

    /**
     * This method will be executed with corresponding Env Var value
     *
     * @param name
     * @param value
     */
    void process(String value);
}
