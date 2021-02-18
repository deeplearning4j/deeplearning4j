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

package org.nd4j.imports.descriptors.properties;

import org.nd4j.autodiff.functions.DifferentialFunction;

import java.lang.reflect.Field;

public interface AttributeAdapter {

    /**
     * Map the attribute using the specified field
     * on the specified function on
     * adapting the given input type to
     * the type of the field for the specified function.
     * @param inputAttributeValue the evaluate to adapt
     * @param fieldFor the field for
     * @param on the function to map on
     */
    void mapAttributeFor(Object inputAttributeValue, Field fieldFor, DifferentialFunction on);

}
