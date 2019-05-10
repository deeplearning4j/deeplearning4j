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

package org.nd4j.imports.descriptors.properties.adapters;

import lombok.AllArgsConstructor;
import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;

import java.lang.reflect.Field;

/**
 * Comparison for whether a string not equals a target string
 * returning a boolean
 */
@AllArgsConstructor
public class StringNotEqualsAdapter implements AttributeAdapter {
    private String compString;

    @Override
    public void mapAttributeFor(Object inputAttributeValue, Field fieldFor, DifferentialFunction on) {
        val b = !inputAttributeValue.toString().equals(compString);
        on.setValueFor(fieldFor, b);
    }
}
