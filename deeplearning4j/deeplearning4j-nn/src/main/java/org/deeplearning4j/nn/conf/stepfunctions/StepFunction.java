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

package org.deeplearning4j.nn.conf.stepfunctions;

import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo.As;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo.Id;

import java.io.Serializable;

/**
 * Custom step function for line search.
 */
@JsonTypeInfo(use = Id.NAME, include = As.WRAPPER_OBJECT)
@JsonSubTypes(value = {@JsonSubTypes.Type(value = DefaultStepFunction.class, name = "default"),
                @JsonSubTypes.Type(value = GradientStepFunction.class, name = "gradient"),
                @JsonSubTypes.Type(value = NegativeDefaultStepFunction.class, name = "negativeDefault"),
                @JsonSubTypes.Type(value = NegativeGradientStepFunction.class, name = "negativeGradient"),})
public class StepFunction implements Serializable, Cloneable {

    private static final long serialVersionUID = -1884835867123371330L;

    @Override
    public StepFunction clone() {
        try {
            StepFunction clone = (StepFunction) super.clone();
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }
}
