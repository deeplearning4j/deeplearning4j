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

package org.deeplearning4j.ui.components.component.style;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.ui.api.Style;
import org.nd4j.shade.jackson.annotation.JsonInclude;

/** Style for Div components.
 *
 * @author Alex Black
 */
@NoArgsConstructor
@Data
@EqualsAndHashCode(callSuper = true)
@JsonInclude(JsonInclude.Include.NON_NULL)
public class StyleDiv extends Style {

    /** Enumeration: possible values for float style option */
    public enum FloatValue {
        non, left, right, initial, inherit
    }

    private FloatValue floatValue;

    private StyleDiv(Builder builder) {
        super(builder);
        this.floatValue = builder.floatValue;
    }


    public static class Builder extends Style.Builder<Builder> {

        private FloatValue floatValue;

        /** CSS float styling option */
        public Builder floatValue(FloatValue floatValue) {
            this.floatValue = floatValue;
            return this;
        }

        public StyleDiv build() {
            return new StyleDiv(this);
        }
    }

}
