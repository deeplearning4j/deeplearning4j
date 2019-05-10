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

package org.deeplearning4j.ui.components.decorator;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.ui.api.Component;
import org.deeplearning4j.ui.components.decorator.style.StyleAccordion;
import org.nd4j.shade.jackson.annotation.JsonInclude;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Accordion decorator component: i.e., create an accordion (i.e., collapseable componenet) with multiple sub-components internally
 * Current implementation supports only one accordion section
 *
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true)
@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
public class DecoratorAccordion extends Component {
    public static final String COMPONENT_TYPE = "DecoratorAccordion";

    private String title;
    private boolean defaultCollapsed;
    private Component[] innerComponents;

    public DecoratorAccordion() {
        super(COMPONENT_TYPE, null);
        //No arg constructor for Jackson
    }

    private DecoratorAccordion(Builder builder) {
        super(COMPONENT_TYPE, builder.style);
        this.title = builder.title;
        this.defaultCollapsed = builder.defaultCollapsed;
        this.innerComponents = builder.innerComponents.toArray(new Component[builder.innerComponents.size()]);
    }

    public static class Builder {

        private StyleAccordion style;
        private String title;
        private List<Component> innerComponents = new ArrayList<>();
        private boolean defaultCollapsed;

        public Builder(StyleAccordion style) {
            this(null, style);
        }

        public Builder(String title, StyleAccordion style) {
            this.title = title;
            this.style = style;
        }

        public Builder title(String title) {
            this.title = title;
            return this;
        }

        /**
         * Components to show internally in the accordion element
         */
        public Builder addComponents(Component... innerComponents) {
            Collections.addAll(this.innerComponents, innerComponents);
            return this;
        }

        /**
         * Set the default collapsed/expanded state
         *
         * @param defaultCollapsed If true: default to collapsed
         */
        public Builder setDefaultCollapsed(boolean defaultCollapsed) {
            this.defaultCollapsed = defaultCollapsed;
            return this;
        }

        public DecoratorAccordion build() {
            return new DecoratorAccordion(this);
        }
    }

}
