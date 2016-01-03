/*
 *
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
 *
 */
package org.arbiter.optimize.ui.components;

import lombok.Data;

/**Renderable component within an accordion-type component
 */
@Data
public class RenderableComponentAccordionDecorator extends RenderableComponent {
    public static final String COMPONENT_TYPE = "accordion";

    private String title;
    private boolean defaultCollapsed;
    private RenderableComponent[] innerComponents;

    public RenderableComponentAccordionDecorator(){
        super(COMPONENT_TYPE);
        //No arg constructor for Jackson
    }

    public RenderableComponentAccordionDecorator(String title, boolean defaultCollapsed, RenderableComponent... innerComponents) {
        super(COMPONENT_TYPE);
        this.title = title;
        this.defaultCollapsed = defaultCollapsed;
        this.innerComponents = innerComponents;
    }
}
