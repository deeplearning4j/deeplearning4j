/*-
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
package org.deeplearning4j.ui.api;

import lombok.Data;
import org.deeplearning4j.ui.components.chart.*;
import org.deeplearning4j.ui.components.component.ComponentDiv;
import org.deeplearning4j.ui.components.decorator.DecoratorAccordion;
import org.deeplearning4j.ui.components.table.ComponentTable;
import org.deeplearning4j.ui.components.text.ComponentText;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

/**
 * A component is anything that can be rendered, such at charts, text or tables.
 * The intended use of these components is for Java -> JavaScript interop for UIs
 *
 * @author Alex Black
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
@JsonSubTypes(value = {@JsonSubTypes.Type(value = ChartHistogram.class, name = "ChartHistogram"),
                @JsonSubTypes.Type(value = ChartHorizontalBar.class, name = "ChartHorizontalBar"),
                @JsonSubTypes.Type(value = ChartLine.class, name = "ChartLine"),
                @JsonSubTypes.Type(value = ChartScatter.class, name = "ChartScatter"),
                @JsonSubTypes.Type(value = ChartStackedArea.class, name = "ChartStackedArea"),
                @JsonSubTypes.Type(value = ChartTimeline.class, name = "ChartTimeline"),
                @JsonSubTypes.Type(value = ComponentDiv.class, name = "ComponentDiv"),
                @JsonSubTypes.Type(value = DecoratorAccordion.class, name = "DecoratorAccordion"),
                @JsonSubTypes.Type(value = ComponentTable.class, name = "ComponentTable"),
                @JsonSubTypes.Type(value = ComponentText.class, name = "ComponentText")})
@Data
public abstract class Component {

    /** Component type: used by the Arbiter UI to determine how to decode and render the object which is
     * represented by the JSON representation of this object*/
    protected final String componentType;
    protected final Style style;

    public Component(String componentType, Style style) {
        this.componentType = componentType;
        this.style = style;
    }

}
