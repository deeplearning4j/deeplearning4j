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

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import lombok.Data;

@JsonTypeInfo(use= JsonTypeInfo.Id.NAME, include= JsonTypeInfo.As.WRAPPER_OBJECT)
@JsonSubTypes(value={
        @JsonSubTypes.Type(value = RenderableComponentString.class, name = "RenderableComponentString"),
        @JsonSubTypes.Type(value = RenderableComponentLineChart.class, name = "RenderableComponentLineChart"),
        @JsonSubTypes.Type(value = RenderableComponentScatterPlot.class, name = "RenderableComponentScatterPlot"),
        @JsonSubTypes.Type(value = RenderableComponentTable.class, name = "RenderableComponentTable"),
        @JsonSubTypes.Type(value = RenderableComponentAccordionDecorator.class, name = "RenderableComponentAccordionDecorator")
})
@Data
public abstract class RenderableComponent {

    /** Component type: used by the Arbiter UI to determine how to decode and render the object which is
     * represented by the JSON representation of this object*/
    protected final String componentType;

    public RenderableComponent(String componentType){
        this.componentType = componentType;
    }

}
