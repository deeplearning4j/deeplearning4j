package org.arbiter.optimize.ui.components;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import lombok.Data;

@JsonTypeInfo(use= JsonTypeInfo.Id.NAME, include= JsonTypeInfo.As.WRAPPER_OBJECT)
@JsonSubTypes(value={
        @JsonSubTypes.Type(value = RenderableComponentString.class, name = "RenderableComponentString"),
        @JsonSubTypes.Type(value = RenderableComponentLineChart.class, name = "RenderableComponentLineChart"),
        @JsonSubTypes.Type(value = RenderableComponentTable.class, name = "RenderableComponentTable"),
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
