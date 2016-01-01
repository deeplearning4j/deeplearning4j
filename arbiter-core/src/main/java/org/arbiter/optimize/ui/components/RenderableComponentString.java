package org.arbiter.optimize.ui.components;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
public class RenderableComponentString extends RenderableComponent {
    public static final String COMPONENT_TYPE = "string";
    private String string;

    public RenderableComponentString(){
        super(COMPONENT_TYPE);
        //No arg constructor for Jackson deserialization
        string = null;
    }

    public RenderableComponentString(String string){
        super(COMPONENT_TYPE);
        this.string = string;
    }


    @Override
    public String toString(){
        return "RenderableComponentString(" + string + ")";
    }

}
