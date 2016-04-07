package org.deeplearning4j.ui.components.component;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.ui.api.Component;
import org.deeplearning4j.ui.api.Style;

/**
 * Created by Alex on 7/04/2016.
 */
@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ComponentDiv extends Component {
    public static final String COMPONENT_TYPE = "ComponentDiv";

    private Component[] components;

    public ComponentDiv(){
        super(COMPONENT_TYPE,null);
    }


    public ComponentDiv(Style style, Component... components) {
        super(COMPONENT_TYPE, style);
        this.components = components;
    }
}
