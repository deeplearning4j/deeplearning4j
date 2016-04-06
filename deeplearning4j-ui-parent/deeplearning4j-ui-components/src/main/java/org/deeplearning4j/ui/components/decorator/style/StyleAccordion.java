package org.deeplearning4j.ui.components.decorator.style;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.ui.api.Style;

/**
 * Created by Alex on 6/04/2016.
 */
@NoArgsConstructor @Data
public class StyleAccordion extends Style {

    private StyleAccordion(Builder builder){
        super(builder);
    }


    public static class Builder extends Style.Builder<Builder>{


        public StyleAccordion build(){
            return new StyleAccordion(this);
        }

    }

}
