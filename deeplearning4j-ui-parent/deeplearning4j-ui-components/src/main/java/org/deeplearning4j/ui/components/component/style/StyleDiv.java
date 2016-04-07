package org.deeplearning4j.ui.components.component.style;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.ui.api.Style;

@NoArgsConstructor @Data
@JsonInclude(JsonInclude.Include.NON_NULL)
public class StyleDiv extends Style {

    public enum FloatValue {non, left, right, initial, inherit};

    private FloatValue floatValue;

    private StyleDiv(Builder builder){
        super(builder);
        this.floatValue = builder.floatValue;
    }


    public static class Builder extends Style.Builder<Builder>{

        private FloatValue floatValue;

        /** CSS float styling option */
        public Builder floatValue(FloatValue floatValue){
            this.floatValue = floatValue;
            return this;
        }

        public StyleDiv build(){
            return new StyleDiv(this);
        }
    }

}
