package org.deeplearning4j.ui.components.text.style;


import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.ui.api.Style;
import org.deeplearning4j.ui.api.Utils;

import java.awt.*;

/**
 * Created by Alex on 3/04/2016.
 */
@Data @NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class StyleText extends Style {

    private String font;
    private Double fontSize;
    private Boolean underline;
    private String color;

    private StyleText(Builder builder){
        this.font = builder.font;
        this.fontSize = builder.fontSize;
        this.underline = builder.underline;
        this.color = builder.color;
    }


    public static class Builder extends Style.Builder<Builder>{

        private String font;
        private Double fontSize;
        private Boolean underline;
        private String color;

        /** Specify the font to be used for the text */
        public Builder font(String font){
            this.font = font;
            return this;
        }

        /** Size of the font (pt) */
        public Builder fontSize(double size){
            this.fontSize = size;
            return this;
        }

        /** If true: text should be underlined (default: not) */
        public Builder underline(boolean underline){
            this.underline = underline;
            return this;
        }

        /** Color for the text */
        public Builder color(Color color){
            return color(Utils.colorToHex(color));
        }

        /** Color for the text */
        public Builder color(String color){
            this.color = color;
            return this;
        }

        public StyleText build(){
            return new StyleText(this);
        }

    }

}
