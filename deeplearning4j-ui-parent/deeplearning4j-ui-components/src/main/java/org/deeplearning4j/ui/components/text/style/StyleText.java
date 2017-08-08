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
package org.deeplearning4j.ui.components.text.style;


import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.ui.api.Style;
import org.deeplearning4j.ui.api.Utils;
import org.nd4j.shade.jackson.annotation.JsonInclude;

import java.awt.*;

/**
 * Style for text
 */
@Data
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
@JsonInclude(JsonInclude.Include.NON_NULL)
public class StyleText extends Style {

    private String font;
    private Double fontSize;
    private Boolean underline;
    private String color;
    private Boolean whitespacePre;

    private StyleText(Builder builder) {
        super(builder);
        this.font = builder.font;
        this.fontSize = builder.fontSize;
        this.underline = builder.underline;
        this.color = builder.color;
        this.whitespacePre = builder.whitespacePre;
    }


    public static class Builder extends Style.Builder<Builder> {

        private String font;
        private Double fontSize;
        private Boolean underline;
        private String color;
        private Boolean whitespacePre;

        /** Specify the font to be used for the text */
        public Builder font(String font) {
            this.font = font;
            return this;
        }

        /** Size of the font (pt) */
        public Builder fontSize(double size) {
            this.fontSize = size;
            return this;
        }

        /** If true: text should be underlined (default: not) */
        public Builder underline(boolean underline) {
            this.underline = underline;
            return this;
        }

        /** Color for the text */
        public Builder color(Color color) {
            return color(Utils.colorToHex(color));
        }

        /** Color for the text */
        public Builder color(String color) {
            this.color = color;
            return this;
        }

        /**
         * If set to true: add a "white-space: pre" to the style.
         * In effect, this stops the representation from compressing the whitespace characters, and messing up/removing
         * text that contains newlines, tabs, etc.
         */
        public Builder whitespacePre(boolean whitespacePre) {
            this.whitespacePre = whitespacePre;
            return this;
        }

        public StyleText build() {
            return new StyleText(this);
        }

    }

}
