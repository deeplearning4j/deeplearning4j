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
package org.deeplearning4j.ui.api;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.ui.components.chart.style.StyleChart;
import org.deeplearning4j.ui.components.component.style.StyleDiv;
import org.deeplearning4j.ui.components.decorator.style.StyleAccordion;
import org.deeplearning4j.ui.components.table.style.StyleTable;
import org.deeplearning4j.ui.components.text.style.StyleText;

import java.awt.*;

/**
 * Style defines things such as size of elements, an their margins.
 * Subclasses/concrete implementations have additional settings specific to the type of component
 *
 * @author Alex Black
 */
@JsonTypeInfo(use= JsonTypeInfo.Id.NAME, include= JsonTypeInfo.As.WRAPPER_OBJECT)
@JsonSubTypes(value={
        @JsonSubTypes.Type(value = StyleChart.class, name = "StyleChart"),
        @JsonSubTypes.Type(value = StyleTable.class, name = "StyleTable"),
        @JsonSubTypes.Type(value = StyleText.class, name = "StyleText"),
        @JsonSubTypes.Type(value = StyleAccordion.class, name = "StyleAccordion"),
        @JsonSubTypes.Type(value = StyleDiv.class, name = "StyleDiv")
})
@Data @AllArgsConstructor @NoArgsConstructor
public abstract class Style {

    private Double width;
    private Double height;
    private LengthUnit widthUnit;
    private LengthUnit heightUnit;

    protected LengthUnit marginUnit;
    protected Double marginTop;
    protected Double marginBottom;
    protected Double marginLeft;
    protected Double marginRight;

    protected String backgroundColor;

    public Style(Builder b){
        this.width = b.width;
        this.height = b.height;
        this.widthUnit = b.widthUnit;
        this.heightUnit = b.heightUnit;

        this.marginUnit = b.marginUnit;
        this.marginTop = b.marginTop;
        this.marginBottom = b.marginBottom;
        this.marginLeft = b.marginLeft;
        this.marginRight = b.marginRight;

        this.backgroundColor = b.backgroundColor;
    }


    @SuppressWarnings("unchecked")
    public static abstract class Builder<T extends Builder<T>>{
        protected Double width;
        protected Double height;
        protected LengthUnit widthUnit;
        protected LengthUnit heightUnit;

        protected LengthUnit marginUnit;
        protected Double marginTop;
        protected Double marginBottom;
        protected Double marginLeft;
        protected Double marginRight;

        protected String backgroundColor;

        public T width(double width, LengthUnit widthUnit){
            this.width = width;
            this.widthUnit = widthUnit;
            return (T)this;
        }

        public T height(double height, LengthUnit heightUnit){
            this.height = height;
            this.heightUnit = heightUnit;
            return (T)this;
        }

        public T margin(LengthUnit unit, Integer marginTop, Integer marginBottom, Integer marginLeft, Integer marginRight){
            return margin(unit, (marginTop != null ? marginTop.doubleValue() : null),
                    (marginBottom != null ? marginBottom.doubleValue() : null),
                    (marginLeft != null ? marginLeft.doubleValue() : null),
                    (marginRight != null ? marginRight.doubleValue() : null));
        }

        public T margin(LengthUnit unit, Double marginTop, Double marginBottom, Double marginLeft, Double marginRight){
            this.marginUnit = unit;
            this.marginTop = marginTop;
            this.marginBottom = marginBottom;
            this.marginLeft = marginLeft;
            this.marginRight = marginRight;
            return (T)this;
        }

        public T backgroundColor(Color color){
            return backgroundColor(Utils.colorToHex(color));
        }

        public T backgroundColor(String color){
            this.backgroundColor = color;
            return (T)this;
        }
    }
}
