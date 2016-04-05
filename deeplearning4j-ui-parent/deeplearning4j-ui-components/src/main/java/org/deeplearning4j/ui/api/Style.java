package org.deeplearning4j.ui.api;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.ui.components.chart.style.ChartStyle;
import org.deeplearning4j.ui.components.table.style.TableStyle;

@JsonTypeInfo(use= JsonTypeInfo.Id.NAME, include= JsonTypeInfo.As.WRAPPER_OBJECT)
@JsonSubTypes(value={
        @JsonSubTypes.Type(value = ChartStyle.class, name = "StyleChart"),
        @JsonSubTypes.Type(value = TableStyle.class, name = "StyleTable")
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
    }
}
