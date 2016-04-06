package org.deeplearning4j.ui.components.chart;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;
import lombok.Getter;
import org.deeplearning4j.ui.api.Component;
import org.deeplearning4j.ui.components.chart.style.StyleChart;

/**
 * Created by Alex on 3/04/2016.
 */
@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
public abstract class Chart extends Component {

    private String title;
    private Boolean suppressAxisHorizontal;
    private Boolean suppressAxisVertical;
    private boolean showLegend;

    private Double setXMin;
    private Double setXMax;
    private Double setYMin;
    private Double setYMax;

    private Double gridVerticalStrokeWidth;
    private Double gridHorizontalStrokeWidth;

    public Chart(String componentType, Builder builder) {
        super(componentType, builder.getStyle());
        this.title = builder.title;
        this.suppressAxisHorizontal = builder.suppressAxisHorizontal;
        this.suppressAxisVertical = builder.suppressAxisVertical;

        this.setXMin = builder.setXMin;
        this.setXMax = builder.setXMax;
        this.setYMin = builder.setYMin;
        this.setYMax = builder.setYMax;

        this.gridVerticalStrokeWidth = builder.gridVerticalStrokeWidth;
        this.gridHorizontalStrokeWidth = builder.gridHorizontalStrokeWidth;
    }


    @Getter
    public static abstract class Builder<T extends Builder<T>> {

        private String title;
        private StyleChart style;
        private Boolean suppressAxisHorizontal;
        private Boolean suppressAxisVertical;
        private boolean showLegend;

        private Double setXMin;
        private Double setXMax;
        private Double setYMin;
        private Double setYMax;

        private Double gridVerticalStrokeWidth;
        private Double gridHorizontalStrokeWidth;

        public Builder(String title, StyleChart style) {
            this.title = title;
            this.style = style;
        }

        public T suppressAxisHorizontal(Boolean suppressAxisHorizontal) {
            this.suppressAxisHorizontal = suppressAxisHorizontal;
            return (T) this;
        }

        public T suppressAxisVertical(Boolean suppressAxisVertical) {
            this.suppressAxisVertical = suppressAxisVertical;
            return (T) this;
        }

        public T showLegend(boolean showLegend) {
            this.showLegend = showLegend;
            return (T) this;
        }

        public T setXMin(Double xMin) {
            this.setXMin = xMin;
            return (T) this;
        }

        public T setXMax(Double xMax) {
            this.setXMax = xMax;
            return (T) this;
        }

        public T setYMin(Double yMin) {
            this.setYMin = yMin;
            return (T) this;
        }

        public T setYMax(Double yMax) {
            this.setYMax = yMax;
            return (T) this;
        }

        public T setGridWidth(Double gridXStrokeWidth, Double gridYStrokeWidth){
            this.gridVerticalStrokeWidth = gridXStrokeWidth;
            this.gridHorizontalStrokeWidth = gridYStrokeWidth;
            return (T) this;
        }

        public T setGridWidth(Integer gridXStrokeWidth, Integer gridYStrokeWidth){
            return setGridWidth((gridXStrokeWidth != null ? gridXStrokeWidth.doubleValue() : null),
                    (gridYStrokeWidth != null ? gridYStrokeWidth.doubleValue() : null));
        }

    }

}
