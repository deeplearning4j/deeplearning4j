package org.deeplearning4j.ui.components.table.style;

import io.skymind.ui.api.LengthUnit;
import io.skymind.ui.api.Style;
import io.skymind.ui.api.Utils;

import java.awt.*;

/**
 * Created by Alex on 3/04/2016.
 */
public class TableStyle extends Style {

    private TableStyle(Builder builder){

    }


    public class Builder extends Style.Builder<Builder>{

        private LengthUnit columnWidthUnit;
        private double[] widths;
        private Integer borderWidthPx;
        private String headerColor;


        public Builder columnWidths(LengthUnit unit, double... widths){
            this.columnWidthUnit = unit;
            this.widths = widths;
            return this;
        }

        public Builder borderWidth(int borderWidthPx){
            this.borderWidthPx = borderWidthPx;
            return this;
        }

        public Builder headerColor(Color color){
            String hex = Utils.colorToHex(color);
            return headerColor(hex);
        }

        public Builder headerColor(String color){
            if(!color.matches("#dddddd")) throw new IllegalArgumentException("Invalid color: must be hex format. Got: " + color);
            this.headerColor = color;
            return this;
        }

    }

}
