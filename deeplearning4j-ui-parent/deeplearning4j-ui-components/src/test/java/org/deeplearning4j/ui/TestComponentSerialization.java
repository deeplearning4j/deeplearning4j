package org.deeplearning4j.ui;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.deeplearning4j.ui.api.Component;
import org.deeplearning4j.ui.api.LengthUnit;
import org.deeplearning4j.ui.api.Style;
import org.deeplearning4j.ui.components.chart.ChartHistogram;
import org.deeplearning4j.ui.components.chart.ChartLine;
import org.deeplearning4j.ui.components.chart.ChartScatter;
import org.deeplearning4j.ui.components.chart.ChartStackedArea;
import org.deeplearning4j.ui.components.chart.style.StyleChart;
import org.deeplearning4j.ui.components.component.ComponentDiv;
import org.deeplearning4j.ui.components.component.style.StyleDiv;
import org.deeplearning4j.ui.components.decorator.DecoratorAccordion;
import org.deeplearning4j.ui.components.decorator.style.StyleAccordion;
import org.deeplearning4j.ui.components.table.ComponentTable;
import org.deeplearning4j.ui.components.table.style.StyleTable;
import org.deeplearning4j.ui.components.text.ComponentText;
import org.deeplearning4j.ui.components.text.style.StyleText;
import org.junit.Test;

import java.awt.*;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 9/04/2016.
 */
public class TestComponentSerialization {

    @Test
    public void testSerialization() throws Exception {

        //Common style for all of the charts
        StyleChart s = new StyleChart.Builder()
                .width(640, LengthUnit.Px)
                .height(480, LengthUnit.Px)
                .margin(LengthUnit.Px, 100,40,40,20)
                .strokeWidth(2)
                .pointSize(4)
                .seriesColors(Color.GREEN, Color.MAGENTA)
                .titleStyle(new StyleText.Builder().font("courier").fontSize(16).underline(true).color(Color.GRAY).build())
                .build();
        assertSerializable(s);


        //Line chart with vertical grid
        Component c1 = new ChartLine.Builder("Line Chart!",s)
                .addSeries("series0", new double[]{0,1,2,3}, new double[]{0,2,1,4})
                .addSeries("series1", new double[]{0,1,2,3}, new double[]{0,1,0.5,2.5})
                .setGridWidth(1.0,null) //Vertical grid lines, no horizontal grid
                .build();
        assertSerializable(c1);

        //Scatter chart
        Component c2 = new ChartScatter.Builder("Scatter!",s)
                .addSeries("series0", new double[]{0,1,2,3}, new double[]{0,2,1,4})
                .showLegend(true)
                .setGridWidth(0,0)
                .build();
        assertSerializable(c2);

        //Histogram with variable sized bins
        Component c3 = new ChartHistogram.Builder("Histogram!",s)
                .addBin(-1,-0.5,0.2)
                .addBin(-0.5,0,0.5)
                .addBin(0,1,2.5)
                .addBin(1,2,0.5)
                .build();
        assertSerializable(c3);

        //Stacked area chart
        Component c4 = new ChartStackedArea.Builder("Area Chart!",s)
                .setXValues(new double[]{0,1,2,3,4,5})
                .addSeries("series0",new double[]{0,1,0,2,0,1})
                .addSeries("series1",new double[]{2,1,2,0.5,2,1})
                .build();
        assertSerializable(c4);

        //Table
        StyleTable ts = new StyleTable.Builder()
                .backgroundColor(Color.LIGHT_GRAY)
                .headerColor(Color.ORANGE)
                .borderWidth(1)
                .columnWidths(LengthUnit.Percent, 20,40,40)
                .width(500, LengthUnit.Px)
                .height(200, LengthUnit.Px)
                .build();
        assertSerializable(ts);

        Component c5 = new ComponentTable.Builder(ts)
                .header("H1","H2","H3")
                .content(new String[][]{
                        {"row0col0", "row0col1", "row0col2"},
                        {"row1col0", "row1col1", "row1col2"}
                }).build();
        assertSerializable(c5);

        //Accordion decorator, with the same chart
        StyleAccordion ac = new StyleAccordion.Builder()
                .height(480,LengthUnit.Px).width(640,LengthUnit.Px).build();
        assertSerializable(ac);

        Component c6 = new DecoratorAccordion.Builder(ac)
                .title("Accordion - Collapsed By Default!")
                .setDefaultCollapsed(true)
                .addComponents(c5)
                .build();
        assertSerializable(c6);

        //Text with styling
        Component c7 = new ComponentText.Builder("Here's some blue text in a green div!",
                new StyleText.Builder().font("courier").fontSize(30).underline(true).color(Color.BLUE).build()).build();
        assertSerializable(c7);

        //Div, with a chart inside
        Style divStyle = new StyleDiv.Builder()
                .width(30,LengthUnit.Percent).height(200,LengthUnit.Px)
                .backgroundColor(Color.GREEN)
                .floatValue(StyleDiv.FloatValue.right)
                .build();
        assertSerializable(divStyle);
        Component c8 = new ComponentDiv(divStyle, c7, new ComponentText("(Also: it's float right, 30% width, 200 px high )",null));
        assertSerializable(c8);
    }


    private static void assertSerializable(Component component) throws Exception {

        ObjectMapper om = new ObjectMapper();

        String json = om.writeValueAsString(component);

        Component fromJson = om.readValue(json,Component.class);

        assertEquals(component.toString(),fromJson.toString());     //Yes, this is a bit hacky, but lombok equal method doesn't seem to work properly for List<double[]> etc
    }

    private static void assertSerializable(Style style) throws Exception {
        ObjectMapper om = new ObjectMapper();

        String json = om.writeValueAsString(style);

        Style fromJson = om.readValue(json,Style.class);

        assertEquals(style.toString(),fromJson.toString());         //Yes, this is a bit hacky, but lombok equal method doesn't seem to work properly for List<double[]> etc
    }

}
