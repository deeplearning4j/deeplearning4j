package org.deeplearning4j.ui;

import org.deeplearning4j.ui.api.LengthUnit;
import org.deeplearning4j.ui.components.chart.ChartHistogram;
import org.deeplearning4j.ui.components.chart.ChartLine;
import org.deeplearning4j.ui.components.chart.style.StyleChart;
import org.deeplearning4j.ui.components.table.ComponentTable;
import org.deeplearning4j.ui.components.table.style.StyleTable;
import org.deeplearning4j.ui.standalone.StaticPageUtil;
import org.junit.Test;

import java.awt.*;

/**
 * Created by Alex on 2/06/2016.
 */
public class TestStandAlone {

    @Test
    public void testStandAlone() throws Exception {


        ComponentTable ct = new ComponentTable.Builder(new StyleTable.Builder().backgroundColor(Color.LIGHT_GRAY)
                        .columnWidths(LengthUnit.Px, 100, 100).build())
                                        .content(new String[][] {{"First", "Second"}, {"More", "More2"}}).build();

        ChartLine cl = new ChartLine.Builder("Title",
                        new StyleChart.Builder().axisStrokeWidth(1.0).seriesColors(Color.BLACK, Color.ORANGE)
                                        .width(640, LengthUnit.Px).height(480, LengthUnit.Px).build())
                                                        .addSeries("First Series", new double[] {0, 1, 2, 3, 4, 5},
                                                                        new double[] {10, 20, 30, 40, 50, 60})
                                                        .addSeries("Second", new double[] {0, 0.5, 1, 1.5, 2},
                                                                        new double[] {5, 10, 15, 10, 5})
                                                        .build();

        ChartHistogram ch = new ChartHistogram.Builder("Histogram",
                        new StyleChart.Builder().axisStrokeWidth(1.0).seriesColors(Color.MAGENTA)
                                        .width(640, LengthUnit.Px).height(480, LengthUnit.Px).build()).addBin(0, 1, 1)
                                                        .addBin(1, 2, 2).addBin(2, 3, 1).build();

        System.out.println(StaticPageUtil.renderHTML(ct, cl, ch));
    }

}
