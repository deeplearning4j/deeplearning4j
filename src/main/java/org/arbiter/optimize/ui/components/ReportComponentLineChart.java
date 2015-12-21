package org.arbiter.optimize.ui.components;

import lombok.AllArgsConstructor;

@AllArgsConstructor
public class ReportComponentLineChart implements ReportComponent {

    private final String title;
    private final double[] x;
    private final double[] y;


}
