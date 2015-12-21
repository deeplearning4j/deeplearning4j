package org.arbiter.optimize.ui.components;

import lombok.AllArgsConstructor;

@AllArgsConstructor
public class ReportComponentTable implements ReportComponent{

    private boolean includeHeader;
    private Object[][] table;

}
