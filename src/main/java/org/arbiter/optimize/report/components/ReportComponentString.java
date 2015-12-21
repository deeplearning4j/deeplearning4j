package org.arbiter.optimize.report.components;

import lombok.AllArgsConstructor;

@AllArgsConstructor
public class ReportComponentString implements ReportComponent {
    private final String string;


    @Override
    public String toString(){
        return string;
    }

}
