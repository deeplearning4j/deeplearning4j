package org.arbiter.optimize.ui;

import lombok.AllArgsConstructor;
import lombok.Data;

@AllArgsConstructor @Data
public class UpdateStatus {

    private final long summaryLastUpdateTime;
    private final long settingsLastUpdateTime;
    private final long resultsLastUpdateTime;


}
