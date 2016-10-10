package org.deeplearning4j.ui.stats.impl;

import lombok.AllArgsConstructor;
import org.deeplearning4j.ui.stats.api.StatsInitializationConfiguration;

/**
 * Created by Alex on 07/10/2016.
 */
@AllArgsConstructor
public class DefaultStatsInitializationConfiguration implements StatsInitializationConfiguration {

    private final boolean collectSoftwareInfo;
    private final boolean collectHardwareInfo;
    private final boolean collectModelInfo;

    @Override
    public boolean collectSoftwareInfo() {
        return collectSoftwareInfo;
    }

    @Override
    public boolean collectHardwareInfo() {
        return collectHardwareInfo;
    }

    @Override
    public boolean collectModelInfo() {
        return collectModelInfo;
    }
}
