package org.nd4j.fileupdater.impl;

import org.nd4j.fileupdater.FileUpdater;

import java.util.HashMap;
import java.util.Map;

public class ScalaVersionUpdater implements FileUpdater {
    private String scalaVersion;
    @Override
    public Map<String, String> patterns() {
        Map<String, String> ret = new HashMap<>();
        ret.put("_[0-9\\.]*",scalaVersion);
        return ret;
    }
}
