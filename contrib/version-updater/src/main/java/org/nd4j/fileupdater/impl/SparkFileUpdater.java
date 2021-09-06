package org.nd4j.fileupdater.impl;

import org.nd4j.fileupdater.FileUpdater;

import java.util.HashMap;
import java.util.Map;

public class SparkFileUpdater implements FileUpdater {

    private String sparkVersion;

    public SparkFileUpdater(String sparkVersion) {
        this.sparkVersion = sparkVersion;
    }

    @Override
    public Map<String, String> patterns() {
        Map<String, String> ret = new HashMap<>();
        ret.put("\\<spark.version\\>[0-9\\.]*\\<\\/spark.version\\>", String.format("<spark.version>%s</spark.version>", sparkVersion));
        ret.put("\\<spark.version\\>[0-9\\.]*\\<\\/spark.version\\>", String.format("<spark.version>%s</spark.version>", sparkVersion));
        if (sparkVersion.contains("3")) {
            ret.put("\\<artifactId\\>spark_[0-9\\.]+\\<\\/artifactId\\>", "<artifactId>spark3_2.12</artifactId>");
            ret.put("\\<artifactId\\>dl4j-spark_[0-9\\.]+\\<\\/artifactId\\>", "<artifactId>dl4j-spark3_2.12</artifactId>");
            ret.put("\\<artifactId\\>datavec-spark_[0-9\\.]+\\<\\/artifactId\\>", "<artifactId>datavec-spark3_2.12</artifactId>");

        } else {
            if (sparkVersion.contains("2")) {
                ret.put("\\<artifactId\\>spark3_[0-9\\.]+\\<\\/artifactId\\>", "<artifactId>spark_2.12</artifactId>");
                ret.put("\\<artifactId\\>dl4j-spark3_[0-9\\.]+\\<\\/artifactId\\>", "<artifactId>dl4j-spark_2.12</artifactId>");
                ret.put("\\<artifactId\\>datavec-spark3_[0-9\\.]+\\<\\/artifactId\\>", "<artifactId>datavec-spark_2.12</artifactId>");
            }
        }

     return ret;
    }
}
