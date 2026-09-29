package org.nd4j.fileupdater.impl;

import org.junit.Test;

import java.io.File;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class CudaFileUpdaterTest {

    @Test
    public void updatesPomCudaReferences() {
        CudaFileUpdater updater = new CudaFileUpdater("13.0", "1.6.0", "10.0");
        String input = "<artifactId>nd4j-cuda-12.9-platform</artifactId>\n"
                + "<artifactId>nd4j-zluda-12.9-platform</artifactId>\n"
                + "<cuda.version>12.9</cuda.version>\n"
                + "<cudnn.version>9.10</cudnn.version>\n"
                + "<javacpp-presets.cuda.version>1.5.12</javacpp-presets.cuda.version>";

        String actual = input;
        for (Map.Entry<String, String> replacement : updater.patterns().entrySet()) {
            actual = actual.replaceAll(replacement.getKey(), replacement.getValue());
        }

        String expected = "<artifactId>nd4j-cuda-13.0-platform</artifactId>\n"
                + "<artifactId>nd4j-zluda-13.0-platform</artifactId>\n"
                + "<cuda.version>13.0</cuda.version>\n"
                + "<cudnn.version>10.0</cudnn.version>\n"
                + "<javacpp-presets.cuda.version>1.6.0</javacpp-presets.cuda.version>";
        assertEquals(expected, actual);
    }

    @Test
    public void leavesMultiConfigurationManifestsAlone() {
        CudaFileUpdater updater = new CudaFileUpdater("13.0", "1.6.0", "10.0");

        assertTrue(updater.pathMatches(new File("module/pom.xml")));
        // Release plans and workflows list several CUDA configurations and switch
        // the checkout per shard or job; one selected version must not flatten them.
        assertFalse(updater.pathMatches(new File("release/azure/release-plan.json")));
        assertFalse(updater.pathMatches(new File(".github/workflows/build-cuda.yml")));
        assertFalse(updater.pathMatches(new File(".github/workflows/build-cuda.yaml")));
        assertFalse(updater.pathMatches(new File("src/main/java/CudaVersion.java")));
        assertFalse(updater.pathMatches(new File("target/pom.xml")));
    }
}
