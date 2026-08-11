package org.nd4j.fileupdater.impl;

import org.nd4j.fileupdater.FileUpdater;

import java.util.LinkedHashMap;
import java.util.Map;

public class CudaFileUpdater implements FileUpdater {
    
    private String cudaVersion;
    private String javacppVersion;
    private String cudnnVersion;

    public CudaFileUpdater(String cudaVersion,String javacppVersion,String cudnnVersion) {
        this.cudaVersion = cudaVersion;
        this.javacppVersion = javacppVersion;
        this.cudnnVersion = cudnnVersion;
    }

    @Override
    public boolean pathMatches(java.io.File inputPath) {
        if (inputPath == null || inputPath.getParentFile() == null
                || inputPath.getParentFile().getName().equals("target")) {
            return false;
        }
        String name = inputPath.getName();
        return name.equals("pom.xml") || name.equals("release-plan.json")
                || name.endsWith(".yml") || name.endsWith(".yaml");
    }

    @Override
    public Map<String,String> patterns() {
        Map<String,String> ret = new LinkedHashMap<>();
        // Replace the versioned backend token everywhere it is selected: artifact IDs,
        // module names, display names, and backend properties. Suffixes such as
        // -preset and -platform remain intact.
        ret.put("nd4j-cuda-[0-9]+(?:\\.[0-9]+)+", String.format("nd4j-cuda-%s", cudaVersion));
        // Release-plan variants use classifierSuffix values such as
        // -cuda-12.9-compile rather than the artifact-id token above.
        ret.put("-cuda-[0-9]+(?:\\.[0-9]+)+(?=-|\")", String.format("-cuda-%s", cudaVersion));
        // Keep the machine-readable plan's selected CUDA version in sync with
        // the POMs.  This is deliberately scoped to the camel-case JSON key so
        // image tags and unrelated text are not rewritten.
        ret.put("\"cudaVersion\"\\s*:\\s*\"[0-9]+(?:\\.[0-9]+)+\"",
                String.format("\"cudaVersion\": \"%s\"", cudaVersion));
        ret.put( "\\<cuda.version\\>[0-9\\.]*<\\/cuda.version\\>",String.format("<cuda.version>%s</cuda.version>",cudaVersion));
        ret.put( "\\<cudnn.version\\>[0-9\\.]*\\<\\/cudnn.version\\>",String.format("<cudnn.version>%s</cudnn.version>",cudnnVersion));
        ret.put( "\\<javacpp-presets.cuda.version\\>[0-9\\.]*<\\/javacpp-presets.cuda.version\\>",String.format("<javacpp-presets.cuda.version>%s</javacpp-presets.cuda.version>",javacppVersion));
        return ret;
    }
}
