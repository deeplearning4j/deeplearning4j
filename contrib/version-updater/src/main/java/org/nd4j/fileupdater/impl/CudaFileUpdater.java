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

    // Only POMs are rewritten: they hold the one CUDA configuration this checkout
    // builds.  Release plans and workflows enumerate several configurations
    // (12.6, 12.9, 13.1, ZLUDA on 12.9) and select one per shard or job by running
    // change-cuda-versions.sh themselves, so rewriting them would collapse every
    // configuration onto whichever version was selected last.

    @Override
    public Map<String,String> patterns() {
        Map<String,String> ret = new LinkedHashMap<>();
        // Replace the versioned backend token everywhere it is selected: artifact IDs,
        // module names, display names, and backend properties. Suffixes such as
        // -preset and -platform remain intact.
        ret.put("nd4j-cuda-[0-9]+(?:\\.[0-9]+)+", String.format("nd4j-cuda-%s", cudaVersion));
        // ZLUDA artifacts are tied to the CUDA ABI/toolchain that produced their
        // native payload and generated binding, so their Maven identity must move
        // with the selected CUDA version as well.
        ret.put("nd4j-zluda-[0-9]+(?:\\.[0-9]+)+", String.format("nd4j-zluda-%s", cudaVersion));
        ret.put( "\\<cuda.version\\>[0-9\\.]*<\\/cuda.version\\>",String.format("<cuda.version>%s</cuda.version>",cudaVersion));
        ret.put( "\\<cudnn.version\\>[0-9\\.]*\\<\\/cudnn.version\\>",String.format("<cudnn.version>%s</cudnn.version>",cudnnVersion));
        ret.put( "\\<javacpp-presets.cuda.version\\>[0-9\\.]*<\\/javacpp-presets.cuda.version\\>",String.format("<javacpp-presets.cuda.version>%s</javacpp-presets.cuda.version>",javacppVersion));
        return ret;
    }
}
