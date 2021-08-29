package org.nd4j.fileupdater.impl;

import org.nd4j.fileupdater.FileUpdater;

import java.util.HashMap;
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
    public Map<String,String> patterns() {
        Map<String,String> ret = new HashMap<>();
        ret.put( "\\<artifactId\\>nd4j-cuda-[0-9\\.]+\\<\\/artifactId\\>",String.format("<artifactId>nd4j-cuda-%s</artifactId>",cudaVersion));
        ret.put( "\\<artifactId\\>nd4j-cuda-[0-9\\.]*-preset<\\/artifactId\\>",String.format("<artifactId>nd4j-cuda-%s-preset</artifactId>",cudaVersion));
        ret.put( "\\<artifactId\\>nd4j-cuda-[0-9\\.]*-platform\\<\\/artifactId\\>",String.format("<artifactId>nd4j-cuda-%s-platform</artifactId>",cudaVersion));
        ret.put( "\\<artifactId\\>deeplearning4j-cuda-[0-9\\.]*\\<\\/artifactId\\>",String.format("<artifactId>deeplearning4j-cuda-%s</artifactId>",cudaVersion));
        ret.put( "\\<cuda.version\\>[0-9\\.]*<\\/cuda.version\\>",String.format("<cuda.version>%s</cuda.version>",cudaVersion));
        ret.put( "\\<cudnn.version\\>[0-9\\.]*\\<\\/cudnn.version\\>",String.format("<cudnn.version>%s</cudnn.version>",cudnnVersion));
        ret.put( "\\<javacpp-presets.cuda.version\\>[0-9\\.]*<\\/javacpp-presets.cuda.version\\>",String.format("<javacpp-presets.cuda.version>%s</javacpp-presets.cuda.version>",javacppVersion));
        return ret;
    }
}
