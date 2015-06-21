package org.nd4j.linalg.cpu.javacpp;

import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 * @author Adam Gibson
 */
public class CpuInfoMapper implements InfoMapper {

    public static native void process(java.nio.Buffer buffer, int size);


    @Override
    public void map(InfoMap infoMap) {
    }
}
