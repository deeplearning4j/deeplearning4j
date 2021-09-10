package org.nd4j.aurora;

import org.bytedeco.javacpp.annotation.NoException;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.*;

/**
 *
 * @author saudet
 */
@Properties(
        value = {
                @Platform(
                        value = "linux-x86_64",
                        cinclude = "ve_offload.h",
                        link = "veo@.0",
                        includepath = "/opt/nec/ve/veos/include/",
                        linkpath = "/opt/nec/ve/veos/lib64/",
                        library = "aurora",
                        resource = {"aurora", "libaurora.so"}
                )
        },
        target = "org.nd4j.aurora.Aurora"
)
@NoException
public class AuroraPresets implements InfoMapper, BuildEnabled {

    private Logger logger;
    private java.util.Properties properties;
    private String encoding;

    @Override
    public void init(Logger logger, java.util.Properties properties, String encoding) {
        this.logger = logger;
        this.properties = properties;
        this.encoding = encoding;
    }

    @Override
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("char").cast().valueTypes("byte").pointerTypes("BytePointer", "ByteBuffer", "byte[]", "String"));
    }
}