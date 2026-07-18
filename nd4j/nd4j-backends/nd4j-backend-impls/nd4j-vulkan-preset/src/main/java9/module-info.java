open module nd4j.vulkan.preset {
    requires nd4j.api;
    requires nd4j.cpu.api;
    requires nd4j.presets.common;
    requires org.bytedeco.javacpp;
    exports org.nd4j.presets.vulkan;
}
