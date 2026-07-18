open module nd4j.vulkan {
    requires nd4j.api;
    requires nd4j.cpu.api;
    requires org.bytedeco.javacpp;
    requires slf4j.api;
    exports org.nd4j.linalg.vulkan;
    exports org.nd4j.linalg.vulkan.bindings;
    exports org.nd4j.linalg.vulkan.cache;
    exports org.nd4j.linalg.vulkan.ops.executioner;
    exports org.nd4j.linalg.vulkan.rng;
    provides org.nd4j.linalg.factory.Nd4jBackend with org.nd4j.linalg.vulkan.VulkanBackend;
}
