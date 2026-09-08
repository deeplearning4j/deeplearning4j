open module nd4j.vulkan {
    requires nd4j.api;
    requires nd4j.common;
    requires nd4j.vulkan.preset;
    requires org.apache.commons.lang3;
    requires nd4j.cpu.api;
    requires org.bytedeco.javacpp;
    requires org.slf4j;
    requires static lombok;
    exports org.nd4j.linalg.vulkan;
    exports org.nd4j.linalg.vulkan.bindings;
    exports org.nd4j.linalg.vulkan.cache;
    exports org.nd4j.linalg.vulkan.ops.executioner;
    exports org.nd4j.linalg.vulkan.rng;
    provides org.nd4j.linalg.factory.Nd4jBackend with org.nd4j.linalg.vulkan.VulkanBackend;
}
