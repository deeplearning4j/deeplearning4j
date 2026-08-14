#include <jni.h>

/*
 * JavaCPP's generated primary JNI translation unit normally exports JNI_OnLoad
 * and forwards it to each generated library-specific hook. Android replaces
 * that primary translation unit with a direct ART bridge, because ART and the
 * embedded Graal VM must not share JavaCPP's process-global JNI caches.
 *
 * The standalone core bridge still needs the canonical VM lifecycle entry
 * points so the embedded Graal VM initializes every cached class, field, and
 * method ID before any JavaCPP entry point is used.
 */
extern "C" JNIEXPORT jint JNICALL
JNI_OnLoad_jnijavacpp(JavaVM* vm, void* reserved);

extern "C" JNIEXPORT void JNICALL
JNI_OnUnload_jnijavacpp(JavaVM* vm, void* reserved);

extern "C" JNIEXPORT jint JNICALL
JNI_OnLoad(JavaVM* vm, void* reserved) {
    return JNI_OnLoad_jnijavacpp(vm, reserved);
}

extern "C" JNIEXPORT void JNICALL
JNI_OnUnload(JavaVM* vm, void* reserved) {
    JNI_OnUnload_jnijavacpp(vm, reserved);
}
