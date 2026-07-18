/* sdx_exports.lds — GNU linker version script for libsdx_llm
 *
 * Export ONLY the sdx* public C API symbols (sdxLlm*, sdxAudio*, sdxVlm*).
 * Everything else (graal_*, JNI_*, IsolateEnterStub__*, __svm_*, and all
 * GraalVM internals) is hidden with local visibility so they cannot collide
 * with a co-loaded GraalVM native image (e.g. libkompile_reasoning.so) in
 * the same process.
 *
 * This mirrors kompile-graph-reasoning-local/src/main/linker/kgr_exports.lds
 * (which hides the kgr_* reasoning API under KGR_1).
 *
 * macOS note: Apple's ld(1) does not support GNU version scripts.  On macOS
 * use -exported_symbols_list instead (list one symbol per line, no wildcards).
 * This script is only used on Linux where the GNU ld / lld linker is present.
 *
 * Process model contract: with this script applied, RTLD_LOCAL loading of
 * libsdx_llm.so alongside libkompile_reasoning.so (or any other GraalVM native
 * image) is safe — each library's graal_create_isolate / JNI_* are hidden and
 * cannot be resolved across library boundaries via dlsym(RTLD_DEFAULT, ...).
 */
SDX_LLM_1 {
  global:
    sdx*;
  local:
    *;
};
