#include "veda_helper.h"

// https://github.com/SX-Aurora/veda/issues/16
// to solve the issue above we will use ThreadLocalScopVeda and SCOPED_VEDA_CONTEXT
struct ThreadLocalScopVeda {
  bool isOk = false;
  // VEDA *ptrToVeda = nullptr;

  ThreadLocalScopVeda() = default;

  VEDA_STATUS initVeda() {
    auto status = VEDA_CALL(vedaInit(0));
    if (status) isOk = true;
    return status;
  }

  // void setVedaPtr(VEDA *veda){
  //   if(isOk) ptrToVeda = veda;
  // }

  ~ThreadLocalScopVeda() {
    if (isOk) {
      sd_debug("cleaning %s %d\n", __FILE__, __LINE__);
      // if(ptrToVeda){
      //   //sync context and unload libraries
      //   for(auto &handles : ptrToVeda->ve_handles){
      //     if(handles.status){
      //         auto device = handles.getDevice();
      //         sd_debug("cleaning device %d\n", (int)device);
      //         VEDAcontext ctx;
      //         vedaDevicePrimaryCtxRetain(&ctx, device);
      //         vedaCtxPushCurrent(ctx);
      //         vedaCtxSynchronize();
      //         vedaCtxPopCurrent(&ctx);
      //         //unload library
      //         vedaModuleUnload(handles.mod);
      //         if(handles.ctx == ctx ){
      //           vedaCtxDestroy(ctx);
      //         }
      //     }
      //   }
      // }
      VEDA_CALL(vedaExit());
    }
  }
};

thread_local ThreadLocalScopVeda scopedVeda;

VEDA::VEDA(const char* library_name) {
  int devcnt = 0;
  auto status = scopedVeda.initVeda();
  if (status) {
    status = VEDA_CALL(vedaDeviceGetCount(&devcnt));
  }
  const char* dir_name = sd::Environment::getInstance().getVedaDeviceDir();
  int use = (devcnt > MAX_DEVICE_USAGE) ? MAX_DEVICE_USAGE : devcnt;
  sd_debug("Veda devices: available %d \t will be in use %d\n", devcnt, use);
  for (int i = 0; i < use; i++) {
    VEDAdevice device;
    vedaDeviceGet(&device, i);
    VEDA_HANDLE v(library_name, device, dir_name);
    ve_handles.emplace_back(std::move(v));
  }
  // //as our Veda is singleton instance we wil inject its cleanup inside threadlocal scope
  // //as well
  // scopedVeda.setVedaPtr(this);
}
