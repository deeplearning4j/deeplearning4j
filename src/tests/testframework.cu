#include "unittest/testframework.h"
#include <cuda_runtime.h>
#include "testframework.h"

__global__ void dummy_kernel() {}

bool binary_exists_for_current_device()
{
  // check against the dummy_kernel
  // if we're unable to get the attributes, then
  // we didn't compile a binary compatible with the current device
  cudaFuncAttributes attr;
  cudaError_t error = cudaFuncGetAttributes(&attr, dummy_kernel);
  return error == cudaSuccess;
}

void list_devices(void)
{
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if(deviceCount == 0)
  {
    std::cout << "There is no device supporting CUDA" << std::endl;
  }

  int selected_device;
  cudaGetDevice(&selected_device);

  for (int dev = 0; dev < deviceCount; ++dev)
  {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    if(dev == 0)
    {
      if(deviceProp.major == 9999 && deviceProp.minor == 9999)
        std::cout << "There is no device supporting CUDA." << std::endl;
      else if(deviceCount == 1)
        std::cout << "There is 1 device supporting CUDA" << std:: endl;
      else
        std::cout << "There are " << deviceCount <<  " devices supporting CUDA" << std:: endl;
    }

    std::cout << "\nDevice " << dev << ": \"" << deviceProp.name << "\"";
    if(dev == selected_device)
      std::cout << "  [SELECTED]";
    std::cout << std::endl;

    std::cout << "  Major revision number:                         " << deviceProp.major << std::endl;
    std::cout << "  Minor revision number:                         " << deviceProp.minor << std::endl;
    std::cout << "  Total amount of global memory:                 " << deviceProp.totalGlobalMem << " bytes" << std::endl;
  }
  std::cout << std::endl;
}

// provide next, which c++03 doesn't have
template<typename Iterator> Iterator my_next(Iterator iter)
{
  return ++iter;
}


std::vector<int> CUDATestDriver::target_devices(const ArgumentMap &kwargs)
{
  std::vector<int> result;

  // by default, test all devices in the system (device id -1)
  int device_id = kwargs.count("device") ? atoi(kwargs.find("device")->second.c_str()) : -1;

  if(device_id < 0)
  {
    // target all devices in the system
    int count = 0;
    cudaGetDeviceCount(&count);

    result.resize(count);
    // XXX iota is not available in c++03
    for(int i = 0; i < count; ++i)
      result[i] = i;
  }
  else
  {
    // target the specified device
    result = std::vector<int>(1,device_id);
  }

  return result;
}

bool CUDATestDriver::check_cuda_error(bool concise)
{
  cudaError_t error = cudaGetLastError();
  if(error)
  {
    if(!concise)
    {
      std::cout << "[ERROR] CUDA Error detected before running tests: [";
      std::cout << std::string(cudaGetErrorString(error));
      std::cout << "]" << std::endl;
    }
  }

  return error;
}

bool CUDATestDriver::post_test_sanity_check(const UnitTest &test, bool concise)
{
  cudaError_t error = cudaGetLastError();
  if(error && error != cudaErrorMemoryAllocation)
  {
    if(!concise)
    {
      std::cout << "\t[ERROR] CUDA Error detected after running " << test.name << ": [";
      std::cout << std::string(cudaGetErrorString(error));
      std::cout << "]" << std::endl;
    }
  }

  return error == cudaSuccess;
}

bool CUDATestDriver::run_tests(const ArgumentSet &args, const ArgumentMap &kwargs)
{
  bool verbose = kwargs.count("verbose");
  bool concise = kwargs.count("concise");

  if(verbose && concise)
  {
    std::cout << "--verbose and --concise cannot be used together" << std::endl;
    exit(EXIT_FAILURE);
  }

  // check error status before doing anything
  if(check_cuda_error(concise)) return false;

  bool result = true;

  if(kwargs.count("verbose"))
  {
    list_devices();
  }

  // figure out which devices to target
  std::vector<int> devices = target_devices(kwargs);

  // target each device
  for(std::vector<int>::iterator device = devices.begin();
      device != devices.end();
      ++device)
  {
    // set the device
    cudaSetDevice(*device);

    // check if a binary exists for this device
    // if none exists, skip the device silently unless this is the only one we're targeting
    if(devices.size() > 1 && !binary_exists_for_current_device())
    {
      continue;
    }

    if(!concise)
    {
      // note which device we're testing
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, *device);

      std::cout << "Testing Device " << *device << ": \"" << deviceProp.name << "\"" << std::endl;
    }

    // check error status before running any tests
    if(check_cuda_error(concise)) return false;

    // run tests
    result &= UnitTestDriver::run_tests(args, kwargs);

    if(!concise && my_next(device) != devices.end())
    {
      // provide some separation between the output of separate tests
      std::cout << std::endl;
    }
  }

  return result;
}

int CUDATestDriver::current_device_architecture() const
{
  int current = -1;
  cudaGetDevice(&current);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, current);

  return 100 * deviceProp.major + 10 * deviceProp.minor;
}




