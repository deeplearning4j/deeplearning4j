/* ******************************************************************************
 * NDArray_print.cu - Print buffer operations
 * Split from NDArray.cu to reduce object file size for large binary builds
 ******************************************************************************/

#include <array/NDArray.h>
#include <system/Environment.h>

namespace sd {

//////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray::printCurrentBuffer(const bool host, const char* msg, const int precision) {
  if (!isScalar() && _length == 0) {
    printf("NDArray::printActualBuffer: array length is zero !\n");
    return;
  }

  // Get print options for truncation
  int edgeItems = Environment::getInstance().printEdgeItems();
  int threshold = Environment::getInstance().printThreshold();
  bool summarize = (_length > threshold);

  if(isScalar()) {
    if(host) {
      if (msg) printf("%s", msg);

      if (buffer() == nullptr) {
        printf("NDArray::printActualBuffer: host buffer is nullptr !\n");
        return;
      }

      const T* buff = bufferAsT<T>();
      printf("%.*f\n", precision, (double)buff[getOffset(0)]);
      return;
    } else {
      if (msg) printf("%s", msg);

      if (specialBuffer() == nullptr) {
        printf("NDArray::printSpecialBuffer: special buffer is nullptr !\n");
        return;
      }

      const auto sizeOfBuffer = sizeOfT();
      void* pHost = operator new(sizeOfBuffer);

      cudaMemcpyAsync(pHost, specialBuffer(), sizeOfBuffer, cudaMemcpyDeviceToHost, *getContext()->getCudaStream());
      cudaError_t cudaResult = cudaStreamSynchronize(*getContext()->getCudaStream());
      auto cast = reinterpret_cast<T*>(pHost);
      if (cudaResult != 0) THROW_EXCEPTION("NDArray::printSpecialBuffer: cudaStreamSynchronize failed!");
      printf("%.*f\n", precision, (double)cast[0]);
      operator delete(pHost);
      return;
    }
  }

  if (msg) printf("%s", msg);

  // Helper lambda to print elements with truncation
  auto printElements = [&](const T* buff) {
    printf("[");
    if (summarize && _length > 2 * edgeItems) {
      // Print first edgeItems
      for (LongType i = 0; i < edgeItems; i++) {
        printf("%.*f, ", precision, (double)buff[getOffset(i)]);
      }
      printf("..., ");
      // Print last edgeItems
      for (LongType i = _length - edgeItems; i < _length; i++) {
        printf("%.*f", precision, (double)buff[getOffset(i)]);
        if (i < _length - 1) printf(", ");
      }
    } else {
      for (LongType i = 0; i < _length; i++) {
        printf("%.*f", precision, (double)buff[getOffset(i)]);
        if (i < _length - 1) printf(", ");
      }
    }
    printf("]\n");
  };

  if (host) {
    if (buffer() == nullptr || _length == 0) {
      printf("NDArray::printActualBuffer: host buffer is nullptr !\n");
      return;
    }

    const T* buff = bufferAsT<T>();
    printElements(buff);
  } else {
    if (specialBuffer() == nullptr) {
      printf("NDArray::printSpecialBuffer: special buffer is nullptr !\n");
      return;
    }

    // Only copy what we need for printing
    LongType elementsToFetch;
    if (summarize && _length > 2 * edgeItems) {
      elementsToFetch = _length;
    } else {
      elementsToFetch = _length;
    }

    const auto sizeOfBuffer = sizeOfT() * (getOffset(elementsToFetch - 1) + 1);
    void* pHost = operator new(sizeOfBuffer);

    cudaMemcpyAsync(pHost, specialBuffer(), sizeOfBuffer, cudaMemcpyDeviceToHost, *getContext()->getCudaStream());
    cudaError_t cudaResult = cudaStreamSynchronize(*getContext()->getCudaStream());
    if (cudaResult != 0) THROW_EXCEPTION("NDArray::printSpecialBuffer: cudaStreamSynchronize failed!");

    printElements(reinterpret_cast<T*>(pHost));
    operator delete(pHost);
  }
}

#define PRINT_BUFFER(T) template SD_LIB_EXPORT void NDArray::printCurrentBuffer<GET_SECOND(T)>(const bool host, const char* msg, const int precision);
ITERATE_LIST((SD_COMMON_TYPES), PRINT_BUFFER)

}  // namespace sd
