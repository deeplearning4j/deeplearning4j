#include <scalar.h>

namespace functions {
namespace scalar {
/**
 * Base class for scalar
 * transforms
 */
template<typename T>
class BaseScalarTransform : public ScalarTransform<T> {

public:
	/**
	 *
	 * @param n
	 * @param idx
	 * @param dx
	 * @param dy
	 * @param incy
	 * @param params
	 * @param result
	 * @param blockSize
	 */
	__device__ void transform(int n, int idx, T dx, T *dy, int incy, T *params, T *result, int blockSize) {
		int totalThreads = gridDim.x * blockDim.x;
		int tid = threadIdx.x;
		int i = blockIdx.x * blockDim.x + tid;

		for (; i < n; i += totalThreads) {
			result[idx + i * incy] = op(dx, dy[idx + i * incy], params);
		}

	}

};


}
}

