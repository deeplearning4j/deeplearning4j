#include <pairwise_transform.h>

namespace functions {
namespace pairwise_transforms {
template<typename T>
class BasePairWiseTransform : public PairWiseTransform<T> {

public:

	/**
	 *
	 * @param n
	 * @param xOffset
	 * @param yOffset
	 * @param resultOffset
	 * @param dx
	 * @param dy
	 * @param incx
	 * @param incy
	 * @param params
	 * @param result
	 * @param incz
	 * @param blockSize
	 */
	__device__ void transform(
			int n,
			int xOffset,
			int yOffset,
			int resultOffset,
			T *dx,
			T *dy,
			int incx,
			int incy,
			T *params,
			T *result, int incz, int blockSize) {

		int totalThreads = gridDim.x * blockDim.x;
		int tid = threadIdx.x;
		int i = blockIdx.x * blockDim.x + tid;

		if (incy == 0) {
			if ((blockIdx.x == 0) && (tid == 0)) {
				for (; i < n; i++) {
					result[resultOffset + i * incz] = op(dx[xOffset + i * incx], params);
				}

			}
		} else if ((incx == incy) && (incx > 0)) {
			/* equal, positive, increments */
			if (incx == 1) {
				/* both increments equal to 1 */
				for (; i < n; i += totalThreads) {
					result[resultOffset + i * incz] = op(dx[xOffset + i * incx], dy[yOffset + i * incy],
							params);
				}
			} else {
				/* equal, positive, non-unit increments. */
				for (; i < n; i += totalThreads) {
					result[resultOffset + i * incz] = op(dx[xOffset + i * incx], dy[yOffset + i * incy],
							params);
				}
			}
		} else {
			/* unequal or nonpositive increments */
			for (; i < n; i += totalThreads) {
				result[resultOffset + i * incz] = op(dx[xOffset + i * incx], dy[yOffset + i * incy],
						params);
			}
		}
	}

};


}
}

