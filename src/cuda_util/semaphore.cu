/*
 * semaphore_impl.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#include <semaphore.h>

inline __device__ CudaSpinLock::CudaSpinLock(int *p) {
	m_p = p;
}

inline __device__ void CudaSpinLock::acquire() {
	while (atomicCAS(m_p, 0, 1))
		;
}

inline __device__ void CudaSpinLock::release() {
	atomicExch(m_p, 0);
}

