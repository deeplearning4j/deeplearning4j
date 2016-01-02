/*
 * semaphore.h
 *
 *  Created on: Dec 26, 2015
 *      Author: agibsonccc
 */

#ifndef SEMAPHORE_H_
#define SEMAPHORE_H_


/*
 *
 *
 *
 * Implements reduction (sum) of double-precision values in terms of a
 * two-stage algorithm that accumulates partial sums in shared memory,
 * then each block acquires a spinlock on the output value to
 * atomically add its partial sum to the final output.
 *
 * Copyright (C) 2012 by Archaea Software, LLC.  All rights reserved.
 *
 * Build line: nvcc --gpu-architecture sm_20 -I ../chLib spinlockReduction.cu
 * Microbenchmark to measure performance of spin locks.
 *
 * Build with: nvcc -I ../chLib --gpu-architecture sm_20 <options> spinlockReduction.cu
 * Requires: SM 2.0 for
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

class CudaSpinLock {
public:
    CudaSpinLock(int *p);

    void acquire();

    void release();

private:
    int *m_p;
};


#endif /* SEMAPHORE_H_ */
