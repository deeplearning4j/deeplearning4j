//
// Created by raver119 on 30.11.17.
//

#ifndef LIBND4J_OPS_HELPERS_H
#define LIBND4J_OPS_HELPERS_H

#include <pointercast.h>
#include <op_boilerplate.h>
#include <graph/LaunchContext.h>
#include <types/float16.h>
#include <helpers/shape.h>
#include <NDArray.h>
#include <vector>
#include <array>
#include <Status.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#endif // CUDACC

#endif // LIBND4J_HELPERS_H
