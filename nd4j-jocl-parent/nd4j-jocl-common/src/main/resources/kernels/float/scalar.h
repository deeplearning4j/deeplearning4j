extern "C"
#include <stdio.h>
#include <stdlib.h>
#include "deeplearning4j.h"
//scalar and current element
__global  float op(float d1,float d2,float *params);

__global  void transform(int n, int idx,float dx,float *dy,int incy,float *params,float *result) {
	int totalThreads = get_num_groups(0) * get_local_size(0);
	int tid = get_local_id(0);
	int i = get_group_id(0) * get_local_size(0) + tid;

	for (; i < n; i += totalThreads) {
		result[i * incy] = op(dx,dy[i * incy],params);
	}

}


