extern "C"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "deeplearning4j.h"
//an op for the kernel
__global  double op(double d1,double *params);

__global  void transform(int n,int idx,double *dy,int incy,double *params,double *result) {
	int totalThreads = get_num_groups(0) * get_local_size(0);
	int tid = get_local_id(0);
	int i = get_group_id(0) * get_local_size(0) + tid;
	/* equal, positive, non-unit increments. */
	for (; i < n; i += totalThreads) {
		       result[i * incy] = op(dy[i * incy],params);

	}


}
