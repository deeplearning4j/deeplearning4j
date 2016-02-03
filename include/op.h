/*
 * op.h
 *
 *  Created on: Dec 29, 2015
 *      Author: agibsonccc
 */

#ifndef OP_H_
#define OP_H_
#include <string>
#ifdef JNI
#include <jni.h>
#endif
namespace functions {
namespace ops {
template<typename T>
/**
 * Base class
 * for all operations
 */
class Op {
protected:
    int extraParamsLen = 0;
public:
	/**
	 * Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	__host__

#endif
	std::string name() = 0;
	virtual
#ifdef __CUDACC__
	__host__ __device__
#endif
	int extraParamsLength() {
		return extraParamsLen;
	}


	virtual inline
#ifdef __CUDACC__
	__host__ __device__
#endif
	~Op() {
	}

	inline
#ifdef __CUDACC__
	__host__ __device__
#endif
	Op() {
	}



};





template<typename T>
class OpFactory {
public:
	/**
	 * Create the op with the given name
	 * @param name
	 * @return
	 */
#ifdef __CUDACC__
	__host__ __device__
#endif
	virtual Op<T> * create(int op) = 0;
#ifdef __CUDACC__
	__device__ __host__
#endif
	virtual ~OpFactory() {
	}
};



#ifdef __CUDACC__
__host__ __device__
#endif
int strcmp(const char* s1, const char* s2) {
	while(*s1 && (*s1==*s2))
		s1++,s2++;
	return *(const unsigned char*)s1-*(const unsigned char*)s2;
}
}
}



#endif /* OP_H_ */
