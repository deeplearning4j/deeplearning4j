/*
 * indexreduce.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef INDEXREDUCE_H_
#define INDEXREDUCE_H_
template<typename T>
struct IndexValue {
    T value;
    int index;
};

template<>
struct IndexValue<double> {
    double value;
    int index;
};

template<>
struct IndexValue<float> {
    float value;
    int index;
};

// This is the un-specialized struct.  Note that we prevent instantiation of this
// struct by putting an undefined symbol in the function body so it won't compile.
template<typename T>
struct SharedIndexValue {
    // Ensure that we won't compile any un-specialized types
    __device__ T
    *

    getPointer() {
        extern __device__ void error(void);
        error();
        return 0;
    }
};



// Following are the specializations for the following types.
// int, uint, char, uchar, short, ushort, long, ulong, bool, float, and double
// One could also specialize it for user-defined types.

template<>
struct SharedIndexValue<float> {
    __device__ IndexValue<float>
    *

    getPointer() {
        extern __shared__ IndexValue<float>
        s_int2[];
        return s_int2;
    }
};
// Following are the specializations for the following types.
// int, uint, char, uchar, short, ushort, long, ulong, bool, float, and double
// One could also specialize it for user-defined types.

template<>
struct SharedIndexValue<double> {
    __device__ IndexValue<double>
    *

    getPointer() {
        extern __shared__ IndexValue<double>
        s_int6[];
        return s_int6;
    }
};


template<typename T>
class IndexReduce {

public:
    /**
     *
     * @param val
     * @param extraParams
     * @return
     */
    //an op for the kernel
    virtual __device__ IndexValue<T>
    op(IndexValue<T>
    val,
    T *extraParams
    );

    /**
     *
     * @param old
     * @param opOutput
     * @param extraParams
     * @return
     */
    //calculate an update of the reduce operation
    virtual __device__ IndexValue<T>
    update(IndexValue<T>
    old,
    IndexValue <T> opOutput, T
    *extraParams);

    /**
     *
     * @param f1
     * @param f2
     * @param extraParams
     * @return
     */
    //invoked when combining two kernels
    virtual __device__ IndexValue<T>
    merge(IndexValue<T>
    f1,
    IndexValue <T> f2, T
    *extraParams);

    /**
     *
     * @param reduction
     * @param n
     * @param xOffset
     * @param dx
     * @param incx
     * @param extraParams
     * @param result
     * @return
     */
    //post process result (for things like means etc)
    virtual __device__ IndexValue<T>
    postProcess(IndexValue<T>
    reduction,
    int n,
    int xOffset, T
    *dx,
    int incx, T
    *extraParams,
    T *result
    );

    /**
     *
     * @param d1
     * @param d2
     * @param extraParams
     * @return
     */
    virtual __device__ T
    op(IndexValue<T>
    d1,
    IndexValue <T> d2, T
    *extraParams);

    /**
     *
     * @param sPartialsRef
     * @param tid
     * @param extraParams
     */
    virtual __device__ void aggregatePartials(IndexValue <T> **sPartialsRef, int tid, T *extraParams);

    virtual ~IndexReduce();

};


#endif /* INDEXREDUCE_H_ */
