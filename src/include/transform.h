/*
 * transform.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef TRANSFORM_H_
#define TRANSFORM_H_
#include <templatemath.h>
#include <op.h>
namespace functions {
    namespace transform {



        template<typename T>
        class Transform : public virtual functions::ops::Op<T> {
            /**
             * The op for transforms
             * @param d1
             * @param params
             * @return
             */
            virtual inline
#ifdef __CUDACC__
            __host__ __device__
#endif
            T op(T d1, T *params) = 0;

#ifdef __CUDACC__
            __device__ void transform(int n, int idx, T *dy, int incy, T *params, T *result, int blockSize) {
				int totalThreads = gridDim.x * blockDim.x;
				int tid = threadIdx.x;
				int i = blockIdx.x * blockDim.x + tid;
				/* equal, positive, non-unit increments. */
				for (; i < n; i += totalThreads) {
					result[i * incy] = op(dy[i * incy], params);
				}


			}
#endif

        public:
            virtual void exec(T *dx,int xStride,T *result, int resultStride,T *extraParams,int n) {
                if(xStride == 1 && resultStride == 1) {
#pragma omp simd
#pragma omp parallel for
                    for(int i = 0; i < n; i++) {
                        result[i] = op(dx[i],extraParams);
                    }

                }
                else {
#pragma omp simd
#pragma omp parallel for
                    for(int i = 0; i < n; i++) {
                        result[i * resultStride] = op(dx[i * resultStride],extraParams);
                    }
                }

            }
            virtual ~Transform() {}

        };




        namespace ops {
            template <typename T>
            class Abs : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual inline
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_abs<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("abs_strided");
                }

                virtual ~Abs() {}
            };

            template <typename T>
            class Ceiling : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual inline
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_ceil<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("ceil_strided");
                }

                virtual ~Ceiling() {}
            };

            template <typename T>
            class Cosine : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual inline
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_cos<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual inline
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("cos_strided");
                }

                virtual ~Cosine() {}
            };

            template <typename T>
            class Exp : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual inline
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_exp<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual inline
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("exp_strided");
                }

                virtual ~Exp() {}
            };

            template <typename T>
            class Floor : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual inline
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_floor<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("floor_strided");
                }

                virtual ~Floor() {}
            };

            template <typename T>
            class Log : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual inline
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_log<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("log_strided");
                }

                virtual ~Log() {}
            };

            template <typename T>
            class Neg : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual inline
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return -d1;
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("neg_strided");
                }

                virtual ~Neg() {}
            };

            template <typename T>
            class Pow : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual inline
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_pow<T>(d1,params[0]);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("pow_strided");
                }

                virtual ~Pow() {}
            };


            template <typename T>
            class Round : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual inline
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_round<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("round_strided");
                }

                virtual ~Round() {}
            };



            template <typename T>
            class Sigmoid : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual inline
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_sigmoid<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("sigmoid_strided");
                }

                virtual ~Sigmoid() {}
            };


            template <typename T>
            class SetRange : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual inline
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    T min = params[0];
                    T max = params[1];
                    if(d1 >= min && d1 <= max)
                        return d1;
                    if(min == 0 && max == 1) {
                        T val = 1 / (1 + nd4j::math::nd4j_exp<T>(-d1));
                        return (nd4j::math::nd4j_floor<T>(val * (max - min)) + min);
                    }

                    T ret =  (nd4j::math::nd4j_floor<T>(d1 * (max - min)) + min);
                    return ret;
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual inline
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("setrange_strided");
                }

                virtual ~SetRange() {}
            };

            template <typename T>
            class Sin : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual inline
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_sin<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("sin_strided");
                }

                virtual ~Sin() {}
            };

            template <typename T>
            class Sqrt : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual inline
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_sqrt<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("sqrt_strided");
                }

                virtual ~Sqrt() {}
            };


            template <typename T>
            class SoftPlus : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual inline
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return nd4j::math::softplus<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("softplus_strided");
                }

                virtual ~SoftPlus() {}
            };

            template <typename T>
            class Sign : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual inline
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return (d1 > 0) - (d1 < 0);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("sigmoid_strided");
                }

                virtual ~Sign() {}
            };


            template <typename T>
            class Tanh : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual inline
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_tanh<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("tanh_strided");
                }

                virtual ~Tanh() {}
            };


            template <typename T>
            class ACos : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual inline
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_acos<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual inline
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("acos_strided");
                }

                virtual ~ACos() {}
            };

            template <typename T>
            class ASin : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual inline
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_asin<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("asin_strided");
                }

                virtual ~ASin() {}
            };

            template <typename T>
            class ATan : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual inline
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_atan<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("atan_strided");
                }

                virtual ~ATan() {}
            };




        }


        template <typename T>
        class TransformOpFactory {
        public:
            TransformOpFactory() {
            }

            Transform<T> * getOp(std::string name) {
                if(name == "abs_strided")  return new transform::ops::Abs<T>();
                if(name == "ceil_strided")  return new transform::ops::Ceiling<T>();
                if(name == "cos_strided")  return new transform::ops::Cosine<T>();
                if(name == "exp_strided")  return new transform::ops::Exp<T>();
                if(name == "floor_strided")  return new transform::ops::Floor<T>();
                if(name == "log_strided")  return new transform::ops::Log<T>();
                if(name == "neg_strided")  return new transform::ops::Neg<T>();
                if(name == "pow_strided")  return new transform::ops::Pow<T>();
                if(name == "round_strided")  return new transform::ops::Round<T>();
                if(name == "setrange_strided")  return new transform::ops::SetRange<T>();
                if(name == "sigmoid_strided")  return new transform::ops::Sigmoid<T>();
                if(name == "sign_strided")  return new transform::ops::Sign<T>();
                if(name == "sin_strided")  return new transform::ops::Sin<T>();
                if(name == "softplus_strided")  return new transform::ops::SoftPlus<T>();
                if(name == "sqrt_strided")  return new transform::ops::Sqrt<T>();
                if(name == "tanh_strided")  return new transform::ops::Tanh<T>();
                if(name == "acos_strided")  return new transform::ops::ACos<T>();
                if(name == "asin_strided")  return new transform::ops::ASin<T>();
                if(name == "atan_strided")  return new transform::ops::ATan<T>();
                return NULL;
            }



        };

    }



}



#endif /* TRANSFORM_H_ */
