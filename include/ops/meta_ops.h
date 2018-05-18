#pragma once
#ifndef FUSED_OPS_H_
#define FUSED_OPS_H_

#include <pointercast.h>
#include <op_boilerplate.h>

#include <ops/ops.h>

namespace metaOps {
    /**
     * InvertedMetaOp shares the same idea as MetaOp, but op being applied to op.Y in pairwise/broadcast ops
     */
    template<typename T, typename OpTypeA, typename OpTypeB>
    class InvertedMetaOp {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

        /*
         * PREDICATE
         */

        // scalar, transform, reduce, indexreduce entry
		op_def static T op(T d1, T *params) {
            /*
             * We assume, that this method won't be EVER called
             */
            printf("You should NEVER see this message in output\n");
            return (T) 0.0f;
        }

        // PWT, broadcast entry. Predicate can be only scalar, transform
        op_def static T op(T d1, T d2, T *params) {
            Nd4jPointer *wrap = reinterpret_cast<Nd4jPointer *> (params);
            T *paramsA = reinterpret_cast<T *> (wrap[0]);
            T *paramsB = reinterpret_cast<T *> (wrap[1]);

            return OpTypeB::op(OpTypeA::op(d1, d2, paramsA), paramsB);
        }

        /*
         * POSTULATE
         */

        // will be called for reduce, reduce3
        op_def static T postProcess(T reduction, Nd4jLong n, T *params) {
            /*
             * We assume, that this method won't be EVER called
             */
            printf("You should NEVER EVER see this message in output\n");

            return (T) 0.0f;
        }
    };


    /**
    * Special case here: MetaOp which consist of 2 operations.
    *
    * Predicate can be either scalar or transform, to process data before actual op call
    * Postulate will be the scalar/transform, but will be applied to result of broadcast/reduce/reduce3
    */
    template<typename T, typename OpTypeA, typename OpTypeB>
	class MetaOp {
	public:
		no_op_exec_special
		no_op_exec_special_cuda

		/*
		 * PREDICATE
		 */

		meta_def static T startingValue(const T *input) {
            return (T) 0.0f;
        }

		// scalar, transform, reduce, indexreduce entry
		meta_def static T op(T d1, T *params) {
			/*
			 * We assume, that params for MetaOp is a set of pointers to actual op A & B extraArgs
			 */
			Nd4jPointer *wrap = reinterpret_cast<Nd4jPointer *> (params);
			T *paramsA = reinterpret_cast<T *> (wrap[0]);
			T *paramsB = reinterpret_cast<T *> (wrap[1]);

			return OpTypeB::op(OpTypeA::op(d1, paramsA), paramsB);
		}

		// PWT, broadcast entry. Predicate can be only scalar, transform
		meta_def static T op(T d1, T d2, T *params) {
			Nd4jPointer *wrap = reinterpret_cast<Nd4jPointer *> (params);
			T *paramsA = reinterpret_cast<T *> (wrap[0]);
			T *paramsB = reinterpret_cast<T *> (wrap[1]);

			return OpTypeB::op(OpTypeA::op(d1, paramsA), d2, paramsB);
		}

		/*
		 * POSTULATE
		 */

		// will be called for reduce, reduce3
		meta_def static T postProcess(T reduction, Nd4jLong n, T *params) {
			Nd4jPointer *wrap = reinterpret_cast<Nd4jPointer *> (params);
			T *paramsA = reinterpret_cast<T *> (wrap[0]);
			T *paramsB = reinterpret_cast<T *> (wrap[1]);

			return OpTypeB::op(OpTypeA::postProcess(reduction, n, paramsA), paramsB);
		}
	};


    template<typename T, typename OpTypeA, typename OpTypeB>
    class ReduceMetaOp {
    public:
        no_op_exec_special
        no_op_exec_special_cuda

		meta_def static T startingValue(const T *input) {
            return OpTypeB::startingValue(input);
        }

		meta_def static T merge(T old, T opOutput, T *params) {
            Nd4jPointer *wrap = reinterpret_cast<Nd4jPointer *> (params);
//            T *paramsA = reinterpret_cast<T *> (wrap[0]);
            T *paramsB = reinterpret_cast<T *> (wrap[1]);

            return OpTypeB::merge(old, opOutput, paramsB);
        }

		meta_def static T update(T old, T opOutput, T *params) {
            Nd4jPointer *wrap = reinterpret_cast<Nd4jPointer *> (params);
            //T *paramsA = reinterpret_cast<T *> (wrap[0]);
            T *paramsB = reinterpret_cast<T *> (wrap[1]);

            return OpTypeB::update(old, opOutput, paramsB);
        }

		meta_def static T op(T d1, T *params) {
            Nd4jPointer *wrap = reinterpret_cast<Nd4jPointer *> (params);
            T *paramsA = reinterpret_cast<T *> (wrap[0]);
            T *paramsB = reinterpret_cast<T *> (wrap[1]);

            return OpTypeB::op(OpTypeA::op(d1, paramsA), paramsB);
        }

		meta_def static T postProcess(T reduction, Nd4jLong n, T *params) {
            Nd4jPointer *wrap = reinterpret_cast<Nd4jPointer *> (params);
//            T *paramsA = reinterpret_cast<T *> (wrap[0]);
            T *paramsB = reinterpret_cast<T *> (wrap[1]);

            return OpTypeB::postProcess(reduction, n, paramsB);
        }
    };
}

#endif