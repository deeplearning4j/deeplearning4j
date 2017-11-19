//
// These ops are provided for features parity with TF
//
// @author raver119@gmail.com
//

#ifndef LIBND4J_PARITY_OPS_H
#define LIBND4J_PARITY_OPS_H

#include <climits>
#include <op_boilerplate.h>
#include <memory>
#include <shape.h>
#include <ops/ops.h>
#include <loops/random.h>
#include <NDArray.h>
#include <graph/Variable.h>
#include <ops/declarable/DeclarableOp.h>
#include <ops/declarable/DeclarableReductionOp.h>
#include <ops/declarable/DeclarableCustomOp.h>
#include <NDArrayFactory.h>
#include <ops/declarable/CustomOperations.h>
#include <graph/Context.h>
#include <ops/declarable/OpRegistrator.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(merge, -1, 1, true) {
            // TODO: to be removed
            return ND4J_STATUS_OK;
        }


        OP_IMPL(noop, -1, -1, true) {
            // Fastest op ever.
            return ND4J_STATUS_OK;
        }

        //////////////////////////////////////////////////////////////////////////
        // test op, non-divergent
        OP_IMPL(testop2i2o, 2, 2, true) {
            //nd4j_printf("CPU op used!\n","");
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);

            auto xO = OUTPUT_VARIABLE(0);
            auto yO = OUTPUT_VARIABLE(1);

            x->template applyScalar<simdOps::Add<T>>(1.0, xO, nullptr);
            y->template applyScalar<simdOps::Add<T>>(2.0, yO, nullptr);

            STORE_2_RESULTS(*xO, *yO);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(TestOp2i2o, testop2i2o);

        //////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(testcustom, 1, 1, false, 0, -1) {
            auto z = this->getZ(block);

            //new NDArray<T>('c', {100, 100});

            STORE_RESULT(*z);
            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(testcustom) {
            // this test op will just return back original shape doubled
            int *shapeOf;
            ALLOCATE(shapeOf, block.getWorkspace(), shape::rank(inputShape->at(0)), int);

            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inputShape->at(0)), int);

            for (int e = 0; e < shape::rank(inputShape->at(0)); e++)
                shapeOf[e] = inputShape->at(0)[e+1] * 2;


            shape::shapeBuffer(shape::rank(inputShape->at(0)), shapeOf, newShape);

            RELEASE(shapeOf, block.getWorkspace());

            return new ShapeList(newShape);
        }

        REDUCTION_OP_IMPL(testreduction, 1, 1, false, 0, -1) {
            auto z = OUTPUT_VARIABLE(0);

            STORE_RESULT(*z);
            return ND4J_STATUS_OK;
        }

/////////////////////////////////////////
        OP_IMPL(assign, 2, 1, false) {
            NDArray<T> *x = INPUT_VARIABLE(0);
            NDArray<T> *y = INPUT_VARIABLE(1);

            auto z = OUTPUT_VARIABLE(0);

            if (y->isScalar()) {

                z->assign(y->getScalar(0));
            } else {
                REQUIRE_OK(this->validateInputLengthMatch(block));
                REQUIRE_OK(this->validateInputDimensionsMatch(block));

                z->assign(y);
            }


            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(set, assign);
        DECLARE_SYN(copy, assign);


        OP_IMPL(mergemax, -1, 1, false) {
            REQUIRE_OK(this->validateInputDimensionsMatch(block));

            Nd4jIndex numArgs = block.width();
            NDArray<T> *x = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);


#pragma omp parallel for proc_bind(close)
            for (Nd4jIndex e = 0; e < x->lengthOf(); e++) {
                T max = -MAX_FLOAT;
                for (int i = 0; i < numArgs; i++){
                    NDArray<T> *o = INPUT_VARIABLE(i);
                    T v = o->getIndexedScalar(e);
                    if (v > max)
                        max = v;
                }
                z->putIndexedScalar(e, max);
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(MergeMax, mergemax);

        OP_IMPL(mergemaxindex, -1, 1, false) {
            REQUIRE_OK(this->validateInputDimensionsMatch(block));

            Nd4jIndex numArgs = block.width();
            NDArray<T> *x = INPUT_VARIABLE(0);
            auto z = this->getZ(block);


#pragma omp parallel for proc_bind(close)
            for (Nd4jIndex e = 0; e < x->lengthOf(); e++) {
                T max = -MAX_FLOAT;
                Nd4jIndex idx = 0;
                for (int i = 0; i < numArgs; i++){
                    NDArray<T> *o = INPUT_VARIABLE(i);
                    T v = o->getIndexedScalar(e);
                    if (v > max) {
                        max = v;
                        idx = i;
                    }
                }
                z->putIndexedScalar(e, idx);
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(MergeMaxIndex, mergemaxindex);

        OP_IMPL(mergeadd, -1, 1, false) {
            REQUIRE_OK(this->validateInputDimensionsMatch(block));

            Nd4jIndex numArgs = block.width();
            NDArray<T> *x = INPUT_VARIABLE(0);
            auto z = this->getZ(block);


#pragma omp parallel for proc_bind(close)
            for (Nd4jIndex e = 0; e < x->lengthOf(); e++) {
                T sum = (T) 0.0f;
                for (int i = 0; i < numArgs; i++){
                    NDArray<T> *o = INPUT_VARIABLE(i);
                    T v = o->getIndexedScalar(e);
                    sum += v;
                }
                z->putIndexedScalar(e, sum);
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(mergesum, mergeadd);
        DECLARE_SYN(add_n, mergeadd);
        DECLARE_SYN(addn, mergeadd);

        OP_IMPL(mergeavg, -1, 1, false) {
            REQUIRE_OK(this->validateInputDimensionsMatch(block));

            Nd4jIndex numArgs = block.width();
            NDArray<T> *x = INPUT_VARIABLE(0);
            auto z = this->getZ(block);


#pragma omp parallel for proc_bind(close)
            for (Nd4jIndex e = 0; e < x->lengthOf(); e++) {
                T sum = (T) 0.0f;
                for (int i = 0; i < numArgs; i++){
                    NDArray<T> *o = INPUT_VARIABLE(i);
                    T v = o->getIndexedScalar(e);
                    sum += v;
                }
                z->putIndexedScalar(e, sum / numArgs);
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }

        /**
         * scatter update operation
         *
         * IArgs map:
         * IArgs[0] - update operation: 0 - add; 1 - sub; 2 - mul; 3 - div; 4 - rsub; 5 - rdiv; 6 - assign
         * IArgs[1] - number of dimensions
         * IArgs[...] - dimensions
         * IArgs[...] - number of indices
         * IArgs[...] - indices
         *
         * @tparam T
         */
        CONFIGURABLE_OP_IMPL(scatter_update, 2, 1, true, 0, -1) {
            NDArray<T> *operand = INPUT_VARIABLE(0);
            NDArray<T> *updates = INPUT_VARIABLE(1);
            NDArray<T> *z = this->getZ(block);

            int opCode = INT_ARG(0);
            int dimSize = INT_ARG(1);
            std::vector<int> tadDimension;
            unsigned long e;
            unsigned long limg = 2 + dimSize;
            for (e = 2; e < limg; e++)
                tadDimension.push_back((int) INT_ARG(e));

            // increasing counter to skip numIndices
            e++;
            std::vector<int> indices;
            std::vector<int> indicesU;
            int cnt = 0;
            for (; e< block.getIArguments()->size(); e++) {
                indices.push_back((int) INT_ARG(e));
                indicesU.push_back(cnt++);
            }

            std::unique_ptr<ResultSet<T>> tadsOperand(nd4j::NDArrayFactory<T>::multipleTensorsAlongDimension(operand, indices, tadDimension));
            std::unique_ptr<ResultSet<T>> tadsUpdate(nd4j::NDArrayFactory<T>::multipleTensorsAlongDimension(updates, indicesU, tadDimension));

//#pragma omp parallel for schedule(dynamic) proc_bind(close) shared(tadsOperand, tadsUpdate)
            for (unsigned long x = 0; x < indices.size(); x++) {
                NDArray<T> *tad = tadsOperand->at(x);
                NDArray<T> *tadUpdates = tadsUpdate->at(x);

                if (tad->lengthOf() != tadUpdates->lengthOf())
                    continue;

                switch (opCode) {
                    case 0:
                        tad->template applyPairwiseTransform<simdOps::Add<T>>(tadUpdates, tad, nullptr);
                        break;
                    case 1:
                        tad->template applyPairwiseTransform<simdOps::Subtract<T>>(tadUpdates, tad, nullptr);
                        break;
                    case 2:
                        tad->template applyPairwiseTransform<simdOps::Multiply<T>>(tadUpdates, tad, nullptr);
                        break;
                    case 3:
                        tad->template applyPairwiseTransform<simdOps::Divide<T>>(tadUpdates, tad, nullptr);
                        break;
                    case 4:
                        tad->template applyPairwiseTransform<simdOps::ReverseSubtract<T>>(tadUpdates, tad, nullptr);
                        break;
                    case 5:
                        tad->template applyPairwiseTransform<simdOps::ReverseDivide<T>>(tadUpdates, tad, nullptr);
                        break;
                    case 6:
                        tad->template applyPairwiseTransform<simdOps::Copy<T>>(tadUpdates, tad, nullptr);
                        break;
                    default:
                        continue;
                        //return ND4J_STATUS_BAD_PARAMS;
                }
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(scatterupdate, scatter_update);

		//////////////////////////////////////////////////////////////////////////
		REDUCTION_OP_IMPL(sum, 1, 1, false, 0, -1) {

			std::vector<int> argI = *(block.getIArguments());
			std::vector<int> argItrunc(argI.size()-1);
			for(int i=0; i< (int) argItrunc.size(); ++i)
				argItrunc[i] = argI[i+1];

			auto x = INPUT_VARIABLE(0);
			auto z = OUTPUT_VARIABLE(0);

            //nd4j_printv("argi", argItrunc);

            x->template reduceAlongDimension<simdOps::Sum<T>>(z, argItrunc);

			return ND4J_STATUS_OK;
		}
    }
}

#endif //LIBND4J_PARITY_OPS_H

