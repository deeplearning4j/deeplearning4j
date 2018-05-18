//
// Created by raver119 on 13.10.2017.
//

#include "ops/declarable/BooleanOp.h"
#include <vector>
#include <initializer_list>

namespace nd4j {
    namespace ops {
        template <typename T>
        BooleanOp<T>::BooleanOp(const char *name, int numInputs, bool scalar) : DeclarableOp<T>::DeclarableOp(name, numInputs, scalar) {
            //
        }

        template <typename T>
        BooleanOp<T>::~BooleanOp() {
            //
        }

        /**
        * Output shape of any BooleanOp is ALWAYS scalar
        */
        template <typename T>
        ShapeList *BooleanOp<T>::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context<T> &block) {
            Nd4jLong *shapeNew;
            ALLOCATE(shapeNew, block.getWorkspace(), shape::shapeInfoLength(2), Nd4jLong);
            shapeNew[0] = 2;
            shapeNew[1] = 1;
            shapeNew[2] = 1;
            shapeNew[3] = 1;
            shapeNew[4] = 1;
            shapeNew[5] = 0;
            shapeNew[6] = 1;
            shapeNew[7] = 99;

            return SHAPELIST(shapeNew);
        }

        template <typename T>
        bool BooleanOp<T>::evaluate(nd4j::graph::Context<T> &block) {
            // check if scalar or not

            // validation?

            Nd4jStatus status = this->validateNonEmptyInput(block);
            if (status != ND4J_STATUS_OK) {
                nd4j_printf("Inputs should be not empty for BooleanOps","");
                throw "Bad inputs";
            }

            status = this->validateAndExecute(block);
            if (status == ND4J_STATUS_TRUE)
                return true;
            else if (status == ND4J_STATUS_FALSE)
                return false;
            else {
                nd4j_printf("Got error %i during [%s] evaluation: ", (int) status, this->getOpDescriptor()->getOpName()->c_str());
                throw "Internal error";
            }
        }

        template <typename T>
        bool BooleanOp<T>::evaluate(std::initializer_list<nd4j::NDArray<T> *> args) {
            std::vector<nd4j::NDArray<T> *> vec(args);
            return this->evaluate(vec);
        }

        template <typename T>
        bool BooleanOp<T>::prepareOutputs(Context<T>& ctx) {

            auto variableSpace = ctx.getVariableSpace();

            for (int e = 0; e < this->getOpDescriptor()->getNumberOfOutputs(); e++) {
                std::pair<int, int> pair(ctx.nodeId(), e);

                if (!variableSpace->hasVariable(pair))
                    variableSpace->putVariable(pair, new Variable<T>());

                Variable<T>* var = ctx.variable(pair);

                if (var->getNDArray() == nullptr) {
                    var->setNDArray(new NDArray<T>('c', {1, 1}, ctx.getWorkspace()));
                    var->markRemovable(true);
                }
            }

            return true;
        }

        template <typename T>
        Nd4jStatus nd4j::ops::BooleanOp<T>::execute(Context<T>* block)  {

            // basic validation: ensure inputs are set
            REQUIRE_OK(this->validateNonEmptyInput(*block));

            // ensure number of IArgs, TArgs match our expectations
            REQUIRE_OK(this->validateArguments(*block));

            // this method will allocate output NDArrays for this op
            this->prepareOutputs(*block);

            auto timeStart = std::chrono::system_clock::now();

            Nd4jStatus status = this->validateAndExecute(*block);

            auto timeEnd = std::chrono::system_clock::now();
            auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds> (timeEnd - timeStart).count();
            block->setInnerTime(outerTime);

            // basically we're should be putting 0.0 as FALSE, and any non-0.0 value will be treated as TRUE
            std::pair<int,int> p(block->nodeId(), 0);
            auto var = block->variable(p);
            var->getNDArray()->putScalar(0, status == ND4J_STATUS_TRUE ? (T) 1.0f : (T) 0.0f);

            if (status == ND4J_STATUS_FALSE || status == ND4J_STATUS_TRUE)
                return ND4J_STATUS_OK;
            
            nd4j_printf("%s: node_%i got unexpected result instead of boolean: [%i]\n", this->getOpName()->c_str(), block->nodeId(), status);
            return ND4J_STATUS_KERNEL_FAILURE;
        }

        template <typename T>
        bool BooleanOp<T>::evaluate(std::vector<nd4j::NDArray<T> *> &args) {
            VariableSpace<T> variableSpace;

            int cnt = -1;
            std::vector<int> in;
            for (auto v: args) {
                auto var = new Variable<T>(v);
                var->markRemovable(false);
                in.push_back(cnt);
                variableSpace.putVariable(cnt--, var);
            }

            Context<T> block(1, &variableSpace, false);
            block.fillInputs(in);

            return this->evaluate(block);
        }


        template class ND4J_EXPORT BooleanOp<float>;
        template class ND4J_EXPORT BooleanOp<float16>;
        template class ND4J_EXPORT BooleanOp<double>;
    }
}

