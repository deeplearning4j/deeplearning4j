//
// @author raver119@gmail.com
//

#ifndef LIBND4J_BLOCK_H
#define LIBND4J_BLOCK_H

#include <vector>
#include "Variable.h"
#include "VariableSpace.h"
#include <memory/Workspace.h>


// CUDA-specific includes
#ifdef __CUDACC__

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

#endif

namespace nd4j {
    namespace graph {
        /**
         * This class defines input desired for any given node/operation within graph
         */
        template <typename T>
        class Block {
        protected:
            nd4j::memory::Workspace* _workspace;

            // int ids of the input nodes
            std::vector<std::pair<int, int>> _inputs;

            //std::vector<nd4j::graph::Variable<T> *> _variables;

            nd4j::graph::VariableSpace<T>* _variableSpace;
            std::pair<Nd4jIndex, Nd4jIndex> _executionTime;
            nd4j::random::RandomBuffer* _rng;
            int _nodeId;

            std::vector<T> _tArgs;
            std::vector<int> _iArgs;            
			
			bool _isInplace;

            // branch for divergent_op
            int _branch = 0;

            // opNum for legacy XYZ ops
            int _opNum = -1;

        public:
            // TODO: maybe override new here as well?

            // CUDA-specific fields
#ifdef __CUDACC__
            cudaStream_t* _stream;
#endif

            Block(int nodeId, VariableSpace<T> *variableSpace = nullptr);
            Block(int nodeId, VariableSpace<T> *variableSpace, bool isInplace);

            ~Block();

            void setOuterTime(Nd4jIndex time);
            void setInnerTime(Nd4jIndex time);

            Nd4jIndex getOuterTime();
            Nd4jIndex getInnerTime();

            bool hasVariablesFilled();
            bool hasWorkspaceProvided();

            void attachWorkspace(nd4j::memory::Workspace* workspace);
            void setVariableSpace(VariableSpace<T> *variableSpace);
            void forgetWorkspace();
            nd4j::memory::Workspace* getWorkspace();
            nd4j::random::RandomBuffer* getRNG();
            void setRNG(nd4j::random::RandomBuffer* rng);
            int getNodeId();
            std::vector<T>* getTArguments();
            std::vector<int>* getIArguments();

            bool isInplace();
            void markInplace(bool reallyInplace);

            void pickInput(int input);
            void pickInput(std::pair<int, int>& p);
            void fillInputs(std::initializer_list<int> inputs);
            void fillInputs(std::vector<int>& inputs);
            std::vector<std::pair<int, int>>* inputs();

            int getBranch();
            void setBranch(int branch);

            //void updateVariables();

            /**
             * This method returns number of inputs available in this block
             * @return
             */
            unsigned long width();

            /**
            * This method returns variableSpace used in this block
            * @return
            */
            VariableSpace<T>* getVariableSpace();

            //std::vector<nd4j::graph::Variable<T> *>* getVariables();

            Variable<T>* getVariable(int idx);
            Variable<T>* variable(int idx);
            Variable<T>* variable(std::pair<int,int>& p);

            int opNum();
            void setOpNum(int opNum);
        };
    }
}


#endif //LIBND4J_BLOCK_H
