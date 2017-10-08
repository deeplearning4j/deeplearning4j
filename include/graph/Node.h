//
// @author raver119@gmail.com
//

#ifndef LIBND4J_GNODE_H
#define LIBND4J_GNODE_H

#include <atomic>
#include <string>
#include <NDArray.h>
#include "Block.h"
#include <ops/declarable/DeclarableOp.h>
#include <graph/generated/node_generated.h>


namespace nd4j {
    namespace graph {

        template <typename T>
        class Graph;

        template <typename T>
        class Node {
        protected:
            DataType _dataType;
            OpType _opType;
            Block<T>* _block = nullptr;
            Nd4jIndex _opNum;
            int _id;
            std::vector<std::pair<int, int>> _input;
            std::vector<int> _output;
            std::vector<int> _dimensions;

            int * _dim = nullptr;
            std::string _name;


            // this variable points to onion layer within graph
            int _layer = -1;

            // many ops require extra parameters to run
            T *_extraParams = nullptr;


            // optional scalar. used in scalar ops and in summary stats
            float _scalar;

            bool _hasExternalOutputs;
            bool _hasExternalInputs;
            bool _hasInternalOutputs;
            bool _hasInternalInputs;

            bool _isInplace = false;

            OpClass _opClass;

            nd4j::graph::Graph<T> * _graph= nullptr;
            nd4j::ops::DeclarableOp<T> *_customOp = nullptr;

            bool _active = true;

        public:
            Node(OpType opType = OpType_TRANSFORM, int opNum = 0, int id = 0, std::initializer_list<int> input = {}, std::initializer_list<int> output = {},  std::initializer_list<int> dimensions = {}, float scalar = 0.0f);
            Node(const nd4j::graph::FlatNode *node);
            ~Node();

            bool equals(Node *other);

            OpType opType();
            Nd4jIndex opNum();
            int id();
            std::vector<std::pair<int,int>> *input();
            std::vector<int> *output();

            void setId(int id);

            T *extraParams();

            bool isMultiInput();
            bool isMultiOutput();

            int getLayer();
            void setLayer(int layer);

            bool isDivergencePoint();
            void setActive(bool reallyActive);
            bool isActive();

            bool hasExternalOutputs();
            bool hasExternalInputs();
            bool hasInternalOutputs();
            bool hasInternalInputs();

            T scalar();

            std::vector<int> * getDimensions();
            int * getDimensionsPtr();


            void pickOutput(int outputId);
            void pickExternalOutput(int outputId);
            void pickInput(int inputId);
            void pickInput(int nodeId, int outputId);
            void pickInput(std::pair<int,int>& id);

            void setName(std::string *name);
            void setName(const std::string& name);
            std::string * getName();


            void setBlock(Block<T> *block);
            Block<T>* getBlock();
            bool hasBlockAttached();

            void setCustomOp(nd4j::ops::DeclarableOp<T> *customOp = nullptr);
            nd4j::ops::DeclarableOp<T>* getCustomOp();
            bool hasCustomOp();

            void setGraph(nd4j::graph::Graph<T>* graph = nullptr);
            nd4j::graph::Graph<T>* getGraph();
            bool hasGraphEmbedded();

            bool isInplace();
            void markInplace(bool reallyInplace);
            OpClass getOpClass();

            void setOuterTime(Nd4jIndex time);
            void setInnerTime(Nd4jIndex time);
        };
    }
}

#endif //LIBND4J_GNODE_H
