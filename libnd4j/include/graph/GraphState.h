//
// Created by raver119 on 23.01.18.
//

#ifndef LIBND4J_GRAPHSTATE_H
#define LIBND4J_GRAPHSTATE_H

#include <pointercast.h>
#include <op_boilerplate.h>
#include <dll.h>
#include <vector>
#include <map>
#include <graph/Scope.h>
#include <Status.h>
#include <graph/VariableSpace.h>
#include <ops/declarable/DeclarableOp.h>
#include <types/pair.h>
#include <graph/ArgumentsList.h>
#include <graph/Graph.h>

namespace nd4j {
namespace graph {
    template <typename T>
    class ND4J_EXPORT GraphState {
    protected:
        // id of this GraphState instance
        Nd4jLong _id = 0;

        // map of scopes. Scope id is used as key, since it's referred in calls later anyway
        std::map<int, Scope<T> *> _scopes;

        // this variable space holds temp references
        VariableSpace<T> _variableSpace;

        Graph<T> *_graph;

    public:
        explicit GraphState(Nd4jLong id);
        ~GraphState();

        /**
         *
         * @return
         */
        Nd4jLong id();

        /**
         * This method adds scope to this state tracker
         *
         * @param scopeId
         * @return
         */
        Nd4jStatus registerScope(int scopeId);

        /**
         * This method cheks if scope with given ID exists
         * 
         * @param scopeId - ID of the scope
         * @return - TRUE if scope exists, FALSE otherwise
         */
        bool hasScope(int scopeId);

        /**
         * This method removes specified scope from this state tracker
         *
         * @param scopeId
         * @return
         */
        Nd4jStatus forgetScope(int scopeId);

#ifndef __JAVACPP_HACK__
        /**
         * This method adds given op to the end of specified scope
         * PLEASE NOTE: This method is used for tests mostly
         *
         * @param scopeId
         * @param op
         * @return
         */
        Nd4jStatus attachOpToScope(int scopeId, int nodeId, DeclarableOp<T> *op, ArgumentsList inputs);

        /**
         * This method returns pointer to the scope with given id
         * 
         * @param scopeId - id of the scope
         */
        Scope<T>* getScope(int scopeId);

        Graph<T>* graph();
#endif
        /**
         * This method adds given op to the end of specified scope
         *
         * @param scopeId
         * @param opNum
         * @param type
         * @return
         */
        Nd4jStatus attachOpToScope(int scopeId, Nd4jLong opNum, int type, ArgumentsList inputs);

        /**
         * This method adds return statement to specified scope
         *
         * PLEASE NOTE: should be used only in body scopes
         *
         * @param scopeId
         * @param nodeId
         * @param args
         * @return
         */
        Nd4jStatus defineReturn(int scopeId, int nodeId, ArgumentsList args);

        /**
         * This method returns current variable space of this state holder
         *
         * @return
         */
        VariableSpace<T>*  variableSpace();
    };
}
}



#endif //LIBND4J_GRAPHSTATE_H
