//
// @author raver119@gmail.com
//

#ifndef LIBND4J_SESSIONLOCALSTORAGE_H
#define LIBND4J_SESSIONLOCALSTORAGE_H

#include <thread>
#include "VariableSpace.h"
#include "Context.h"
#include "Stash.h"
#include <memory/Workspace.h>

namespace nd4j{
    namespace graph {
        template <typename T>
        class SessionLocalStorage {
        protected:
            std::atomic<Nd4jIndex> _sessionCounter;
            std::map<Nd4jIndex, Nd4jIndex> _threadSession;
            std::map<Nd4jIndex, VariableSpace<T>*> _threadVariableSpace;

            VariableSpace<T>* _variableSpace;
            Stash<T>* _stash;

            std::mutex _mutex;

            Nd4jIndex getSessionId();
            Nd4jIndex getThreadId();
        public:
            SessionLocalStorage(VariableSpace<T>* variableSpace = nullptr, Stash<T>* stash = nullptr);

            ~SessionLocalStorage();

            VariableSpace<T>* localVariableSpace();
            VariableSpace<T>* localVariableSpace(Nd4jIndex sessionId);


            Nd4jIndex startSession();
            void endSession(Nd4jIndex sessionId);
            void endSession();

            int numberOfSessions();
        };
    }
}

#endif //LIBND4J_SESSIONLOCALSTORAGE_H
