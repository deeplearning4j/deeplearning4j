//
// @author raver119@gmail.com
//

#ifndef LIBND4J_SESSIONLOCALSTORAGE_H
#define LIBND4J_SESSIONLOCALSTORAGE_H

#include <thread>
#include "VariableSpace.h"
#include "Block.h"
#include "Stash.h"
#include "Workspace.h"

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
            SessionLocalStorage(VariableSpace<T>* variableSpace = nullptr, Stash<T>* stash = nullptr) {
                // we start from 1, since key 0 holds original VariableSpace
                _sessionCounter.store(1);
                _variableSpace = variableSpace;
                _stash = stash;
            }

            ~SessionLocalStorage() {
                //
            }

            VariableSpace<T>* localVariableSpace();
            VariableSpace<T>* localVariableSpace(Nd4jIndex sessionId);


            Nd4jIndex startSession();
            void endSession(Nd4jIndex sessionId);
            void endSession();

            int numberOfSessions();
        };
    }
}

template <typename T>
VariableSpace<T>* nd4j::graph::SessionLocalStorage<T>::localVariableSpace(Nd4jIndex sessionId) {
    _mutex.lock();
    auto varSpace = _threadVariableSpace.at(sessionId);
    _mutex.unlock();

    return varSpace;
}

template <typename T>
VariableSpace<T>* nd4j::graph::SessionLocalStorage<T>::localVariableSpace() {
    return localVariableSpace(getSessionId());
}


template <typename T>
Nd4jIndex nd4j::graph::SessionLocalStorage<T>::getThreadId() {
#ifdef __APPLE__
    // syscall?
#elif _WIN32
    // some win32api
#else
    // syscall!
#endif
    auto id=std::this_thread::get_id();
    uint64_t* ptr=(uint64_t*) &id;
    return (*ptr);
}

template <typename T>
int nd4j::graph::SessionLocalStorage<T>::numberOfSessions() {
    _mutex.lock();
    int size = (int) _threadSession.size();
    _mutex.unlock();
    return size;
}

template <typename T>
void nd4j::graph::SessionLocalStorage<T>::endSession(Nd4jIndex sessionId) {
    // we should delete specific holders here
    _mutex.lock();
    _threadVariableSpace.erase(sessionId);
    _mutex.unlock();
}

template <typename T>
void nd4j::graph::SessionLocalStorage<T>::endSession() {
    auto tid = getThreadId();

    _mutex.lock();

    auto ntid = _threadSession[tid];
    _threadSession.erase(tid);

    _mutex.unlock();

    endSession(ntid);
}

template <typename T>
Nd4jIndex nd4j::graph::SessionLocalStorage<T>::getSessionId() {
    auto tid = getThreadId();

    _mutex.lock();
    auto ntid = _threadSession[tid];

    _mutex.unlock();

    return ntid;
}

template <typename T>
Nd4jIndex nd4j::graph::SessionLocalStorage<T>::startSession() {
    auto tid = getThreadId();

    nd4j_debug("Adding ThreadId: %i;\n", (int) tid);
    Nd4jIndex ntid = _sessionCounter++;
    _mutex.lock();

    _threadSession[tid] = ntid;
    _threadVariableSpace[ntid] = _variableSpace->clone();

    _mutex.unlock();

    return ntid;
}

#endif //LIBND4J_SESSIONLOCALSTORAGE_H
