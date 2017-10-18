//
// Created by raver119 on 06.10.2017.
//

#ifndef LIBND4J_ENVIRONMENT_H
#define LIBND4J_ENVIRONMENT_H

#include <atomic>
#include <dll.h>

namespace nd4j{
    class ND4J_EXPORT Environment {
    private:
        std::atomic<int> _tadThreshold;
        std::atomic<int> _elementThreshold;
        std::atomic<bool> _verbose;
        std::atomic<bool> _debug;

        static Environment* _instance;

        Environment();
        ~Environment();
    public:

        static Environment* getInstance();
        bool isVerbose();
        void setVerbose(bool reallyVerbose);
        bool isDebug();
        bool isDebugAndVerbose();
        void setDebug(bool reallyDebug);
        int tadThreshold();
        void setTadThreshold(int threshold);
        int elementwiseThreshold();
        void setElementwiseThreshold(int threshold);
    };
}


#endif //LIBND4J_ENVIRONMENT_H
