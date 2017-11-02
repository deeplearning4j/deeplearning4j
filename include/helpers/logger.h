//
// Created by raver119 on 09.01.17.
//

#ifndef LIBND4J_LOGGER_H
#define LIBND4J_LOGGER_H

#include <vector>
#include <cstdarg>
#include <Environment.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef __CUDACC__

#define nd4j_debug(FORMAT, ...) if (nd4j::Environment::getInstance()->isDebug() && nd4j::Environment::getInstance()->isVerbose()) nd4j::Logger::info(FORMAT, __VA_ARGS__);
#define nd4j_logger(FORMAT, ...) if (nd4j::Environment::getInstance()->isDebug() && nd4j::Environment::getInstance()->isVerbose()) nd4j::Logger::info(FORMAT, __VA_ARGS__);
#define nd4j_verbose(FORMAT, ...) if (nd4j::Environment::getInstance()->isVerbose()) nd4j::Logger::info(FORMAT, __VA_ARGS__);
#define nd4j_printf(FORMAT, ...) nd4j::Logger::info(FORMAT, __VA_ARGS__);

#else

#define nd4j_debug(FORMAT, A, ...)
#define nd4j_logger(FORMAT, A, ...)
#define nd4j_verbose(FORMAT, ...)
#define nd4j_printf(FORMAT, ...) nd4j::Logger::info(FORMAT, __VA_ARGS__);

#endif

namespace nd4j {
    class Logger {

    public:

#ifdef __CUDACC__
        __host__
#endif
        static void info(const char *format, ...);

#ifdef __CUDACC__
        __host__
#endif
        static void printv(const char *format, std::vector<int>& vec);
    };

}


#endif //LIBND4J_LOGGER_H
