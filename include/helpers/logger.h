//
// Created by raver119 on 09.01.17.
//

#ifndef LIBND4J_LOGGER_H
#define LIBND4J_LOGGER_H

#include <cstdarg>

#ifndef __CUDACC__

#define nd4j_debug(FORMAT, ...) if (debug && verbose) nd4j::Logger::info(FORMAT, __VA_ARGS__);
#define nd4j_logger(FORMAT, ...) if (debug && verbose) nd4j::Logger::info(FORMAT, __VA_ARGS__);
#define nd4j_verbose(FORMAT, ...) if (verbose) nd4j::Logger::info(FORMAT, __VA_ARGS__);
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
        static void info(const char *format, ...) {
            va_list args;
            va_start(args, format);

            vprintf(format, args);

            va_end(args);

            fflush(stdout);
        }

#ifdef __CUDACC__
        __host__
#endif
        static void printv(const char *format, std::vector<int>& vec) {
            printf("%s: [", format);
            for(auto v: vec) {
                printf("%i, ", v);
            }
            printf("]\n");
            fflush(stdout);
        }
    };

}


#endif //LIBND4J_LOGGER_H
