//
// Created by raver119 on 31.10.2017.
//

#include <helpers/logger.h>

namespace nd4j {


#ifdef __CUDACC__
    __host__
#endif
    void Logger::info(const char *format, ...) {
        va_list args;
        va_start(args, format);

        vprintf(format, args);

        va_end(args);

        fflush(stdout);
    }

#ifdef __CUDACC__
    __host__
#endif
     void Logger::printv(const char *format, std::vector<int>& vec) {
        printf("%s: {", format);
        for(int e = 0; e < vec.size(); e++) {
            auto v = vec[e];
            printf("%i", v);
            if (e < vec.size() - 1)
                printf(", ");
        }
        printf("}\n");
        fflush(stdout);
    }

    #ifdef __CUDACC__
    __host__
#endif
     void Logger::printv(const char *format, std::vector<Nd4jLong>& vec) {
        printf("%s: {", format);
        for(int e = 0; e < vec.size(); e++) {
            auto v = vec[e];
            printf("%lld", (long long) v);
            if (e < vec.size() - 1)
                printf(", ");
        }
        printf("}\n");
        fflush(stdout);
    }
}