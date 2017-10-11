//
// Created by raver119 on 11.10.2017.
//

#ifndef LIBND4J_MEMORYREPORT_H
#define LIBND4J_MEMORYREPORT_H

#include <pointercast.h>

namespace nd4j {
    namespace memory {
        class MemoryReport {
        private:
            Nd4jIndex _vm = 0;
            Nd4jIndex _rss = 0;

        public:
            MemoryReport() = default;
            ~MemoryReport() = default;

            bool operator < (const MemoryReport& other) const;
            bool operator <= (const MemoryReport& other) const;
            bool operator > (const MemoryReport& other) const;
            bool operator >= (const MemoryReport& other) const;
            bool operator == (const MemoryReport& other) const;
            bool operator != (const MemoryReport& other) const;

            Nd4jIndex getVM() const;
            void setVM(Nd4jIndex vm);

            Nd4jIndex getRSS() const;
            void setRSS(Nd4jIndex rss);
        };
    }
}



#endif //LIBND4J_MEMORYREPORT_H
