//
// @author raver119@gmail.com
//

#ifndef LIBND4J_EXECUTORCONFIGURATION_H
#define LIBND4J_EXECUTORCONFIGURATION_H

#include <graph/generated/config_generated.h>
#include <pointercast.h>

namespace nd4j {
    namespace graph {
        class ExecutorConfiguration {
        public:
            nd4j::graph::ProfilingMode _profilingMode;
            nd4j::graph::ExecutionMode _executionMode;
            nd4j::graph::OutputMode _outputMode;
            bool _timestats;
            Nd4jLong _footprintForward = 0L;
            Nd4jLong _footprintBackward = 0L;
            Direction _direction = Direction_FORWARD_ONLY;

            ExecutorConfiguration(const nd4j::graph::FlatConfiguration *conf = nullptr);
            ~ExecutorConfiguration() = default;
            
            ExecutorConfiguration* clone();
        };
    }
}

#endif //LIBND4J_EXECUTORCONFIGURATION_H
