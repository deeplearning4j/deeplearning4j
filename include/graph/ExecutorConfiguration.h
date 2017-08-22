//
// @author raver119@gmail.com
//

#ifndef LIBND4J_EXECUTORCONFIGURATION_H
#define LIBND4J_EXECUTORCONFIGURATION_H

#include <graph/generated/config_generated.h>

namespace nd4j {
    namespace graph {
        class ExecutorConfiguration {
        protected:
            ProfilingMode _profilingMode;
            ExecutionMode _executionMode;
            OutputMode _outputMode;
            bool _timestats;

        public:
            ExecutorConfiguration(const nd4j::graph::FlatConfiguration *conf = nullptr) {
                if (conf != nullptr) {
                    _profilingMode = conf->profilingMode();
                    _executionMode = conf->executionMode();
                    _outputMode = conf->outputMode();
                    _timestats = conf->timestats();
                } else {
                    _profilingMode = ProfilingMode_NONE;
                    _executionMode = ExecutionMode_SEQUENTIAL;
                    _outputMode = OutputMode_IMPLICIT;
                    _timestats = false;
                }
            }

            ~ExecutorConfiguration() {
                // no pointers here, just enums. so no-op.
            }
        };
    }
}

#endif //LIBND4J_EXECUTORCONFIGURATION_H
