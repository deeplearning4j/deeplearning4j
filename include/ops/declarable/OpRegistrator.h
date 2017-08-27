//
// @author raver119@gmail.com
//

#ifndef LIBND4J_REGISTRATOR_H
#define LIBND4J_REGISTRATOR_H

#include <string>
#include <map>
#include <ops/declarable/declarable_ops.h>

namespace nd4j {
    namespace ops {
        class OpRegistrator {

        protected:
            //std::map<std::string>

        public:
            /**
             * This method registers operation
             *
             * @param op
             */
            static bool registerOperation(nd4j::ops::DeclarableOp *op);

            /**
             * This method returns registered Op by name
             *
             * @param name
             * @return
             */
            static nd4j::ops::DeclarableOp *getOperation(std::string *name);
        };
    }
}

#endif //LIBND4J_REGISTRATOR_H
