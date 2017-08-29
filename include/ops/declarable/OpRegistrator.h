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
            //static std::map<std::string *, nd4j::ops::DeclarableOp<float> *> _declarables;

        public:
            /**
             * This method registers operation
             *
             * @param op
             */
            //static bool registerOperation(nd4j::ops::DeclarableOp *op);

            /**
             * This method returns registered Op by name
             *
             * @param name
             * @return
             */
           // static nd4j::ops::DeclarableOp *getOperation(std::string *name);
        };


        template <typename OpName>
        struct __registratorFloat {
            __registratorFloat() {
                OpName *ptr = new OpName();
                nd4j_printf("Float OpCreated: %s\n", ptr->getOpName()->c_str());
            }
        };

        template <typename OpName>
        struct __registratorDouble {
            __registratorDouble() {
                OpName *ptr = new OpName();
                nd4j_printf("Double OpCreated: %s\n", ptr->getOpName()->c_str());
            }
        };
    }
}

#endif //LIBND4J_REGISTRATOR_H
