//
// @author raver119@gmail.com
//

#ifndef LIBND4J_DECLARABLE_OPS_H
#define LIBND4J_DECLARABLE_OPS_H

#include "OpDescriptor.h"

namespace nd4j {
    namespace ops {


        class DeclarableOp {
        protected:
            OpDescriptor *_descriptor;

        public:
            DeclarableOp() {

            }

            ~DeclarableOp() {
                if (_descriptor != nullptr)
                    delete _descriptor;
            }


            OpDescriptor *getOpDescriptor() {
                return _descriptor;
            }

            /**
             * This method sets arguments for op
             */
            void setArguments();

            /**
             * This method returns pointer to results
             */
            void getResults();


            /**
             * This method executes this Op b
             */
            void prepareAndExecute();


            virtual void execute() = 0;
        };
    }
}

#endif //LIBND4J_DECLARABLE_OPS_H
