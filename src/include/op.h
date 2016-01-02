/*
 * op.h
 *
 *  Created on: Dec 29, 2015
 *      Author: agibsonccc
 */

#ifndef OP_H_
#define OP_H_
#include <string>
namespace functions {
    namespace ops {
        template <typename T>
/**
 * Base class
 * for all operations
 */
        class Op {

        public:
            /**
             * Name of the op
             * @return the name of the operation
             */
            virtual
#ifdef __CUDACC__
            __host__
#endif
            std::string name() = 0;

            virtual
#ifdef __CUDACC__
            __host__ __device__
#endif
            ~Op(){}

        };

        template <typename T>
        class OpFactory {
        public:
            /**
             * Create the op with the given name
             * @param name
             * @return
             */
            virtual Op<T> * create(std::string name) = 0;
            virtual ~OpFactory() {}
        };




    }
}



#endif /* OP_H_ */
