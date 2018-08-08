/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Created by raver119 on 07.10.2017.
//

#ifndef LIBND4J_OPREGISTRATOR_H
#define LIBND4J_OPREGISTRATOR_H

#include <pointercast.h>
#include <vector>
#include <map>
#include <mutex>
#include <ops/declarable/DeclarableOp.h>

// handlers part
#include <cstdlib>
#include <csignal>

namespace nd4j {
    namespace ops {
        /**
        *   This class provides runtime ops lookup, based on opName or opHash.
        *   To build lookup directory we use *_OP_IMPL macro, which puts static structs at compile time in .cpp files,
        *   so once binary is executed, static objects are initialized automatically, and we get list of all ops
        *   available at runtime via this singleton.
        *
        */
        class ND4J_EXPORT OpRegistrator {
        private:
            static OpRegistrator* _INSTANCE;
            OpRegistrator() {
                nd4j_debug("OpRegistrator started\n","");

#ifndef _RELEASE
                std::signal(SIGSEGV, &OpRegistrator::sigSegVHandler);
                std::signal(SIGINT, &OpRegistrator::sigIntHandler);
                std::signal(SIGABRT, &OpRegistrator::sigIntHandler);
                std::signal(SIGFPE, &OpRegistrator::sigIntHandler);
                std::signal(SIGILL, &OpRegistrator::sigIntHandler);
                std::signal(SIGTERM, &OpRegistrator::sigIntHandler);
                atexit(&OpRegistrator::exitHandler);
#endif
            };

            std::map<Nd4jLong, std::string> _msvc;

            std::map<Nd4jLong, nd4j::ops::DeclarableOp<float> *> _declarablesLF;
            std::map<std::string, nd4j::ops::DeclarableOp<float> *> _declarablesF;

            std::map<Nd4jLong, nd4j::ops::DeclarableOp<double> *> _declarablesLD;
            std::map<std::string, nd4j::ops::DeclarableOp<double> *> _declarablesD;

            std::map<Nd4jLong, nd4j::ops::DeclarableOp<float16> *> _declarablesLH;
            std::map<std::string, nd4j::ops::DeclarableOp<float16> *> _declarablesH;

            std::map<Nd4jLong, nd4j::ops::DeclarableOp<int> *> _declarablesLI;
            std::map<std::string, nd4j::ops::DeclarableOp<int> *> _declarablesI;

            std::map<Nd4jLong, nd4j::ops::DeclarableOp<Nd4jLong> *> _declarablesLL;
            std::map<std::string, nd4j::ops::DeclarableOp<Nd4jLong> *> _declarablesL;

            std::vector<nd4j::ops::DeclarableOp<float> *> _uniqueF;
            std::vector<nd4j::ops::DeclarableOp<double> *> _uniqueD;
            std::vector<nd4j::ops::DeclarableOp<float16> *> _uniqueH;
            std::vector<nd4j::ops::DeclarableOp<int> *> _uniqueI;
            std::vector<nd4j::ops::DeclarableOp<Nd4jLong> *> _uniqueL;

            std::mutex _locker;
            std::string _opsList;
            bool isInit = false;
        public:
            ~OpRegistrator();

            static OpRegistrator* getInstance();

            static void exitHandler();
            static void sigIntHandler(int sig);
            static void sigSegVHandler(int sig);

            void updateMSVC(Nd4jLong newHash, std::string& oldName);

            template <typename T>
            std::string local_to_string(T value);
            const char * getAllCustomOperations();

            /**
            * This method registers operation
            *
            * @param op
            */
            bool registerOperationFloat(const char* name, nd4j::ops::DeclarableOp<float>* op);
            bool registerOperationDouble(const char* name, nd4j::ops::DeclarableOp<double>* op);
            bool registerOperationHalf(const char* name, nd4j::ops::DeclarableOp<float16>* op);
            bool registerOperationInt(const char* name, nd4j::ops::DeclarableOp<int>* op);
            bool registerOperationLong(const char* name, nd4j::ops::DeclarableOp<Nd4jLong>* op);

            bool registerOperationFloat(nd4j::ops::DeclarableOp<float>* op);
            bool registerOperationHalf(nd4j::ops::DeclarableOp<float16> *op);
            bool registerOperationDouble(nd4j::ops::DeclarableOp<double> *op);
            bool registerOperationInt(nd4j::ops::DeclarableOp<int> *op);
            bool registerOperationLong(nd4j::ops::DeclarableOp<Nd4jLong> *op);



            /**
            * This method returns registered Op by name
            *
            * @param name
            * @return
            */
            nd4j::ops::DeclarableOp<float>* getOperationFloat(const char *name);
            nd4j::ops::DeclarableOp<float> *getOperationFloat(std::string& name);
            nd4j::ops::DeclarableOp<float> *getOperationFloat(Nd4jLong hash);

            nd4j::ops::DeclarableOp<float16> *getOperationHalf(Nd4jLong hash);
            nd4j::ops::DeclarableOp<float16>* getOperationHalf(const char *name);
            nd4j::ops::DeclarableOp<float16> *getOperationHalf(std::string& name);

            nd4j::ops::DeclarableOp<double>* getOperationDouble(const char *name);
            nd4j::ops::DeclarableOp<double> *getOperationDouble(Nd4jLong hash);
            nd4j::ops::DeclarableOp<double> *getOperationDouble(std::string& name);

            nd4j::ops::DeclarableOp<int>* getOperationInt(const char *name);
            nd4j::ops::DeclarableOp<int> *getOperationInt(Nd4jLong hash);
            nd4j::ops::DeclarableOp<int> *getOperationInt(std::string& name);

            nd4j::ops::DeclarableOp<Nd4jLong>* getOperationLong(const char *name);
            nd4j::ops::DeclarableOp<Nd4jLong> *getOperationLong(Nd4jLong hash);
            nd4j::ops::DeclarableOp<Nd4jLong> *getOperationLong(std::string& name);

            template <typename T>
            DeclarableOp<T> * getOperationT(Nd4jLong hash);

            template <typename T>
            std::vector<Nd4jLong> getAllHashes();

            int numberOfOperations();
    };


        /*
         *  These structs are used to "register" our ops in OpRegistrator.
         */
        template <typename OpName>
        struct __registratorFloat {
            __registratorFloat();
        };

        template <typename OpName>
        struct __registratorHalf {
            __registratorHalf();
        };

        template <typename OpName>
        struct __registratorDouble {
            __registratorDouble();
        };


        template <typename OpName>
        struct __registratorInt {
            __registratorInt();
        };


        template <typename OpName>
        struct __registratorLong {
            __registratorLong();
        };

        template <typename OpName>
        struct __registratorSynonymFloat {
            __registratorSynonymFloat(const char *name, const char *oname);
        };

        template <typename OpName>
        struct __registratorSynonymHalf {
            __registratorSynonymHalf(const char *name, const char *oname);
        };

        template <typename OpName>
        struct __registratorSynonymDouble {
            __registratorSynonymDouble(const char *name, const char *oname);
        };

        template <typename OpName>
        struct __registratorSynonymInt {
            __registratorSynonymInt(const char *name, const char *oname);
        };

        template <typename OpName>
        struct __registratorSynonymLong {
            __registratorSynonymLong(const char *name, const char *oname);
        };
    }
}

#endif //LIBND4J_OPREGISTRATOR_H
