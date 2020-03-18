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

#include <system/pointercast.h>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <ops/declarable/DeclarableOp.h>
#include <ops/declarable/PlatformHelper.h>
#include <execution/Engine.h>

// handlers part
#include <cstdlib>
#include <csignal>

#ifndef __JAVACPP_HACK__

namespace std {

    template <>
    class hash<std::pair<Nd4jLong, samediff::Engine>> {
    public:
        size_t operator()(const std::pair<Nd4jLong, samediff::Engine>& k) const;
    };

    template <>
    class hash<std::pair<std::string, samediff::Engine>> {
    public:
        size_t operator()(const std::pair<std::string, samediff::Engine>& k) const;
    };
};

#endif


namespace sd {
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

            MAP_IMPL<Nd4jLong, std::string> _msvc;

            // pointers to our operations
            MAP_IMPL<Nd4jLong, sd::ops::DeclarableOp*> _declarablesLD;
            MAP_IMPL<std::string, sd::ops::DeclarableOp*> _declarablesD;
            std::vector<sd::ops::DeclarableOp *> _uniqueD;

            // pointers to platform-specific helpers
            MAP_IMPL<std::pair<Nd4jLong, samediff::Engine>, sd::ops::platforms::PlatformHelper*> _helpersLH;
            MAP_IMPL<std::pair<std::string, samediff::Engine>, sd::ops::platforms::PlatformHelper*> _helpersH;
            std::vector<sd::ops::platforms::PlatformHelper*> _uniqueH;

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
            * This method registers operation in our registry, so we can use them later
            *
            * @param op
            */
            bool registerOperation(const char* name, sd::ops::DeclarableOp* op);
            bool registerOperation(sd::ops::DeclarableOp *op);

            void registerHelper(sd::ops::platforms::PlatformHelper* op);

            bool hasHelper(Nd4jLong hash, samediff::Engine engine);

            sd::ops::DeclarableOp* getOperation(const char *name);
            sd::ops::DeclarableOp* getOperation(Nd4jLong hash);
            sd::ops::DeclarableOp* getOperation(std::string &name);

            sd::ops::platforms::PlatformHelper* getPlatformHelper(Nd4jLong hash, samediff::Engine engine);

            std::vector<Nd4jLong> getAllHashes();

            int numberOfOperations();
    };


        /*
         *  These structs are used to "register" our ops in OpRegistrator.
         */
        template <typename OpName>
        struct __registrator{
            __registrator();
        };

        template <typename OpName>
        struct __registratorSynonym {
            __registratorSynonym(const char *name, const char *oname);
        };

    }
}

#endif //LIBND4J_OPREGISTRATOR_H
