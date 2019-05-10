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



#include <ops/declarable/OpRegistrator.h>
#include <sstream>

namespace nd4j {
    namespace ops {

        ///////////////////////////////

        template <typename OpName>
        __registrator<OpName>::__registrator() {
            auto ptr = new OpName();
            OpRegistrator::getInstance()->registerOperation(ptr);
        }


        template <typename OpName>
        __registratorSynonym<OpName>::__registratorSynonym(const char *name, const char *oname) {
            auto ptr = reinterpret_cast<OpName *>(OpRegistrator::getInstance()->getOperation(oname));
            if (ptr == nullptr) {
                std::string newName(name);
                std::string oldName(oname);

                OpRegistrator::getInstance()->updateMSVC(nd4j::ops::HashHelper::getInstance()->getLongHash(newName), oldName);
                return;
            }
            OpRegistrator::getInstance()->registerOperation(name, ptr);
        }

        ///////////////////////////////


        OpRegistrator* OpRegistrator::getInstance() {
            if (!_INSTANCE)
                _INSTANCE = new nd4j::ops::OpRegistrator();

            return _INSTANCE;
        }


        void OpRegistrator::updateMSVC(Nd4jLong newHash, std::string& oldName) {
            std::pair<Nd4jLong, std::string> pair(newHash, oldName);
            _msvc.insert(pair);
        }

        template <typename T>
        std::string OpRegistrator::local_to_string(T value) {
            //create an output string stream
            std::ostringstream os ;

            //throw the value into the string stream
            os << value ;

            //convert the string stream into a string and return
            return os.str() ;
        }

        template <>
        std::string OpRegistrator::local_to_string(int value) {
            //create an output string stream
            std::ostringstream os ;

            //throw the value into the string stream
            os << value ;

            //convert the string stream into a string and return
            return os.str() ;
        }

        void OpRegistrator::sigIntHandler(int sig) {
#ifndef _RELEASE
            delete OpRegistrator::getInstance();
#endif
        }

        void OpRegistrator::exitHandler() {
#ifndef _RELEASE
            delete OpRegistrator::getInstance();
#endif
        }

        void OpRegistrator::sigSegVHandler(int sig) {
#ifndef _RELEASE
            delete OpRegistrator::getInstance();
#endif
        }

        OpRegistrator::~OpRegistrator() {
#ifndef _RELEASE
            _msvc.clear();

            for (auto x : _uniqueD)
                delete x;

            _uniqueD.clear();

            _declarablesD.clear();

            _declarablesLD.clear();
#endif
        }

        const char * OpRegistrator::getAllCustomOperations() {
            _locker.lock();

            if (!isInit) {
                for (std::map<std::string, nd4j::ops::DeclarableOp*>::iterator it=_declarablesD.begin(); it!=_declarablesD.end(); ++it) {
                    std::string op = it->first + ":"
                                     + local_to_string(it->second->getOpDescriptor()->getHash()) + ":"
                                     + local_to_string(it->second->getOpDescriptor()->getNumberOfInputs()) + ":"
                                     + local_to_string(it->second->getOpDescriptor()->getNumberOfOutputs()) + ":"
                                     + local_to_string(it->second->getOpDescriptor()->allowsInplace())  + ":"
                                     + local_to_string(it->second->getOpDescriptor()->getNumberOfTArgs())  + ":"
                                     + local_to_string(it->second->getOpDescriptor()->getNumberOfIArgs())  + ":"
                                     + ";" ;
                    _opsList += op;
                }

                isInit = true;
            }

            _locker.unlock();

            return _opsList.c_str();
        }
        bool OpRegistrator::registerOperation(const char* name, nd4j::ops::DeclarableOp* op) {
            std::string str(name);
            std::pair<std::string, nd4j::ops::DeclarableOp*> pair(str, op);
            _declarablesD.insert(pair);

            auto hash = nd4j::ops::HashHelper::getInstance()->getLongHash(str);
            std::pair<Nd4jLong, nd4j::ops::DeclarableOp*> pair2(hash, op);
            _declarablesLD.insert(pair2);
            return true;
        }

        /**
         * This method registers operation
         *
         * @param op
         */
        bool OpRegistrator::registerOperation(nd4j::ops::DeclarableOp *op) {
            _uniqueD.emplace_back(op);
            return registerOperation(op->getOpName()->c_str(), op);
        }

        nd4j::ops::DeclarableOp* OpRegistrator::getOperation(const char *name) {
            std::string str(name);
            return getOperation(str);
        }

        /**
         * This method returns registered Op by name
         *
         * @param name
         * @return
         */
        nd4j::ops::DeclarableOp *OpRegistrator::getOperation(Nd4jLong hash) {
            if (!_declarablesLD.count(hash)) {
                if (!_msvc.count(hash)) {
                    nd4j_printf("Unknown D operation requested by hash: [%lld]\n", hash);
                    return nullptr;
                } else {
                    _locker.lock();

                    auto str = _msvc.at(hash);
                    auto op = _declarablesD.at(str);
                    auto oHash = op->getOpDescriptor()->getHash();

                    std::pair<Nd4jLong, nd4j::ops::DeclarableOp*> pair(oHash, op);
                    _declarablesLD.insert(pair);

                    _locker.unlock();
                }
            }

            return _declarablesLD.at(hash);
        }

        nd4j::ops::DeclarableOp *OpRegistrator::getOperation(std::string& name) {
            if (!_declarablesD.count(name)) {
                nd4j_debug("Unknown operation requested: [%s]\n", name.c_str());
                return nullptr;
            }

            return _declarablesD.at(name);
        }


        int OpRegistrator::numberOfOperations() {
            return (int) _declarablesLD.size();
        }

        std::vector<Nd4jLong> OpRegistrator::getAllHashes() {
            std::vector<Nd4jLong> result;

            for (auto &v:_declarablesLD) {
                result.emplace_back(v.first);
            }

            return result;
        }

        nd4j::ops::OpRegistrator* nd4j::ops::OpRegistrator::_INSTANCE = 0;
    }
}

