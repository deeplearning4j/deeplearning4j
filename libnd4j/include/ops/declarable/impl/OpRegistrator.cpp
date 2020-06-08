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

namespace sd {
    namespace ops {

        ///////////////////////////////

        template <typename OpName>
        __registrator<OpName>::__registrator() {
            auto ptr = new OpName();
            OpRegistrator::getInstance().registerOperation(ptr);
        }


        template <typename OpName>
        __registratorSynonym<OpName>::__registratorSynonym(const char *name, const char *oname) {
            auto ptr = reinterpret_cast<OpName *>(OpRegistrator::getInstance().getOperation(oname));
            if (ptr == nullptr) {
                std::string newName(name);
                std::string oldName(oname);

                OpRegistrator::getInstance().updateMSVC(sd::ops::HashHelper::getInstance().getLongHash(newName), oldName);
                return;
            }
            OpRegistrator::getInstance().registerOperation(name, ptr);
        }

        ///////////////////////////////


        OpRegistrator& OpRegistrator::getInstance() {
          static OpRegistrator instance;
          return instance;
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

        }

        void OpRegistrator::exitHandler() {

        }

        void OpRegistrator::sigSegVHandler(int sig) {

        }

        OpRegistrator::~OpRegistrator() {
#ifndef _RELEASE
            _msvc.clear();

            for (auto x : _uniqueD)
                delete x;

            for (auto x: _uniqueH)
                delete x;

            _uniqueD.clear();

            _uniqueH.clear();

            _declarablesD.clear();

            _declarablesLD.clear();
#endif
        }

        const char * OpRegistrator::getAllCustomOperations() {
            _locker.lock();

            if (!isInit) {
                for (MAP_IMPL<std::string, sd::ops::DeclarableOp*>::iterator it=_declarablesD.begin(); it!=_declarablesD.end(); ++it) {
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


        bool OpRegistrator::registerOperation(const char* name, sd::ops::DeclarableOp* op) {
            std::string str(name);
            std::pair<std::string, sd::ops::DeclarableOp*> pair(str, op);
            _declarablesD.insert(pair);

            auto hash = sd::ops::HashHelper::getInstance().getLongHash(str);
            std::pair<Nd4jLong, sd::ops::DeclarableOp*> pair2(hash, op);
            _declarablesLD.insert(pair2);
            return true;
        }

        /**
         * This method registers operation
         *
         * @param op
         */
        bool OpRegistrator::registerOperation(sd::ops::DeclarableOp *op) {
            _uniqueD.emplace_back(op);
            return registerOperation(op->getOpName()->c_str(), op);
        }

        void OpRegistrator::registerHelper(sd::ops::platforms::PlatformHelper* op) {
            std::pair<Nd4jLong, samediff::Engine> p = {op->hash(), op->engine()};
            if (_helpersLH.count(p) > 0)
                throw std::runtime_error("Tried to double register PlatformHelper");

            _uniqueH.emplace_back(op);

            nd4j_debug("Adding helper for op \"%s\": [%lld - %i]\n", op->name().c_str(), op->hash(), (int) op->engine());

            std::pair<std::pair<std::string, samediff::Engine>, sd::ops::platforms::PlatformHelper*> pair({op->name(), op->engine()}, op);
            _helpersH.insert(pair);

            std::pair<std::pair<Nd4jLong, samediff::Engine>, sd::ops::platforms::PlatformHelper*> pair2(p, op);
            _helpersLH.insert(pair2);
        }

        sd::ops::DeclarableOp* OpRegistrator::getOperation(const char *name) {
            std::string str(name);
            return getOperation(str);
        }

        /**
         * This method returns registered Op by name
         *
         * @param name
         * @return
         */
        sd::ops::DeclarableOp *OpRegistrator::getOperation(Nd4jLong hash) {
            if (!_declarablesLD.count(hash)) {
                if (!_msvc.count(hash)) {
                    nd4j_printf("Unknown D operation requested by hash: [%lld]\n", hash);
                    return nullptr;
                } else {
                    _locker.lock();

                    auto str = _msvc.at(hash);
                    auto op = _declarablesD.at(str);
                    auto oHash = op->getOpDescriptor()->getHash();

                    std::pair<Nd4jLong, sd::ops::DeclarableOp*> pair(oHash, op);
                    _declarablesLD.insert(pair);

                    _locker.unlock();
                }
            }

            return _declarablesLD.at(hash);
        }

        sd::ops::DeclarableOp *OpRegistrator::getOperation(std::string& name) {
            if (!_declarablesD.count(name)) {
                nd4j_debug("Unknown operation requested: [%s]\n", name.c_str());
                return nullptr;
            }

            return _declarablesD.at(name);
        }

        sd::ops::platforms::PlatformHelper* OpRegistrator::getPlatformHelper(Nd4jLong hash, samediff::Engine engine) {
            std::pair<Nd4jLong, samediff::Engine> p = {hash, engine};
            if (_helpersLH.count(p) == 0)
                throw std::runtime_error("Requested helper can't be found");

            return _helpersLH[p];
        }

        bool OpRegistrator::hasHelper(Nd4jLong hash, samediff::Engine engine) {
            std::pair<Nd4jLong, samediff::Engine> p = {hash, engine};
            return _helpersLH.count(p) > 0;
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
    }
}

namespace std {
    size_t hash<std::pair<Nd4jLong, samediff::Engine>>::operator()(const std::pair<Nd4jLong, samediff::Engine>& k) const {
        using std::hash;
        auto res = std::hash<Nd4jLong>()(k.first);
        res ^= std::hash<int>()((int) k.second)  + 0x9e3779b9 + (res << 6) + (res >> 2);
        return res;
    }

    size_t hash<std::pair<std::string, samediff::Engine>>::operator()(const std::pair<std::string, samediff::Engine>& k) const {
        using std::hash;
        auto res = std::hash<std::string>()(k.first);
        res ^= std::hash<int>()((int) k.second)  + 0x9e3779b9 + (res << 6) + (res >> 2);
        return res;
    }
}

