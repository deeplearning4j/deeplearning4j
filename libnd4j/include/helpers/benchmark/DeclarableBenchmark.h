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
// Created by raver on 3/2/2019.
//

#ifndef DEV_TESTS_DECLARABLEBENCHMARK_H
#define DEV_TESTS_DECLARABLEBENCHMARK_H

#include <NDArray.h>
#include <Context.h>
#include <OpBenchmark.h>
#include <declarable/DeclarableOp.h>
#include <declarable/OpRegistrator.h>
#include <PointersManager.h>

namespace nd4j {
    class ND4J_EXPORT DeclarableBenchmark : public OpBenchmark  {
    protected:
        nd4j::ops::DeclarableOp *_op = nullptr;
        nd4j::graph::Context *_context = nullptr;
    public:
        DeclarableBenchmark(nd4j::ops::DeclarableOp &op, std::string name = 0) : OpBenchmark() {
            _op = ops::OpRegistrator::getInstance()->getOperation(op.getOpHash());
            _testName = name;
        }

        void setContext(nd4j::graph::Context *ctx) {
            _context = ctx;
        }

        std::string axis() override {
            return "N/A";
        }

        std::string orders() override {
            if(_context != nullptr && _context->isFastPath()){
                std::vector<NDArray*>& ins = _context->fastpath_in();
                std::string s;
                for( int i=0; i<ins.size(); i++ ){
                    if(i > 0){
                        s += "/";
                    }
                    s += ShapeUtils::strideAsString(_context->getNDArray(i));
                }
                return s;
            }
            return "N/A";
        }

        std::string strides() override {
            if (_context != nullptr && _context->isFastPath()) {
                std::vector<NDArray*>& ins = _context->fastpath_in();
                std::string s("");
                for( int i=0; i<ins.size(); i++ ){
                    if(i > 0){
                        s += "/";
                    }
                    s += ShapeUtils::strideAsString(_context->getNDArray(i));
                }
                return s;
            } else
                return "N/A";
        }

        std::string inplace() override {
            return "N/A";
        }

        void executeOnce() override {
            PointersManager pm(LaunchContext::defaultContext(), "DeclarableBenchmark");
            _op->execute(_context);
            pm.synchronize();
        }

        OpBenchmark *clone() override {
            return new DeclarableBenchmark(*_op, _testName);
        }

        std::string shape() override {
            if (_context != nullptr && _context->isFastPath()) {
                std::vector<NDArray*>& ins = _context->fastpath_in();
                std::string s;
                for( int i=0; i<ins.size(); i++ ){
                    if(i > 0){
                        s += "/";
                    }
                    s += ShapeUtils::shapeAsString(_context->getNDArray(i));
                }
                return s;
            } else
                return "N/A";
        }

        std::string dataType() override {
            if (_context != nullptr && _context->isFastPath()){
                std::vector<NDArray*>& ins = _context->fastpath_in();
                std::string s;
                for( int i=0; i<ins.size(); i++ ){
                    if(i > 0){
                        s += "/";
                    }
                    s += DataTypeUtils::asString(_context->getNDArray(i)->dataType());
                }
                return s;
            } else
                return "N/A";
        }

        std::string extra() override {
            if(_context != nullptr){
                std::vector<int>* iargs = _context->getIArguments();
                std::vector<double>* targs = _context->getTArguments();
                std::vector<bool>* bargs = _context->getBArguments();
                std::string e;
                bool any = false;
                if(iargs != nullptr){
                    e += "iargs=[";
                    for( int i=0; i<iargs->size(); i++ ){
                        if(i > 0)
                            e += ",";
                        e += std::to_string(iargs->at(i));
                    }
                    e += "]";
                    any = true;
                }
                if(targs != nullptr){
                    if(any)
                        e += ",";
                    e += "targs=[";
                    for( int i=0; i<targs->size(); i++ ){
                        if(i > 0)
                            e += ",";
                        e += std::to_string(targs->at(i));
                    }
                    e += "]";
                    any = true;
                }
                if(bargs != nullptr){
                    if(any)
                        e += ",";
                    e += "bargs=[";
                    for( int i=0; i<bargs->size(); i++ ){
                        if(i > 0)
                            e += ",";
                        e += std::to_string(bargs->at(i));
                    }
                    e += "]";
                }
                return e;
            }
            return "N/A";
        }

        ~DeclarableBenchmark() {
            if (_context != nullptr)
                delete _context;
        }
    };
}

#endif //DEV_TESTS_DECLARABLEBENCHMARKS_H