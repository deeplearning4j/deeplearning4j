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
// @author raver119@gmail.com
//

#ifndef SD_PLATFORM_BOILERPLATE_H
#define SD_PLATFORM_BOILERPLATE_H

#include <execution/Engine.h>



#define CONCATP(A,B) A ##_##B


#define DECLARE_PLATFORM_F(NAME, ENGINE, CNAME)      class ND4J_EXPORT PLATFORM_##CNAME : public PlatformHelper {\
                                                     public: \
                                                        PLATFORM_##CNAME() :  PlatformHelper(#NAME, samediff::Engine::ENGINE) { } \
                                                        bool isUsable(graph::Context &context) override; \
                                                        Nd4jStatus invokeHelper(graph::Context &context) override; \
                                                    };

#define DECLARE_PLATFORM(NAME, ENGINE) DECLARE_PLATFORM_F(NAME, ENGINE, NAME ##_## ENGINE)

#define PLATFORM_IMPL_F(NAME, ENGINE, CNAME)         struct ND4J_EXPORT __registratorPlatformHelper_##CNAME { \
                                                        __registratorPlatformHelper_##CNAME() { \
                                                            auto helper = new PLATFORM_##CNAME(); \
                                                            OpRegistrator::getInstance()->registerHelper(helper); \
                                                        } \
                                                    }; \
                                                    static __registratorPlatformHelper_##CNAME platformHelper_##CNAME; \
                                                    Nd4jStatus PLATFORM_##CNAME::invokeHelper(nd4j::graph::Context &block)


#define PLATFORM_IMPL(NAME, ENGINE) PLATFORM_IMPL_F(NAME, ENGINE, NAME ##_## ENGINE)


#define PLATFORM_CHECK_F(NAME, ENGINE, CNAME)        bool PLATFORM_##CNAME::isUsable(graph::Context &block)
#define PLATFORM_CHECK(NAME, ENGINE) PLATFORM_CHECK_F(NAME, ENGINE, NAME ##_## ENGINE)


#endif //SD_PLATFORM_BOILERPLATE_H
