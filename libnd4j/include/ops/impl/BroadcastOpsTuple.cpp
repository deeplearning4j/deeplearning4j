/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
#include <ops/BroadcastOpsTuple.h>

namespace sd {
    BroadcastOpsTuple BroadcastOpsTuple::custom(sd::scalar::Ops scalar, sd::pairwise::Ops pairwise, sd::broadcast::Ops broadcast) {
        BroadcastOpsTuple t(scalar, pairwise, broadcast);
        return t;
    }

    BroadcastOpsTuple BroadcastOpsTuple::Add() {
        return custom(sd::scalar::Add, sd::pairwise::Add, sd::broadcast::Add);
    }

    BroadcastOpsTuple BroadcastOpsTuple::Assign() {
        return custom(sd::scalar::CopyPws, sd::pairwise::CopyPws, sd::broadcast::CopyPws);
    }

    BroadcastOpsTuple BroadcastOpsTuple::Divide() {
        return custom(sd::scalar::Divide, sd::pairwise::Divide, sd::broadcast::Divide);
    }

    BroadcastOpsTuple BroadcastOpsTuple::DivideNoNan() {
        return custom(sd::scalar::DivideNoNan, sd::pairwise::DivideNoNan, sd::broadcast::DivideNoNan);
    }

    BroadcastOpsTuple BroadcastOpsTuple::Multiply() {
        return custom(sd::scalar::Multiply, sd::pairwise::Multiply, sd::broadcast::Multiply);
    }

    BroadcastOpsTuple BroadcastOpsTuple::Subtract() {
        return custom(sd::scalar::Subtract, sd::pairwise::Subtract, sd::broadcast::Subtract);
    }
    BroadcastOpsTuple BroadcastOpsTuple::IGamma() {
        return custom(sd::scalar::IGamma, sd::pairwise::IGamma, sd::broadcast::IGamma);
    }
    BroadcastOpsTuple BroadcastOpsTuple::IGammac() {
        return custom(sd::scalar::IGammac, sd::pairwise::IGammac, sd::broadcast::IGammac);
    }


    BroadcastOpsTuple BroadcastOpsTuple::Pow() {
        return custom(sd::scalar::Pow, sd::pairwise::Pow, sd::broadcast::Pow);
    }
    BroadcastOpsTuple BroadcastOpsTuple::PowDerivative() {
        return custom(sd::scalar::PowDerivative, sd::pairwise::PowDerivative, sd::broadcast::PowDerivative);
    }

}
