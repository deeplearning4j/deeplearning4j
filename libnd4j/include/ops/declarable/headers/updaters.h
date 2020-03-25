/*******************************************************************************
 * Copyright (c) 2019-2020 Konduit K.K.
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
 // @author Oleh Semeniv (oleg.semeniv@gmail.com)
 // 


#ifndef LIBND4J_HEADERS_UPDATERS_H
#define LIBND4J_HEADERS_UPDATERS_H

#include <ops/declarable/headers/common.h>
#include <ops/declarable/CustomOperations.h>  
#include <helpers/ConstantTadHelper.h>
#include <execution/Threads.h>
#include <ops/declarable/helpers/updatersHelpers.h>


namespace sd {
    namespace ops {


        /**
         * SGD updater
         * Input arrays:
         * 0 - input array with gradients.
         * Optional:
         * 1 - scalar learning rate value
         * Optional:
         * T args
         * 0 - scalar learning rate value
         */
#if NOT_EXCLUDED(OP_sgd_updater)
        DECLARE_CONFIGURABLE_OP(sgd_updater, 1, 1, true, 0, 0);
#endif

        /**
        * RmsPropUpdater updater
        * Input arrays:
        * 0 - input array with gradients.
        * 1 - Initial state
        * Optional:
        * 2 - scalar learning rate value
        * 3 - scalar rms decay
        * 4 - epsilon
        * Optional:
        * T args
        * 0 - scalar learning rate value
        * 1 - scalar rms decay
        * 2 - epsilon
        */
#if NOT_EXCLUDED(OP_rms_prop_updater)
        DECLARE_CONFIGURABLE_OP(rms_prop_updater, 2, 2, true, 0, 0);
#endif
        // AdaGrad
        /* Input arrays :
        *  0 - input array with gradients.
        *  1 - historical grad state  
        * Optional :
        * 2 - scalar learning rate value
        * 3 - epsilon
        * Optional:
        * T args
        * 0 - scalar learning rate value
        * 1 - epsilon
        */
#if NOT_EXCLUDED(OP_ada_grad_updater)
            DECLARE_CONFIGURABLE_OP(ada_grad_updater, 2, 2, true, 0, 0);
#endif
         // AdaMax
         /* Input arrays :
         *  0 - input array with gradients.
         *  1 - gradient state V
         *  2 - gradient state M
         * Optional :
         * 3 - scalar learning rate value
         * 4 - beta 1 value
         * 5 - beta 2 value
         * 6 - epsilon
         * Optional:
         * T args
         * 0 - scalar learning rate value
         * 1 - beta 1 value
         * 2 - beta 2 value
         * 3 - epsilon
         * Optional:
         * I args
         * 0 - iteration
         */
#if NOT_EXCLUDED(OP_ada_max_updater)
            DECLARE_CONFIGURABLE_OP(ada_max_updater, 3, 3, true, 0, 0);
#endif
            // Nesterov's momentum
            /* Input arrays :
            *  0 - input array with gradients.
            *  1 - V grad state 
            * Optional :
            * 2 - scalar learning rate value
            * 3 - scalar momentum value
            * Optional:
            * T args
            * 0 - learning rate value
            * 1 - momentum value
            */
#if NOT_EXCLUDED(OP_nesterovs_updater)
            DECLARE_CONFIGURABLE_OP(nesterovs_updater, 2, 2, true, 0, 0);
#endif
            // Adam
            /* Input arrays :
            *  0 - input array with gradients.
            *  1 - gradient state V
            *  2 - gradient state M
            * Optional :
            * 3 - scalar learning rate value
            * 4 - beta 1 value
            * 5 - beta 2 value
            * 6 - epsilon
            * Optional:
            * T args
            * 0 - scalar learning rate value
            * 1 - beta 1 value
            * 2 - beta 2 value
            * 3 - epsilon
            * Optional:
            * I args
            * 0 - iteration
            */
#if NOT_EXCLUDED(OP_adam_updater)
            DECLARE_CONFIGURABLE_OP(adam_updater, 3, 3, true, 0, 0);
#endif
         // AdaDelta
         /* Input arrays :
         *  0 - input array with gradients.
         *  1 - gradient state V
         *  2 - gradient state M
         * Optional :
         * 3 - rho value
         * 6 - epsilon
         * Optional:
         * T args
         * 0 - rho
         * 1 - epsilon
         */
#if NOT_EXCLUDED(OP_ada_delta_updater)
            DECLARE_CONFIGURABLE_OP(ada_delta_updater, 3, 3, true, 0, 0);
#endif
            // Nadam
            /* Input arrays :
            *  0 - input array with gradients.
            *  1 - gradient state V
            *  2 - gradient state M
            * Optional :
            * 3 - scalar learning rate value
            * 4 - beta 1 value
            * 5 - beta 2 value
            * 6 - epsilon
            * Optional:
            * T args
            * 0 - scalar learning rate value
            * 1 - beta 1 value
            * 2 - beta 2 value
            * 3 - epsilon
            * Optional:
            * I args
            * 0 - iteration
            */
#if NOT_EXCLUDED(OP_nadam_updater)
            DECLARE_CONFIGURABLE_OP(nadam_updater, 3, 3, true, 0, 0);
#endif    
            // AmsGrad
            /* Input arrays :
            *  0 - input array with gradients.
            *  1 - gradient state V - sqrd gradients
            *  2 - gradient state M - moving avg
            *  3 - gradient state H - max
            * Optional :
            * 4 - scalar learning rate value
            * 5 - beta 1 value
            * 6 - beta 2 value
            * 7 - epsilon
            * Optional:
            * T args
            * 0 - scalar learning rate value
            * 1 - beta 1 value
            * 2 - beta 2 value
            * 3 - epsilon
            * Optional:
            * I args
            * 0 - iteration
            */
#if NOT_EXCLUDED(OP_ams_grad_updater)
            DECLARE_CONFIGURABLE_OP(ams_grad_updater, 4, 4, true, 0, 0);
#endif    
}
}

#endif
