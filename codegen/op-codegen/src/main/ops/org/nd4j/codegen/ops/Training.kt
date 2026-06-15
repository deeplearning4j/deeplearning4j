/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.codegen.ops

import org.nd4j.codegen.api.DataType.*
import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.dsl.*

fun Training() = Namespace("Training") {
    val updaterPkg = "org.nd4j.linalg.api.ops.impl.updaters"

    Op("sgdUpdater") {
        javaPackage = updaterPkg
        javaOpClass = "SgdUpdater"
        Input(NUMERIC, "gradients") { description = "Gradients array" }
        Arg(NUMERIC, "lr") { description = "Learning rate" }
        Output(NUMERIC, "output") { description = "Updated gradients (in-place)" }
        Doc(Language.ANY, DocScope.ALL) {
            """
            SGD (Stochastic Gradient Descent) updater.
            Applies gradient update: param -= lr * gradient
            """.trimIndent()
        }
    }

    Op("nesterovsUpdater") {
        javaPackage = updaterPkg
        javaOpClass = "NesterovsUpdater"
        Input(NUMERIC, "gradients") { description = "Gradients array" }
        Input(NUMERIC, "state") { description = "Updater state (velocity)" }
        Arg(NUMERIC, "lr") { description = "Learning rate" }
        Arg(NUMERIC, "momentum") { description = "Momentum coefficient" }
        Output(NUMERIC, "updates") { description = "Updated gradients" }
        Output(NUMERIC, "updatedState") { description = "Updated velocity state" }
        Doc(Language.ANY, DocScope.ALL) {
            """
            Nesterov momentum SGD updater.
            Applies Nesterov accelerated gradient update using momentum.
            """.trimIndent()
        }
    }

    Op("adaGradUpdater") {
        javaPackage = updaterPkg
        javaOpClass = "AdaGradUpdater"
        Input(NUMERIC, "gradients") { description = "Gradients array" }
        Input(NUMERIC, "state") { description = "Updater state (accumulated squared gradients)" }
        Arg(NUMERIC, "lr") { description = "Learning rate" }
        Arg(NUMERIC, "epsilon") { description = "Epsilon for numerical stability" }
        Output(NUMERIC, "updates") { description = "Updated gradients" }
        Output(NUMERIC, "updatedState") { description = "Updated accumulated squared gradients state" }
        Doc(Language.ANY, DocScope.ALL) {
            """
            AdaGrad updater.
            Adapts the learning rate for each parameter based on accumulated squared gradients.
            See: Duchi et al. (2011) - Adaptive Subgradient Methods
            """.trimIndent()
        }
    }

    Op("adaDeltaUpdater") {
        javaPackage = updaterPkg
        javaOpClass = "AdaDeltaUpdater"
        Input(NUMERIC, "gradients") { description = "Gradients array" }
        Input(NUMERIC, "stateMsg") { description = "Updater state: mean squared gradients" }
        Input(NUMERIC, "stateMsdx") { description = "Updater state: mean squared delta x" }
        Arg(NUMERIC, "rho") { description = "Decay rate (rho)" }
        Arg(NUMERIC, "epsilon") { description = "Epsilon for numerical stability" }
        Output(NUMERIC, "updates") { description = "Updated gradients" }
        Output(NUMERIC, "updatedStateMsg") { description = "Updated mean squared gradients state" }
        Output(NUMERIC, "updatedStateMsdx") { description = "Updated mean squared delta x state" }
        Doc(Language.ANY, DocScope.ALL) {
            """
            AdaDelta updater.
            An adaptive learning rate method that uses a moving window of gradient updates to adapt the learning rate.
            See: Zeiler (2012) - ADADELTA: An Adaptive Learning Rate Method
            """.trimIndent()
        }
    }

    Op("rmsPropUpdater") {
        javaPackage = updaterPkg
        javaOpClass = "RmsPropUpdater"
        Input(NUMERIC, "gradients") { description = "Gradients array" }
        Input(NUMERIC, "state") { description = "Updater state (mean squared gradients)" }
        Arg(NUMERIC, "lr") { description = "Learning rate" }
        Arg(NUMERIC, "decay") { description = "Decay rate (rho)" }
        Arg(NUMERIC, "epsilon") { description = "Epsilon for numerical stability" }
        Output(NUMERIC, "updates") { description = "Updated gradients" }
        Output(NUMERIC, "updatedState") { description = "Updated mean squared gradients state" }
        Doc(Language.ANY, DocScope.ALL) {
            """
            RMSProp updater.
            Divides the learning rate by a running average of the magnitudes of recent gradients.
            """.trimIndent()
        }
    }

    Op("adamUpdater") {
        javaPackage = updaterPkg
        javaOpClass = "AdamUpdater"
        Input(NUMERIC, "gradients") { description = "Gradients array" }
        Input(NUMERIC, "stateU") { description = "Updater state: first moment (mean)" }
        Input(NUMERIC, "stateM") { description = "Updater state: second moment (uncentered variance)" }
        Arg(NUMERIC, "lr") { description = "Learning rate" }
        Arg(NUMERIC, "beta1") { description = "Beta1 - exponential decay rate for first moment estimate" }
        Arg(NUMERIC, "beta2") { description = "Beta2 - exponential decay rate for second moment estimate" }
        Arg(NUMERIC, "epsilon") { description = "Epsilon for numerical stability" }
        Arg(INT, "iteration") { description = "Current iteration (used for bias correction)" }
        Output(NUMERIC, "updates") { description = "Updated gradients" }
        Output(NUMERIC, "updatedStateU") { description = "Updated first moment state" }
        Output(NUMERIC, "updatedStateM") { description = "Updated second moment state" }
        Doc(Language.ANY, DocScope.ALL) {
            """
            Adam updater.
            Adaptive Moment Estimation - computes adaptive learning rates for each parameter.
            See: Kingma and Ba (2014) - Adam: A Method for Stochastic Optimization
            """.trimIndent()
        }
    }

    Op("nadamUpdater") {
        javaPackage = updaterPkg
        javaOpClass = "NadamUpdater"
        Input(NUMERIC, "gradients") { description = "Gradients array" }
        Input(NUMERIC, "stateV") { description = "Updater state: second moment (uncentered variance)" }
        Input(NUMERIC, "stateM") { description = "Updater state: first moment (mean)" }
        Arg(NUMERIC, "lr") { description = "Learning rate" }
        Arg(NUMERIC, "beta1") { description = "Beta1 - exponential decay rate for first moment estimate" }
        Arg(NUMERIC, "beta2") { description = "Beta2 - exponential decay rate for second moment estimate" }
        Arg(NUMERIC, "epsilon") { description = "Epsilon for numerical stability" }
        Arg(INT, "iteration") { description = "Current iteration (used for bias correction)" }
        Output(NUMERIC, "updates") { description = "Updated gradients" }
        Output(NUMERIC, "updatedStateV") { description = "Updated second moment state" }
        Output(NUMERIC, "updatedStateM") { description = "Updated first moment state" }
        Doc(Language.ANY, DocScope.ALL) {
            """
            Nadam updater.
            Adam with Nesterov momentum. Combines Adam and Nesterov accelerated gradient.
            """.trimIndent()
        }
    }

    Op("adaMaxUpdater") {
        javaPackage = updaterPkg
        javaOpClass = "AdaMaxUpdater"
        Input(NUMERIC, "gradients") { description = "Gradients array" }
        Input(NUMERIC, "stateU") { description = "Updater state: infinity norm (max)" }
        Input(NUMERIC, "stateM") { description = "Updater state: first moment (mean)" }
        Arg(NUMERIC, "lr") { description = "Learning rate" }
        Arg(NUMERIC, "beta1") { description = "Beta1 - exponential decay rate for first moment estimate" }
        Arg(NUMERIC, "beta2") { description = "Beta2 - exponential decay rate for infinity norm" }
        Arg(NUMERIC, "epsilon") { description = "Epsilon for numerical stability" }
        Arg(INT, "iteration") { description = "Current iteration (used for bias correction)" }
        Output(NUMERIC, "updates") { description = "Updated gradients" }
        Output(NUMERIC, "updatedStateU") { description = "Updated infinity norm state" }
        Output(NUMERIC, "updatedStateM") { description = "Updated first moment state" }
        Doc(Language.ANY, DocScope.ALL) {
            """
            AdaMax updater.
            A variant of Adam based on the infinity norm, which can be more stable in some settings.
            See: Kingma and Ba (2014) - Adam: A Method for Stochastic Optimization (Section 7.1)
            """.trimIndent()
        }
    }

    Op("amsGradUpdater") {
        javaPackage = updaterPkg
        javaOpClass = "AmsGradUpdater"
        Input(NUMERIC, "gradients") { description = "Gradients array" }
        Input(NUMERIC, "stateV") { description = "Updater state: second moment (uncentered variance)" }
        Input(NUMERIC, "stateM") { description = "Updater state: first moment (mean)" }
        Input(NUMERIC, "stateH") { description = "Updater state: maximum of past second moments" }
        Arg(NUMERIC, "lr") { description = "Learning rate" }
        Arg(NUMERIC, "beta1") { description = "Beta1 - exponential decay rate for first moment estimate" }
        Arg(NUMERIC, "beta2") { description = "Beta2 - exponential decay rate for second moment estimate" }
        Arg(NUMERIC, "epsilon") { description = "Epsilon for numerical stability" }
        Arg(INT, "iteration") { description = "Current iteration (used for bias correction)" }
        Output(NUMERIC, "updates") { description = "Updated gradients" }
        Output(NUMERIC, "updatedStateV") { description = "Updated second moment state" }
        Output(NUMERIC, "updatedStateM") { description = "Updated first moment state" }
        Output(NUMERIC, "updatedStateH") { description = "Updated maximum second moment state" }
        Doc(Language.ANY, DocScope.ALL) {
            """
            AMSGrad updater.
            A variant of Adam that uses the maximum of past squared gradients to provide
            better convergence guarantees.
            See: Reddi et al. (2018) - On the Convergence of Adam and Beyond
            """.trimIndent()
        }
    }

    Op("adaBeliefUpdater") {
        javaPackage = updaterPkg
        javaOpClass = "AdaBeliefUpdater"
        Input(NUMERIC, "gradients") { description = "Gradients array" }
        Input(NUMERIC, "stateU") { description = "Updater state: exponential moving average of squared gradient deviation" }
        Input(NUMERIC, "stateM") { description = "Updater state: first moment (mean)" }
        Arg(NUMERIC, "lr") { description = "Learning rate" }
        Arg(NUMERIC, "beta1") { description = "Beta1 - exponential decay rate for first moment estimate" }
        Arg(NUMERIC, "beta2") { description = "Beta2 - exponential decay rate for second moment estimate" }
        Arg(NUMERIC, "epsilon") { description = "Epsilon for numerical stability" }
        Arg(INT, "iteration") { description = "Current iteration (used for bias correction)" }
        Output(NUMERIC, "updates") { description = "Updated gradients" }
        Output(NUMERIC, "updatedStateU") { description = "Updated EMA of squared gradient deviation state" }
        Output(NUMERIC, "updatedStateM") { description = "Updated first moment state" }
        Doc(Language.ANY, DocScope.ALL) {
            """
            AdaBelief updater.
            Adapts the step size by the difference between the predicted and observed gradients
            (the 'belief' in the gradient direction).
            See: Zhuang et al. (2020) - AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients
            https://arxiv.org/pdf/2010.07468.pdf
            """.trimIndent()
        }
    }
}
