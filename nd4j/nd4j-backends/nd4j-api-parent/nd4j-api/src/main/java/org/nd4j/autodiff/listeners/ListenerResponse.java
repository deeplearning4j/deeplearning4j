/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.listeners;

/**
 * An enum representing feedback given by listeners during the training loop.<br>
 * CONTINUE: Continue training for more epochs, unless the specified (maximum) number of training epochs have already been completed.<br>
 * STOP: Terminate training at the current point, irrespective of how many total epochs were specified when calling fit.<br>
 */
public enum ListenerResponse {
    CONTINUE, STOP;
}
