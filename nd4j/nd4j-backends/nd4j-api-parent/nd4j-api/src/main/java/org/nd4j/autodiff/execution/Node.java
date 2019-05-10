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

package org.nd4j.autodiff.execution;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.ToString;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;

/**
 * Intermediate Node representation
 *
 * @author raver119@gmail.com
 */
@Data
@Slf4j
@NoArgsConstructor
@ToString(exclude = {"opExecAction"})
public class Node {
    private int id;
    private String name;
    private List<Integer> input = new ArrayList<>();
    private List<Integer> output = new ArrayList<>();
    private List<Integer> unresolved = new ArrayList<>();
    private int[] originalOutput;
}
