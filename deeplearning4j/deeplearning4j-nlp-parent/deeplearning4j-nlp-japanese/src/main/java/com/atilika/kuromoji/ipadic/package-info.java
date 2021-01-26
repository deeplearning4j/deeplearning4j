/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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

/*-*
 * A Japanese morphological analyzer based on the IPADIC dictionary.
 * <p>
 * This dictionary provides a basic set of features and suits many tasks.
 * If you are not sure about which dictionary to use, this one is a good starting point.
 * <p>
 * The following token features are supported:
 * <ul>
 * <li>surface form (表層形)
 * <li>part of speech level 1 (品詞細分類1)
 * <li>part of speech level 2 (品詞細分類2)
 * <li>part of speech level 3 (品詞細分類3)
 * <li>part of speech level 4 (品詞細分類4)
 * <li>conjugation type (活用型)
 * <li>conjugation form (活用形)
 * <li>base form (基本形)
 * <li>reading (読み)
 * <li>pronunciation (発音)
 * </ul>
 */
package com.atilika.kuromoji.ipadic;
