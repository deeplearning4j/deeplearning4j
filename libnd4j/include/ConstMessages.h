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

#ifndef LIBND4J_CONST_MESSAGES_H
#define LIBND4J_CONST_MESSAGES_H

namespace sd{

    #define IS_EMPTY_MSG_INPUT_ "the Emptiness of the Input NDArray"
    #define IS_EMPTY_MSG_OUTPUT_ "the Emptiness of the Output NDArray"
    #define RANK_MSG_INPUT_ "the Rank of the Input NDArray"
    #define LENGTH_MSG_INPUT_ "the Length of the Input NDArray"
    #define SHAPE_MSG_INPUT_ "the Shape of the Input NDArray"
    #define SHAPE_MSG_OUTPUT_ "the Shape of the Output NDArray"
    #define RANK_MSG_OUTPUT_ "the Rank of the Output NDArray"
    #define TYPE_MSG_INPUT_ "the Type of the Input NDArray"
    #define TYPE_MSG_OUTPUT_ "the Type of the Output NDArray"
    #define EWS_MSG_INPUT_ "the EWS of the Input NDArray"
    #define EWS_MSG_OUTPUT_ "the EWS of the Output NDArray"
    #define ORDERING_MSG_INPUT_ "the Ordering of the Input NDArray"
    #define ORDERING_MSG_OUTPUT_ "the Ordering of the Output NDArray"

    extern const char* OP_VALIDATION_FAIL_MSG;
    extern const char* HAVE_SEQLENARR;
    extern const char* HAVE_PEEPHOLE;
    extern const char* MSG_CELL_CLIPPING;
    extern const char* TYPECHECK_MSG;
    extern const char* NO_MSG;
    extern const char* EXPECTED_TRUE;
    extern const char* EXPECTED_FALSE;
    extern const char* EXPECTED_NOT_SUPPORTED;
    extern const char* EXPECTED_EQ_MSG;
    extern const char* EXPECTED_NE_MSG;
    extern const char* EXPECTED_LT_MSG;
    extern const char* EXPECTED_LE_MSG;
    extern const char* EXPECTED_GT_MSG;
    extern const char* EXPECTED_GE_MSG;

    extern const char* EXPECTED_IN;

    extern const char* IS_EMPTY_MSG_INPUT;
    extern const char* IS_EMPTY_MSG_INPUT0;
    extern const char* IS_EMPTY_MSG_INPUT1;
    extern const char* IS_EMPTY_MSG_INPUT2;
    extern const char* IS_EMPTY_MSG_INPUT3;

    extern const char* RANK_MSG_INPUT;
    extern const char* RANK_MSG_INPUT0;
    extern const char* RANK_MSG_INPUT1;
    extern const char* RANK_MSG_INPUT2;
    extern const char* RANK_MSG_INPUT3;

    extern const char* LENGTH_MSG_INPUT;
    extern const char* LENGTH_MSG_INPUT0;
    extern const char* LENGTH_MSG_INPUT1;
    extern const char* LENGTH_MSG_INPUT2;
    extern const char* LENGTH_MSG_INPUT3;

    extern const char* SHAPE_MSG_INPUT;
    extern const char* SHAPE_MSG_INPUT0;
    extern const char* SHAPE_MSG_INPUT1;
    extern const char* SHAPE_MSG_INPUT2;
    extern const char* SHAPE_MSG_INPUT3;

    extern const char* TYPE_MSG_INPUT;
    extern const char* TYPE_MSG_INPUT0;
    extern const char* TYPE_MSG_INPUT1;
    extern const char* TYPE_MSG_INPUT2;
    extern const char* TYPE_MSG_INPUT3;

    extern const char* EWS_MSG_INPUT;
    extern const char* EWS_MSG_INPUT0;
    extern const char* EWS_MSG_INPUT1;
    extern const char* EWS_MSG_INPUT2;
    extern const char* EWS_MSG_INPUT3;

    extern const char* ORDERING_MSG_INPUT;
    extern const char* ORDERING_MSG_INPUT0;
    extern const char* ORDERING_MSG_INPUT1;
    extern const char* ORDERING_MSG_INPUT2;
    extern const char* ORDERING_MSG_INPUT3;

    extern const char* RANK_MSG_OUTPUT;
    extern const char* RANK_MSG_OUTPUT0;
    extern const char* RANK_MSG_OUTPUT1;

    extern const char* IS_EMPTY_OUTPUT;
    extern const char* IS_EMPTY_OUTPUT0;
    extern const char* IS_EMPTY_OUTPUT1;

    extern const char* TYPE_MSG_OUTPUT;
    extern const char* TYPE_MSG_OUTPUT0;
    extern const char* TYPE_MSG_OUTPUT1;

    extern const char* EWS_MSG_OUTPUT;
    extern const char* EWS_MSG_OUTPUT0;
    extern const char* EWS_MSG_OUTPUT1;

    extern const char* ORDERING_MSG_OUTPUT;
    extern const char* ORDERING_MSG_OUTPUT0;
    extern const char* ORDERING_MSG_OUTPUT1;

    extern const char* SHAPE_MSG_OUTPUT;
    extern const char* SHAPE_MSG_OUTPUT0;
    extern const char* SHAPE_MSG_OUTPUT1;

    extern const char* IS_USE_ONEDNN_MSG;
    extern const char * ONEDNN_STREAM_NOT_SUPPORTED;

    extern const char* REQUIREMENTS_MEETS_MSG;
    extern const char* REQUIREMENTS_FAILS_MSG;

}

#endif
