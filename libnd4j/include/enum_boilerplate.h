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
//  @author raver119@gmail.com
//

#ifndef LIBND4J_ENUM_BOILERPLATE_H
#define LIBND4J_ENUM_BOILERPLATE_H

#include <type_boilerplate.h>


#define EN_1(WHAT, OP_PAIR) WHAT(OP_PAIR)
#define EN_2(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_1(WHAT,  __VA_ARGS__))
#define EN_3(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_2(WHAT,  __VA_ARGS__))
#define EN_4(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_3(WHAT,  __VA_ARGS__))
#define EN_5(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_4(WHAT,  __VA_ARGS__))
#define EN_6(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_5(WHAT,  __VA_ARGS__))
#define EN_7(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_6(WHAT,  __VA_ARGS__))
#define EN_8(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_7(WHAT,  __VA_ARGS__))
#define EN_9(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_8(WHAT,  __VA_ARGS__))
#define EN_10(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_9(WHAT,  __VA_ARGS__))
#define EN_11(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_10(WHAT,  __VA_ARGS__))
#define EN_12(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_11(WHAT,  __VA_ARGS__))
#define EN_13(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_12(WHAT,  __VA_ARGS__))
#define EN_14(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_13(WHAT,  __VA_ARGS__))
#define EN_15(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_14(WHAT,  __VA_ARGS__))
#define EN_16(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_15(WHAT,  __VA_ARGS__))
#define EN_17(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_16(WHAT,  __VA_ARGS__))
#define EN_18(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_17(WHAT,  __VA_ARGS__))
#define EN_19(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_18(WHAT,  __VA_ARGS__))
#define EN_20(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_19(WHAT,  __VA_ARGS__))
#define EN_21(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_20(WHAT,  __VA_ARGS__))
#define EN_22(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_21(WHAT,  __VA_ARGS__))
#define EN_23(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_22(WHAT,  __VA_ARGS__))
#define EN_24(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_23(WHAT,  __VA_ARGS__))
#define EN_25(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_24(WHAT,  __VA_ARGS__))
#define EN_26(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_25(WHAT,  __VA_ARGS__))
#define EN_27(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_26(WHAT,  __VA_ARGS__))
#define EN_28(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_27(WHAT,  __VA_ARGS__))
#define EN_29(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_28(WHAT,  __VA_ARGS__))
#define EN_30(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_29(WHAT,  __VA_ARGS__))
#define EN_31(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_30(WHAT,  __VA_ARGS__))
#define EN_32(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_31(WHAT,  __VA_ARGS__))
#define EN_33(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_32(WHAT,  __VA_ARGS__))
#define EN_34(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_33(WHAT,  __VA_ARGS__))
#define EN_35(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_34(WHAT,  __VA_ARGS__))
#define EN_36(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_35(WHAT,  __VA_ARGS__))
#define EN_37(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_36(WHAT,  __VA_ARGS__))
#define EN_38(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_37(WHAT,  __VA_ARGS__))
#define EN_39(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_38(WHAT,  __VA_ARGS__))
#define EN_40(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_39(WHAT,  __VA_ARGS__))
#define EN_41(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_40(WHAT,  __VA_ARGS__))
#define EN_42(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_41(WHAT,  __VA_ARGS__))
#define EN_43(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_42(WHAT,  __VA_ARGS__))
#define EN_44(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_43(WHAT,  __VA_ARGS__))
#define EN_45(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_44(WHAT,  __VA_ARGS__))
#define EN_46(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_45(WHAT,  __VA_ARGS__))
#define EN_47(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_46(WHAT,  __VA_ARGS__))
#define EN_48(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_47(WHAT,  __VA_ARGS__))
#define EN_49(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_48(WHAT,  __VA_ARGS__))
#define EN_50(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_49(WHAT,  __VA_ARGS__))
#define EN_51(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_50(WHAT,  __VA_ARGS__))
#define EN_52(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_51(WHAT,  __VA_ARGS__))
#define EN_53(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_52(WHAT,  __VA_ARGS__))
#define EN_54(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_53(WHAT,  __VA_ARGS__))
#define EN_55(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_54(WHAT,  __VA_ARGS__))
#define EN_56(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_55(WHAT,  __VA_ARGS__))
#define EN_57(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_56(WHAT,  __VA_ARGS__))
#define EN_58(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_57(WHAT,  __VA_ARGS__))
#define EN_59(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_58(WHAT,  __VA_ARGS__))
#define EN_60(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_59(WHAT,  __VA_ARGS__))
#define EN_61(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_60(WHAT,  __VA_ARGS__))
#define EN_62(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_61(WHAT,  __VA_ARGS__))
#define EN_63(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_62(WHAT,  __VA_ARGS__))
#define EN_64(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_63(WHAT,  __VA_ARGS__))
#define EN_65(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_64(WHAT,  __VA_ARGS__))
#define EN_66(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_65(WHAT,  __VA_ARGS__))
#define EN_67(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_66(WHAT,  __VA_ARGS__))
#define EN_68(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_67(WHAT,  __VA_ARGS__))
#define EN_69(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_68(WHAT,  __VA_ARGS__))
#define EN_70(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_69(WHAT,  __VA_ARGS__))
#define EN_71(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_70(WHAT,  __VA_ARGS__))
#define EN_72(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_71(WHAT,  __VA_ARGS__))
#define EN_73(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_72(WHAT,  __VA_ARGS__))
#define EN_74(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_73(WHAT,  __VA_ARGS__))
#define EN_75(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_74(WHAT,  __VA_ARGS__))
#define EN_76(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_75(WHAT,  __VA_ARGS__))
#define EN_77(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_76(WHAT,  __VA_ARGS__))
#define EN_78(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_77(WHAT,  __VA_ARGS__))
#define EN_79(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_78(WHAT,  __VA_ARGS__))
#define EN_80(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_79(WHAT,  __VA_ARGS__))
#define EN_81(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_80(WHAT,  __VA_ARGS__))
#define EN_82(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_81(WHAT,  __VA_ARGS__))
#define EN_83(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_82(WHAT,  __VA_ARGS__))
#define EN_84(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_83(WHAT,  __VA_ARGS__))
#define EN_85(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_84(WHAT,  __VA_ARGS__))
#define EN_86(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_85(WHAT,  __VA_ARGS__))
#define EN_87(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_86(WHAT,  __VA_ARGS__))
#define EN_88(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_87(WHAT,  __VA_ARGS__))
#define EN_89(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_88(WHAT,  __VA_ARGS__))
#define EN_90(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_89(WHAT,  __VA_ARGS__))
#define EN_91(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_90(WHAT,  __VA_ARGS__))
#define EN_92(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_91(WHAT,  __VA_ARGS__))
#define EN_93(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_92(WHAT,  __VA_ARGS__))
#define EN_94(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_93(WHAT,  __VA_ARGS__))
#define EN_95(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_94(WHAT,  __VA_ARGS__))
#define EN_96(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_95(WHAT,  __VA_ARGS__))
#define EN_97(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_96(WHAT,  __VA_ARGS__))
#define EN_98(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_97(WHAT,  __VA_ARGS__))
#define EN_99(WHAT, OP_PAIR, ...) WHAT(OP_PAIR)EVAL(EN_98(WHAT,  __VA_ARGS__))

#define __EXPAND_ENUM(NUM, CLASS) CLASS,
#define _EXPAND_ENUM(OP_PAIR) EVALUATING_PASTE(__EXPAND, _ENUM(UNPAREN(OP_PAIR)))

#define GET_MACROS_ENUM(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62, _63, _64, _65, _66, _67, _68, _69, _70, _71, _72, _73, _74, _75, _76, _77, _78, _79, _80, _81, _82, _83, _84, _85, _86, _87, _88, _89, _90, _91, _92, _93, _94, _95, _96, _97, _98, _99, NAME,...) NAME

#define FOR_EACH_ENUM(WHAT, ...) EXPAND(GET_MACROS_ENUM(__VA_ARGS__,  EN_99, EN_98, EN_97, EN_96, EN_95, EN_94, EN_93, EN_92, EN_91, EN_90, EN_89, EN_88, EN_87, EN_86, EN_85, EN_84, EN_83, EN_82, EN_81, EN_80, EN_79, EN_78, EN_77, EN_76, EN_75, EN_74, EN_73, EN_72, EN_71, EN_70, EN_69, EN_68, EN_67, EN_66, EN_65, EN_64, EN_63, EN_62, EN_61, EN_60, EN_59, EN_58, EN_57, EN_56, EN_55, EN_54, EN_53, EN_52, EN_51, EN_50, EN_49, EN_48, EN_47, EN_46, EN_45, EN_44, EN_43, EN_42, EN_41, EN_40, EN_39, EN_38, EN_37, EN_36, EN_35, EN_34, EN_33, EN_32, EN_31, EN_30, EN_29, EN_28, EN_27, EN_26, EN_25, EN_24, EN_23, EN_22, EN_21, EN_20, EN_19, EN_18, EN_17, EN_16, EN_15, EN_14, EN_13, EN_12, EN_11, EN_10, EN_9, EN_8, EN_7, EN_6, EN_5, EN_4, EN_3, EN_2, EN_1)(WHAT, __VA_ARGS__))
#define _EXEC_ENUM(WHAT, ...) EVAL(FOR_EACH_ENUM(WHAT, __VA_ARGS__))
#define EXEC_ENUMERATOR(...) EXPAND(_EXEC_ENUM(_EXPAND_ENUM, __VA_ARGS__))

#define BUILD_ENUMERATION(OPS) EVAL(EXEC_ENUMERATOR(OPS))

#endif