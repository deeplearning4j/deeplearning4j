################################################################################
#
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# See the NOTICE file distributed with this work for additional
# information regarding copyright ownership.

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0
################################################################################

#///////////////////////////////////////////////////////////////////////////////
# genCompilation: Generates cpp, cu files
# INPUT:
# $FILE_ITEM template-configuration that utilizes libnd4j type, macros helpers 
# defined inside { include/types/types.h, include/system/type_boilerplate.h}
# OUTPUT:
# $CUSTOMOPS_GENERIC_SOURCES  generated files will be added into this List
#////////////////////////////////////////////////////////////////////////////////
#  A simple template-configuration file example:
# // hints and defines what types will be generated
# #cmakedefine LIBND4J_TYPE_GEN 
# #cmakedefine FLOAT_TYPE_GEN 
# // below if defines blocks are needed for correctly handling multiple types
# #if  defined(LIBND4J_TYPE_GEN)
#  BUILD_DOUBLE_TEMPLATE(template void someFunc, (arg_list,..), 
#                          LIBND4J_TYPES_@FL_TYPE_INDEX@, INDEXING_TYPES);
# #endif
# #if defined(FLOAT_TYPE_GEN)
#  BUILD_SINGLE_TEMPLATE(template class SomeClass,, FLOAT_TYPES_@FL_TYPE_INDEX@);
# #endif
#////////////////////////////////////////////////////////////////////////////////

function(genCompilation FILE_ITEM) 
    get_filename_component(FILE_ITEM_WE ${FL_ITEM} NAME_WE)  

    set(EXTENSION "cpp")

    if(FL_ITEM MATCHES "cu.in$")
         set(EXTENSION "cu") 
    endif()

    file(READ ${FL_ITEM} CONTENT_FL)
    #check content for types

    #set all to false 
    set (FLOAT_TYPE_GEN     0) 
    set (INT_TYPE_GEN       0) 
    set (LIBND4J_TYPE_GEN   0) 
    set (PAIRWISE_TYPE_GEN  0)
    set (RANGE_STOP         -1)

    string(REGEX MATCHALL "#cmakedefine[ \t]+[^_]+_TYPE_GEN" TYPE_MATCHES ${CONTENT_FL})

    foreach(TYPEX ${TYPE_MATCHES})   
        set(STOP -1)
        if(TYPEX MATCHES "INT_TYPE_GEN$")
           set (INT_TYPE_GEN  1)
           set(STOP 7)
        endif()
        if(TYPEX MATCHES "LIBND4J_TYPE_GEN$")
           set (LIBND4J_TYPE_GEN 1)
           set(STOP 9)
        endif()
        if(TYPEX MATCHES "FLOAT_TYPE_GEN$")
           set (FLOAT_TYPE_GEN 1)
           set(STOP 3)
        endif()
        if(TYPEX MATCHES "PAIRWISE_TYPE_GEN$")
           set (PAIRWISE_TYPE_GEN  1)
           set(STOP 12)
        endif()
        if(STOP GREATER RANGE_STOP) 
           set(RANGE_STOP ${STOP})
        endif()
         
    endforeach()  

    if(RANGE_STOP GREATER -1)
        foreach(FL_TYPE_INDEX RANGE 0 ${RANGE_STOP}) 
            # set OFF if the index is above
            if(FL_TYPE_INDEX GREATER 3)
                 set (FLOAT_TYPE_GEN     0) 
            endif()
            if(FL_TYPE_INDEX GREATER 7)
                 set (INT_TYPE_GEN     0) 
            endif()
            if(FL_TYPE_INDEX GREATER 9)
                 set (LIBND4J_TYPE_GEN   0) 
            endif()        
            set(GENERATED_SOURCE  "${CMAKE_BINARY_DIR}/compilation_units/${FILE_ITEM_WE}_${FL_TYPE_INDEX}.${EXTENSION}")
            configure_file(  "${FL_ITEM}" "${GENERATED_SOURCE}" @ONLY)
            LIST(APPEND CUSTOMOPS_GENERIC_SOURCES ${GENERATED_SOURCE} )
        endforeach()  
    endif()

    set(CUSTOMOPS_GENERIC_SOURCES ${CUSTOMOPS_GENERIC_SOURCES} PARENT_SCOPE)
endfunction()