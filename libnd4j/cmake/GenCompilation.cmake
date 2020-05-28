function(genCompilation FILE_ITEM)
                get_filename_component(FILE_ITEM_WE ${FL_ITEM} NAME_WE)  

                set(EXTENSION "cpp")

                if(FL_ITEM MATCHES "cu.in$")
                     set(EXTENSION "cu") 
                endif()

                file(READ ${FL_ITEM} CONTENT_FL)
                #check content for types

                #set all to false
                set (INT_TYPE_GEN  0) 
                set (FLOAT_TYPE_GEN  0) 
                set (PAIRWISE_TYPE_GEN  0)

                string(REGEX MATCHALL "#cmakedefine01[ \t]+[^_]+_TYPE_GEN" TYPE_MATCHES ${CONTENT_FL})
                #message("+++++++++++++++++++++++++++++++")
                foreach(TYPEX ${TYPE_MATCHES})   
                    if(TYPEX MATCHES "INT_TYPE_GEN$")
                       set (INT_TYPE_GEN  1)  
                    endif()
                    if(TYPEX MATCHES "FLOAT_TYPE_GEN$")
                       set (FLOAT_TYPE_GEN 1)
                    endif()
                    if(TYPEX MATCHES "PAIRWISE_TYPE_GEN$")
                       set (PAIRWISE_TYPE_GEN  1)
                    endif()
                     
                endforeach()  

                foreach(FL_TYPE_INDEX RANGE 0 7) 
                    configure_file(  "${FL_ITEM}" "${CMAKE_BINARY_DIR}/compilation_units/${FILE_ITEM_WE}_${FL_TYPE_INDEX}.${EXTENSION}" @ONLY)
                    LIST(APPEND CUSTOMOPS_GENERIC_SOURCES ${CMAKE_BINARY_DIR}/compilation_units/${FILE_ITEM_WE}_${FL_TYPE_INDEX}.${EXTENSION} )
                endforeach()  
                #do not generate for INT_TYPE_GEN
                set (INT_TYPE_GEN  0) 
                if(FLOAT_TYPE_GEN)
                    foreach(FL_TYPE_INDEX RANGE 7 9) 
                        configure_file(  "${FL_ITEM}" "${CMAKE_BINARY_DIR}/compilation_units/${FILE_ITEM_WE}_${FL_TYPE_INDEX}.${EXTENSION}" @ONLY)
                        LIST(APPEND CUSTOMOPS_GENERIC_SOURCES ${CMAKE_BINARY_DIR}/compilation_units/${FILE_ITEM_WE}_${FL_TYPE_INDEX}.${EXTENSION} )
                    endforeach() 
                endif()
                #do not generate for FLOAT_TYPE_GEN
                set (FLOAT_TYPE_GEN  0) 
                if(PAIRWISE_TYPE_GEN)
                    foreach(FL_TYPE_INDEX RANGE 9 12) 
                        configure_file(  "${FL_ITEM}" "${CMAKE_BINARY_DIR}/compilation_units/${FILE_ITEM_WE}_${FL_TYPE_INDEX}.${EXTENSION}" @ONLY)
                        LIST(APPEND CUSTOMOPS_GENERIC_SOURCES ${CMAKE_BINARY_DIR}/compilation_units/${FILE_ITEM_WE}_${FL_TYPE_INDEX}.${EXTENSION} )
                    endforeach() 
                endif()

endfunction()