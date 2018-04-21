//
// Created by raver119 on 20/04/18.
//

#ifndef LIBND4J_STRINGUTILS_H
#define LIBND4J_STRINGUTILS_H

#include <pointercast.h>
#include <op_boilerplate.h>
#include <string>
#include <sstream>

namespace nd4j {
    class StringUtils {
    public:
        template <typename T>
        static FORCEINLINE std::string valueToString(T value) {
            std::ostringstream os;

            os << value ;

            //convert the string stream into a string and return
            return os.str() ;
        };
    };
}


#endif //LIBND4J_STRINGUTILS_H
