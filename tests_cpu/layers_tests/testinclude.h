//
// Created by agibsonccc on 1/15/17.
//

#ifndef LIBND4J_TESTINCLUDE_H
#define LIBND4J_TESTINCLUDE_H
#include "testlayers.h"
#include <string>
#include <op_boilerplate.h>

//http://stackoverflow.com/questions/228005/alternative-to-itoa-for-converting-integer-to-string-c
FORCEINLINE std::string int_array_to_string(Nd4jLong int_array[], Nd4jLong size_of_array) {
    std::string returnstring = "[";
    for (int temp = 0; temp < size_of_array; temp++) {
        returnstring += std::to_string(int_array[temp]);
        if(temp < size_of_array - 1)
            returnstring += ",";
    }
    returnstring += "]";
    return returnstring;
}

FORCEINLINE ::testing::AssertionResult arrsEquals(Nd4jLong n, Nd4jLong *assertion,Nd4jLong *other) {
    for(int i = 0; i < n; i++) {
        if(assertion[i] != other[i]) {
            std::string message = std::string("Failure at index  ") + std::to_string(i)  + std::string(" assertion: ") +  int_array_to_string(assertion,n) + std::string(" and test array ") + int_array_to_string(other,n) + std::string("  is not equal");
            return ::testing::AssertionFailure() << message;
        }

    }
    return ::testing::AssertionSuccess();

}


#endif //LIBND4J_TESTINCLUDE_H
