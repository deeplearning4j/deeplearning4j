//
//  @author raver119@gmail.com
//

#ifndef ND4J_STATUS_H
#define ND4J_STATUS_H

#include <pointercast.h>
#include <op_boilerplate.h>
#include <dll.h>
#include <helpers/logger.h>

namespace nd4j {
    class ND4J_EXPORT Status {
    public:
        static FORCEINLINE Nd4jStatus OK() {
            return ND4J_STATUS_OK;
        };

        static FORCEINLINE Nd4jStatus CODE(Nd4jStatus code, const char *message) {
            nd4j_printf("%s\n", message);
            return code;
        }

        static FORCEINLINE Nd4jStatus THROW(const char *message = nullptr) {
            if (message != nullptr) {
                nd4j_printf("%s\n", message);
            }
            return ND4J_STATUS_KERNEL_FAILURE;
        }
    };
}

#endif // STATUS_H