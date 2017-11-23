//
// @author raver119@gmail.com
//

#include <array/DataType.h>
#include <graph/generated/array_generated.h>

namespace nd4j {
    class DataTypeUtils {
    public:
        static int asInt(DataType type);
        static DataType fromInt(int dtype);
        static DataType fromFlatDataType(nd4j::graph::DataType dtype);

        template <typename T>
        static DataType fromT();
        static size_t sizeOfElement(DataType type);
    };
}