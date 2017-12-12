//
// @author raver119@gmail.com
//

#include <array/DataType.h>
#include <graph/generated/array_generated.h>
#include <op_boilerplate.h>

namespace nd4j {
    class DataTypeUtils {
    public:
        static int asInt(DataType type);
        static DataType fromInt(int dtype);
        static DataType fromFlatDataType(nd4j::graph::DataType dtype);

        template <typename T>
        static DataType fromT();
        static size_t sizeOfElement(DataType type);

        // returns the smallest finite value of the given type
        template <typename T>
        FORCEINLINE static T min();

        // returns the largest finite value of the given type
        template <typename T>
        FORCEINLINE static T max();

        // returns the difference between 1.0 and the next representable value of the given floating-point type 
        template <typename T>
        FORCEINLINE static T eps();
        
    };


//////////////////////////////////////////////////////////////////////////
///// IMLEMENTATION OF INLINE METHODS ///// 
//////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////
// returns the smallest finite value of the given type
template<typename T>
FORCEINLINE T DataTypeUtils::min() {
    
    switch (sizeof(T)) {
        case 8:                // T = double            
            return std::numeric_limits<double>::min();    
        case 4:                // T = float            
            return std::numeric_limits<float>::min();    
        case 2:                // T = float16        
            return 6.1035e-05;    
        default:
            throw("DataTypeUtils::min function: type of T is undefined !");    
        }
}

///////////////////////////////////////////////////////////////////
// returns the largest finite value of the given type
template <typename T>
FORCEINLINE T DataTypeUtils::max() {

	switch (sizeof(T)) {
        case 8:                // T = double            
            return std::numeric_limits<double>::max();    
        case 4:                // T = float            
            return std::numeric_limits<float>::max();    
        case 2:                // T = float16        
            return 65504.;    
        default:
            throw("DataTypeUtils::max function: type of T is undefined !");    
        }
}


///////////////////////////////////////////////////////////////////
// returns the difference between 1.0 and the next representable value of the given floating-point type 
template <typename T>
FORCEINLINE T DataTypeUtils::eps() {

	switch (sizeof(T)) {
        case 8:                // T = double            
            return std::numeric_limits<double>::epsilon();    
        case 4:                // T = float            
            return std::numeric_limits<float>::epsilon();    
        case 2:                // T = float16        
            return 0.00097656;    
        default:
            throw("DataTypeUtils::epsilon function: type of T is undefined !");    
        }
}


}