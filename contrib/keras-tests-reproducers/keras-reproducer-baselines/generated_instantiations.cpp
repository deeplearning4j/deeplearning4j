#include <iostream>
#include <typeinfo>
#include <string>

// X-Macro List of Data Types
#define DATA_TYPES \
    X(CHAR, char) \
    X(UNSIGNED_CHAR, unsigned char) \
    X(SHORT, short) \
    X(UNSIGNED_SHORT, unsigned short) \
    X(INT, int) \
    X(UNSIGNED_INT, unsigned int) \
    X(LONG, long) \
    X(UNSIGNED_LONG, unsigned long) \
    X(LONG_LONG, long long) \
    X(UNSIGNED_LONG_LONG, unsigned long long) \
    X(FLOAT, float) \
    X(DOUBLE, double) \
    X(LONG_DOUBLE, long double) \
    X(BOOL, bool) \
    X(VOID_PTR, void*) \
    X(CONST_CHAR_PTR, const char*)

// Generate enum class for DataTypes
enum class DataType {
#define X(enum_name, cpp_type) enum_name,
    DATA_TYPES
#undef X
};

// Generate data type tuples using X-Macro
#define X(enum_name, cpp_type) \
    constexpr auto DATA_TYPE_TUPLE_##enum_name = std::make_tuple(DataType::enum_name, cpp_type);
DATA_TYPES
#undef X

// Argument Extraction Macros
#define EXTRACT_FIRST_HELPER(_1, ...) _1
#define EXTRACT_SECOND_HELPER(_1, _2, ...) _2
#define EXTRACT_THIRD_HELPER(_1, _2, _3, ...) _3

#define EXTRACT_FIRST(tuple) EXTRACT_FIRST_HELPER tuple
#define EXTRACT_SECOND(tuple) EXTRACT_SECOND_HELPER tuple
#define EXTRACT_THIRD(tuple) EXTRACT_THIRD_HELPER tuple

// Processor Macros for Instantiation
// Processor for 2D instantiation
#define PROCESSOR_2D(func, args, t1, t2) \
    template void func<EXTRACT_SECOND(t1), EXTRACT_SECOND(t2)>(args);

// Processor for 3D instantiation
#define PROCESSOR_3D(func, args, t1, t2, t3) \
    template void func<EXTRACT_SECOND(t1), EXTRACT_SECOND(t2), EXTRACT_SECOND(t3)>(args);

// Combination Generation Macros
// Helper macro to generate 2D combinations
#define GENERATE_2D_COMBINATIONS(func, args) \
    #define X(enum_name1, cpp_type1) \
        #define X(enum_name2, cpp_type2) \
            PROCESSOR_2D(func, args, DATA_TYPE_TUPLE_##enum_name1, DATA_TYPE_TUPLE_##enum_name2) \
        DATA_TYPES \
        #undef X \
    DATA_TYPES \
    #undef X

// Helper macro to generate 3D combinations
#define GENERATE_3D_COMBINATIONS(func, args) \
    #define X(enum_name1, cpp_type1) \
        #define X(enum_name2, cpp_type2) \
            #define X(enum_name3, cpp_type3) \
                PROCESSOR_3D(func, args, DATA_TYPE_TUPLE_##enum_name1, DATA_TYPE_TUPLE_##enum_name2, DATA_TYPE_TUPLE_##enum_name3) \
            DATA_TYPES \
            #undef X \
        DATA_TYPES \
        #undef X \
    DATA_TYPES \
    #undef X

// Template Function Definitions
// Template function definition for two parameters
template <typename T1, typename T2>
void someFunction(int arg1, int arg2) {
    std::cout << "someFunction instantiated with types: "
              << typeid(T1).name() << ", "
              << typeid(T2).name() << std::endl;
}

// Template function definition for three parameters
template <typename T1, typename T2, typename T3>
void someFunction3(int arg1, int arg2, int arg3) {
    std::cout << "someFunction3 instantiated with types: "
              << typeid(T1).name() << ", "
              << typeid(T2).name() << ", "
              << typeid(T3).name() << std::endl;
}

// Generate Template Instantiations
// Generate all 2D template instantiations
GENERATE_2D_COMBINATIONS(someFunction, (int arg1, int arg2));

// Generate all 3D template instantiations
GENERATE_3D_COMBINATIONS(someFunction3, (int arg1, int arg2, int arg3));

// Main Function
int main() {

    // Example instantiations for 2D
    someFunction<char, unsigned char>(0, 0);
    someFunction<int, float>(0, 0);
    someFunction<double, bool>(0, 0);
    someFunction<void*, const char*>(0, 0);

    // Example instantiations for 3D
    someFunction3<char, unsigned char, short>(0, 0, 0);
    someFunction3<int, float, double>(0, 0, 0);
    someFunction3<double, bool, void*>(0, 0, 0);
    someFunction3<void*, const char*, char>(0, 0, 0);

    return 0;
}