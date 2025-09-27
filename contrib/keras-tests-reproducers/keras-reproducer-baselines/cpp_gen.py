# Define the data types
data_types = [
    ("CHAR", "char"),
    ("UNSIGNED_CHAR", "unsigned char"),
    ("SHORT", "short"),
    ("UNSIGNED_SHORT", "unsigned short"),
    ("INT", "int"),
    ("UNSIGNED_INT", "unsigned int"),
    ("LONG", "long"),
    ("UNSIGNED_LONG", "unsigned long"),
    ("LONG_LONG", "long long"),
    ("UNSIGNED_LONG_LONG", "unsigned long long"),
    ("FLOAT", "float"),
    ("DOUBLE", "double"),
    ("LONG_DOUBLE", "long double"),
    ("BOOL", "bool"),
    ("VOID_PTR", "void*"),
    ("CONST_CHAR_PTR", "const char*")
]


# Helper macros
def generate_apply_macro():
    return (
        "// Macro to apply another macro\n"
        "#define APPLY(macro, args) macro args\n"
    )


def generate_extract_second_macro():
    return (
        "// Macro to extract the second element of a tuple\n"
        "#define EXTRACT_SECOND(tuple) APPLY(EXTRACT_SECOND_HELPER, tuple)\n"
        "#define EXTRACT_SECOND_HELPER(_, second) second\n"
    )


# Macro processor for 2D combinations
def generate_2d_combinations():
    output = "#define GENERATE_2D_COMBINATIONS(func, args) \\\n"
    for i, (enum_name1, cpp_type1) in enumerate(data_types):
        for j, (enum_name2, cpp_type2) in enumerate(data_types):
            output += f"    PROCESSOR_2D(func, args, DATA_TYPE_TUPLE_{enum_name1}, DATA_TYPE_TUPLE_{enum_name2})"
            if i < len(data_types) - 1 or j < len(data_types) - 1:
                output += " \\\n"
    output += "\n"
    return output


# Macro processor for 3D combinations
def generate_3d_combinations():
    output = "#define GENERATE_3D_COMBINATIONS(func, args) \\\n"
    for i, (enum_name1, cpp_type1) in enumerate(data_types):
        for j, (enum_name2, cpp_type2) in enumerate(data_types):
            for k, (enum_name3, cpp_type3) in enumerate(data_types):
                output += f"    PROCESSOR_3D(func, args, DATA_TYPE_TUPLE_{enum_name1}, DATA_TYPE_TUPLE_{enum_name2}, DATA_TYPE_TUPLE_{enum_name3})"
                if i < len(data_types) - 1 or j < len(data_types) - 1 or k < len(data_types) - 1:
                    output += " \\\n"
    output += "\n"
    return output


# Generate data type tuples
def generate_data_type_tuples():
    output = "// Data type tuples\n"
    for enum_name, cpp_type in data_types:
        output += f"#define DATA_TYPE_TUPLE_{enum_name} DataType::{enum_name}, {cpp_type}\n"
    return output


# Generate the processor macros
def generate_processor_macros():
    output = "#define PROCESSOR_2D(func, args, t1, t2) INSTANTIATE_FUNCTION_2(func, args, (t1, t2))\n\n"
    output += "#define PROCESSOR_3D(func, args, t1, t2, t3) INSTANTIATE_FUNCTION_3(func, args, (t1, t2, t3))\n\n"
    return output


# Generate function instantiation macros
def generate_instantiate_macros():
    output = "#define INSTANTIATE_FUNCTION_2(func, args, types) \\\n"
    output += "    template void func<EXTRACT_SECOND(GET_ARG_1 types), EXTRACT_SECOND(GET_ARG_2 types)>args;\n\n"

    output += "#define INSTANTIATE_FUNCTION_3(func, args, types) \\\n"
    output += "    template void func<EXTRACT_SECOND(GET_ARG_1 types), EXTRACT_SECOND(GET_ARG_2 types), EXTRACT_SECOND(GET_ARG_3 types)>args;\n\n"

    return output


# Generate the argument extraction macros
def generate_argument_extraction_macros():
    return (
        "#define GET_ARG_1(tuple) GET_ARG_1_HELPER tuple\n"
        "#define GET_ARG_2(tuple) GET_ARG_2_HELPER tuple\n"
        "#define GET_ARG_3(tuple) GET_ARG_3_HELPER tuple\n"
        "#define GET_ARG_1_HELPER(first, second) first\n"
        "#define GET_ARG_2_HELPER(first, second) second\n"
        "#define GET_ARG_3_HELPER(first, second, third) third\n"
    )


# Generate actual function declarations and template instantiations
def generate_function_declarations():
    return (
        "template<typename T1, typename T2>\n"
        "void someFunction(T1 arg1, T2 arg2) {\n"
        "    // Function logic goes here\n"
        "}\n\n"
        "template<typename T1, typename T2, typename T3>\n"
        "void someFunction3(T1 arg1, T2 arg2, T3 arg3) {\n"
        "    // Function logic goes here\n"
        "}\n\n"
    )


# Full C++ code generation with function instantiations
def generate_complete_cpp():
    script = "// Generated C++ code for 2D and 3D combinations\n\n"

    # Add function declarations
    script += generate_function_declarations() + "\n\n"

    # Add macros
    script += generate_data_type_tuples() + "\n\n"
    script += generate_apply_macro() + "\n\n"
    script += generate_extract_second_macro() + "\n\n"
    script += generate_argument_extraction_macros() + "\n\n"
    script += generate_processor_macros() + "\n\n"
    script += generate_instantiate_macros() + "\n\n"
    script += generate_2d_combinations() + "\n\n"
    script += generate_3d_combinations() + "\n\n"

    # Generate all instantiations for 2D combinations
    script += "// Instantiate all 2D combinations\n"
    for enum_name1, cpp_type1 in data_types:
        for enum_name2, cpp_type2 in data_types:
            script += f"template void someFunction<{cpp_type1}, {cpp_type2}>(int arg1, int arg2);\n"

    # Generate all instantiations for 3D combinations
    script += "\n// Instantiate all 3D combinations\n"
    for enum_name1, cpp_type1 in data_types:
        for enum_name2, cpp_type2 in data_types:
            for enum_name3, cpp_type3 in data_types:
                script += f"template void someFunction3<{cpp_type1}, {cpp_type2}, {cpp_type3}>(int arg1, int arg2, int arg3);\n"

    return script


# Write the complete C++ code to a file
with open('generated_complete_code.cpp', 'w') as f:
    f.write(generate_complete_cpp())

print("Full code generation complete. Output written to 'generated_complete_code.cpp'.")
