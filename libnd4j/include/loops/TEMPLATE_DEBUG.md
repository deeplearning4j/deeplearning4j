# Template Instantiation Debugging Guide for libnd4j

This guide covers tools and techniques for debugging template instantiation issues in libnd4j.

## Finding Undefined Symbols and Template Issues

Basic symbol inspection:
```bash
# List undefined symbols
nm -u libnd4jcpu.so

# List all symbols with demangled names
nm -C libnd4jcpu.so

# List only undefined C++ symbols with demangling
nm -Cu libnd4jcpu.so | grep PairWiseTransform

# Sort by symbol name for easier comparison
nm -Cu libnd4jcpu.so | sort > symbols.txt

# Find specific template instantiations
nm -C libnd4jcpu.so | grep "PairWiseTransform.*long.*int.*char"
```

## Understanding nm Symbol Types

Symbol types in nm output:
- `U` - Undefined symbol
- `T` - Symbol in text (code) section
- `W` - Weak symbol
- `V` - Weak object
- `D` - Initialized data section
- `B` - Uninitialized data section

## Detailed Binary Analysis

Using objdump for deeper inspection:
```bash
# Show all sections and their sizes
objdump -h libnd4jcpu.so

# Disassemble with C++ name demangling
objdump -C -d libnd4jcpu.so

# Show dynamic symbol table
objdump -T libnd4jcpu.so

# Show specific section
objdump -j .text -d libnd4jcpu.so
```

## Symbol Usage and Dependencies

Track symbol usage across files:
```bash
# Find which object files reference a symbol
find . -name "*.o" -exec nm -A {} \; | grep "symbol_name"

# Check dependencies
ldd libnd4jcpu.so

# Show dynamic symbol resolution
LD_DEBUG=symbols ./your_program
```

## Compiler Template Debugging

Compiler flags for template debugging:
```bash
# Show template instantiation stack
g++ -ftemplate-backtrace-limit=0 ...

# Show all template instantiations
g++ -ftemplate-depth=N -v ...

# Verbose output during compilation
VERBOSE=1 make

# Keep temporary files
g++ -save-temps ...
```

## PairWiseTransform Specific Debugging

For PairWiseTransform template issues:
```bash
# Find all PairWiseTransform instantiations
nm -Cu libnd4jcpu.so | grep "PairWiseTransform" | sort > pairwise_instantiations.txt

# Find missing instantiations
nm -u libnd4jcpu.so | grep "PairWiseTransform" > missing_instantiations.txt

# Compare with expected combinations
comm -23 expected_combinations.txt pairwise_instantiations.txt > missing_combinations.txt
```

## CMake Debugging

CMake debugging options:
```bash
# Enable verbose CMake output
cmake -DCMAKE_VERBOSE_MAKEFILE=ON ..

# Show generator expressions
cmake --trace-expand ..

# Debug template generation
cmake --trace-source="pairwise_instantiations.h" ..
```

## Runtime Analysis

Runtime debugging tools:
```bash
# Check dynamic symbol loading
LD_DEBUG=symbols ./your_program

# Track symbol resolution
LD_DEBUG=bindings ./your_program

# Full debug output
LD_DEBUG=all ./your_program
```

## GDB Template Debugging

Using GDB for template debugging:
```bash
# Break on template instantiation
break PairWiseTransform<long long, int, signed char>::exec

# Show template parameters
info variables PairWiseTransform*

# Watch template instantiations
rwatch PairWiseTransform*
```

## Common Template Issues in libnd4j

1. Missing type combinations in partition definitions
2. Template instantiation order issues
3. Undefined symbols from incomplete type coverage
4. Link-time symbol resolution failures

## Best Practices

1. Always check nm output after builds for undefined symbols
2. Use demangled symbol names for better readability
3. Compare expected vs actual template instantiations
4. Track template instantiations through the build process
5. Use CMake verbose output to verify template generation
6. Keep records of required type combinations

## Notes

- Symbol resolution happens at link time
- Template instantiation happens at compile time
- CMake generates instantiations based on type partitions
- Missing combinations often indicate partition coverage gaps

This guide should help diagnose and fix template instantiation issues in libnd4j. The key is systematic investigation from CMake generation through compilation to linking.