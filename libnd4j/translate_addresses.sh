#!/bin/bash

# Usage: ./translate_addresses.sh <library_path> <base_address> <address1> <address2> ...

if [ $# -lt 3 ]; then
    echo "Usage: $0 <library_path> <base_address> <address1> <address2> ..."
    exit 1
fi

LIBRARY_PATH=$1
BASE_ADDRESS=$2
shift 2

echo "Library: $LIBRARY_PATH"
echo "Base Address: $BASE_ADDRESS"
echo "======================================"

for ADDRESS in "$@"; do
    echo "Analyzing address: $ADDRESS"
    echo "--------------------------------------"

    # Calculate the offset within the library
    OFFSET=$(printf "0x%x" $((ADDRESS - BASE_ADDRESS)))
    echo "Offset in library: $OFFSET"

    # Use addr2line to get file and line number (if debug symbols are available)
    echo "addr2line output:"
    addr2line -e "$LIBRARY_PATH" "$OFFSET"
    echo "--------------------------------------"

    # Use nm to find the closest symbol
    echo "nm output:"
    nm -C "$LIBRARY_PATH" | awk '{print "0x"$1, $3}' | sort | grep -i "$OFFSET"
    echo "--------------------------------------"

    # Use c++filt to demangle C++ symbols (if applicable)
    SYMBOL=$(nm -C "$LIBRARY_PATH" | awk '{print "0x"$1, $3}' | sort | grep -i "$OFFSET" | tail -1 | awk '{print $2}')
    if [ -n "$SYMBOL" ]; then
        echo "Demangled symbol:"
        c++filt "$SYMBOL"
    else
        echo "No symbol found for offset: $OFFSET"
    fi

    echo "======================================"
done