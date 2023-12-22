#!/bin/bash

# Navigating to the methdemon1d submodule directory
cd resources/methdemon1d

# Running the Makefile to compile the project
make all

# Assuming the executable is named 'methdemon1d' and is located in the current directory after compilation
# Change the name 'methdemon1d' below if the actual name of the executable is different

# Check if the executable exists
if [ ! -f "bin/methdemon" ]; then
    echo "Compilation failed or executable not found."
    exit 1
fi

# Navigating back to the parent directory
cd ../../

# Creating a bin directory if it doesn't exist
mkdir -p bin

# Moving the executable to the bin directory
mv resources/methdemon1d/bin/methdemon bin/

echo "Compilation successful. Executable moved to bin directory."
