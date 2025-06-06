import os
import re
import subprocess
import json
from graphviz import Digraph

# CORRECT import based on the source code you provided
from compdb.backend.json import JSONCompilationDatabase

# --- Configuration ---
# The DIRECTORY containing the compile_commands.json file
BUILD_DIRECTORY = './blasbuild/cpu/'
# Project source root to help simplify paths
PROJECT_ROOT = os.path.abspath('.')
# File extensions to consider as project files
PROJECT_FILE_EXTENSIONS = {'.h', '.hpp', '.cpp', '.cxx', '.cu', '.c', '.chpp'}

def is_project_file(path, root_dir):
    """Check if a file belongs to our project and is not a system or external header."""
    if 'flatbuffers-src' in path or 'onednn_external' in path or 'armcompute_external' in path:
        return False
    return path.startswith(root_dir) and os.path.splitext(path)[1] in PROJECT_FILE_EXTENSIONS

def simplify_path(path):
    """Make file paths relative to the project root for cleaner graph nodes."""
    return os.path.relpath(path, PROJECT_ROOT)

def get_dependencies(compile_command_obj):
    """Use the compiler's -M flag to get all dependencies for a file."""
    # The 'arguments' attribute holds the tokenized command
    cmd_list = compile_command_obj.arguments
    source_file = compile_command_obj.file
    directory = compile_command_obj.directory

    base_compiler = cmd_list[0]
    # Build the dependency generation command
    dep_cmd = [base_compiler, '-M'] + [arg for arg in cmd_list[1:] if arg != source_file and not arg.startswith('-o')] + [source_file]

    try:
        output = subprocess.check_output(dep_cmd, stderr=subprocess.PIPE, text=True, cwd=directory)
        deps_str = ' '.join(output.split(':')[1:])
        deps = deps_str.split()
        abs_deps = [os.path.normpath(os.path.join(directory, dep)) for dep in deps]
        return abs_deps
    except subprocess.CalledProcessError as e:
        print(f"Error processing {source_file}:\n{e.stderr}")
        return []

# --- Main Script Logic ---
print("Loading compilation database...")
try:
    # CORRECT usage based on compdb.backend.json.py
    db = JSONCompilationDatabase.probe_directory(BUILD_DIRECTORY)
    if not db:
        raise FileNotFoundError
    all_commands = db.get_all_compile_commands()
except (FileNotFoundError, AttributeError):
    print(f"ERROR: Could not find or load 'compile_commands.json' in '{BUILD_DIRECTORY}'.")
    print("Please ensure you have run CMake and the file has been generated.")
    exit(1)

dot = Digraph('libnd4j_dependencies', comment='Source File Dependencies')
dot.attr('graph', rankdir='LR', splines='true', overlap='false')
dot.attr('node', shape='box', style='rounded,filled', color='skyblue')

edges = set()
project_root_abs = os.path.abspath(PROJECT_ROOT)

print(f"Processing compilation units...")
for i, cmd_obj in enumerate(list(all_commands)):
    # The command object is now a CompileCommand instance from models.py
    source_file = cmd_obj.normfile

    if not is_project_file(source_file, project_root_abs):
        continue

    print(f"[{i+1}] Analyzing: {simplify_path(source_file)}")
    dot.node(simplify_path(source_file))

    dependencies = get_dependencies(cmd_obj)

    for dep_path in dependencies:
        if source_file != dep_path and is_project_file(dep_path, project_root_abs):
            edge = (simplify_path(source_file), simplify_path(dep_path))
            if edge not in edges:
                dot.edge(edge[0], edge[1])
                edges.add(edge)

print("Saving graph...")
try:
    dot.render('dependencies', view=False, format='pdf', cleanup=True)
    print("Graph saved to dependencies.pdf")
except Exception as e:
    print(f"\nError rendering graph. Is Graphviz installed and in your PATH? ({e})")
    print("A file named 'dependencies.dot' was created. You can render it manually:")
    print("dot -Tpdf dependencies.dot -o dependencies.pdf")