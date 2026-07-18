Explain the code or recent changes. $ARGUMENTS

Follow these steps:
1. If args specify a file or function, read that directly
2. Otherwise, run `git diff` to see recent changes
3. If no uncommitted changes, run `git log -1 --format=%H` and `git show HEAD` to see the last commit
4. Read relevant source files for full context
5. Provide a clear explanation covering:
   - **What**: What the code does at a high level
   - **How**: Key implementation details and algorithms
   - **Why**: Design decisions and trade-offs
   - **Dependencies**: What this code interacts with
   - **Edge cases**: Important boundary conditions

Use simple language. Reference specific file:line locations.
If explaining changes, focus on what changed and why.
