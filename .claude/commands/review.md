Review the current uncommitted changes for code quality. $ARGUMENTS

Follow these steps:
1. Run `git diff` to see unstaged changes
2. Run `git diff --cached` to see staged changes
3. If no local changes, run `git log -1 --format=%H` and `git diff HEAD~1` to review the last commit
4. Read any changed files that need more context
5. Analyze for:
   - **Bugs**: Logic errors, off-by-one, null safety, race conditions
   - **Security**: Injection, hardcoded secrets, unsafe operations
   - **Performance**: Unnecessary allocations, N+1 queries, missing caching
   - **Style**: Naming, organization, DRY violations, dead code
   - **Error handling**: Missing try/catch, unclosed resources, swallowed errors
   - **Tests**: Missing test coverage for new/changed code paths

Format your review as:
- **Critical** (must fix): ...
- **Important** (should fix): ...
- **Minor** (nice to fix): ...
- **Positive**: Good patterns worth noting
