Create a git commit for the current changes. $ARGUMENTS

Follow these steps:
1. Run `git status` to see all changed files (never use -uall flag)
2. Run `git diff` to see staged and unstaged changes
3. Run `git log --oneline -5` to see recent commit message style
4. Analyze the changes and draft a concise commit message that:
   - Summarizes what changed and why (not just "what")
   - Follows the repository's existing commit message conventions
   - Is 1-2 sentences for the subject line
5. Stage the relevant files (prefer specific files over `git add -A`)
   - Do NOT stage files that look like secrets (.env, credentials, keys)
6. Create the commit using a heredoc for the message:
   ```
   git commit -m "$(cat <<'EOF'
   Your commit message here.

   Co-Authored-By: kompile-cli <noreply@kompile.ai>
   EOF
   )"
   ```
7. Run `git status` to verify the commit succeeded

If no changes are found, inform the user. Do not create empty commits.
