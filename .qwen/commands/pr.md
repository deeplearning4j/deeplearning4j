Create or update a pull request. $ARGUMENTS

Follow these steps:
1. Run `git status` to check for uncommitted changes
2. Run `git branch --show-current` to get the current branch
3. Run `git log main..HEAD --oneline` (or appropriate base branch) to see all commits
4. Run `git diff main...HEAD` to see all changes relative to base
5. Check if branch has a remote tracking branch: `git rev-parse --abbrev-ref @{upstream} 2>/dev/null`
6. Draft a PR title (under 70 chars) and description with:
   - ## Summary: 1-3 bullet points of key changes
   - ## Test plan: How to verify the changes
7. Push the branch if needed: `git push -u origin HEAD`
8. Create the PR:
   ```
   gh pr create --title "title" --body "$(cat <<'EOF'
   ## Summary
   - ...

   ## Test plan
   - ...
   EOF
   )"
   ```

If uncommitted changes exist, ask the user if they want to commit first.
Return the PR URL when done.
