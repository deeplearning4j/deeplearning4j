#!/usr/bin/env bash
# Stable, content-addressed manifests for checked-in and local source inputs.
#
# Tracked executable modes come from the Git index so a build tool touching a file's
# filesystem mode cannot redirect a retry into a different native build directory.
# Untracked source inputs retain their filesystem mode because the index has no contract
# for them.
sdx_git_source_manifest() {
  local repository_root="${1:?repository root is required}"
  shift
  local entry metadata object_and_stage relative file mode digest
  local -a roots=("$@")
  local -A tracked_modes=()
  local -A tracked_objects=()

  while IFS= read -r -d '' entry; do
    metadata="${entry%%$'\t'*}"
    relative="${entry#*$'\t'}"
    tracked_modes["$relative"]="${metadata%% *}"
    object_and_stage="${metadata#* }"
    tracked_objects["$relative"]="${object_and_stage%% *}"
  done < <(git -C "$repository_root" ls-files -s -z --cached -- "${roots[@]}")

  git -C "$repository_root" ls-files -z --cached --others --exclude-standard -- "${roots[@]}" |
    LC_ALL=C sort -z |
    while IFS= read -r -d '' relative; do
      # Runtime receipts bind production inputs. Test sources and fixtures are allowed to
      # be generated or materialized by Maven and must not fork a native generation.
      case "$relative" in
        */src/test/*|libnd4j/tests/*|libnd4j/tests_*/*|libnd4j/cmake/tests/*) continue ;;
      esac
      file="$repository_root/$relative"
      mode="${tracked_modes[$relative]:-}"
      if [[ -f "$file" ]]; then
        [[ -n "$mode" ]] || mode="$(stat -c '%a' "$file")"
        digest="$(sha256sum "$file" | cut -d ' ' -f 1)"
      elif [[ -n "${tracked_objects[$relative]:-}" ]]; then
        # Maven and code generators may materialize tracked generated sources that were
        # absent from a sparse/clean working tree. Bind their index content from the start;
        # a generated result that differs from the index will still fail the final guard.
        digest="$(git -C "$repository_root" cat-file blob "${tracked_objects[$relative]}" |
          sha256sum | cut -d ' ' -f 1)"
      else
        continue
      fi
      printf '%s\0%s\0%s\0' "$relative" "$mode" "$digest"
    done
}

sdx_git_source_manifest_sha256() {
  sdx_git_source_manifest "$@" | sha256sum | cut -d ' ' -f 1
}
