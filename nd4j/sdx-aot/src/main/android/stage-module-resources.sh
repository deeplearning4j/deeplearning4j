#!/usr/bin/env bash
# Merge module resources into an explicitly compiled class directory without
# silently replacing metadata contributed by another module.

sdx_stage_module_resources() {
  local source_root="${1:?resource source root is required}"
  local target_root="${2:?resource target root is required}"
  local module_id="${3:?module id is required}"
  local source_file relative target_file merge_file

  [[ -d "$source_root" ]] || return 0
  if [[ -L "$source_root" ]]; then
    printf 'Resource root for %s must not be a symlink: %s\n' "$module_id" "$source_root" >&2
    return 1
  fi
  if find "$source_root" -type l -print -quit | grep -q .; then
    printf 'Resources for %s contain a symlink: %s\n' "$module_id" "$source_root" >&2
    return 1
  fi

  mkdir -p -- "$target_root"
  [[ -d "$target_root" && ! -L "$target_root" ]] || {
    printf 'Resource target for %s must be a real directory: %s\n' "$module_id" "$target_root" >&2
    return 1
  }

  while IFS= read -r -d '' source_file; do
    relative="${source_file#"$source_root"/}"
    target_file="$target_root/$relative"
    if [[ -e "$target_file" || -L "$target_file" ]]; then
      if [[ ! -f "$target_file" || -L "$target_file" ]]; then
        printf 'Resource collision while staging %s: %s\n' "$module_id" "$relative" >&2
        return 1
      fi
      if cmp -s -- "$source_file" "$target_file"; then
        continue
      fi
      if [[ "$relative" == META-INF/services/* ]]; then
        merge_file="$(mktemp "$target_file.merge.XXXXXXXX")" || return 1
        if ! LC_ALL=C sort -u -- "$target_file" "$source_file" >"$merge_file"; then
          rm -f -- "$merge_file"
          return 1
        fi
        chmod --reference="$target_file" "$merge_file"
        mv -f -- "$merge_file" "$target_file"
        continue
      fi
      printf 'Resource collision while staging %s: %s\n' "$module_id" "$relative" >&2
      return 1
    fi
    mkdir -p -- "$(dirname -- "$target_file")"
    cp -p -- "$source_file" "$target_file"
  done < <(find "$source_root" -type f -print0 | LC_ALL=C sort -z)
}
