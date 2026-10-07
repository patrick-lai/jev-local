#!/bin/sh
set -eu

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)

fail() {
    printf 'pack-npm: %s\n' "$1" >&2
    exit 1
}

for tool in node npm; do
    command -v "$tool" >/dev/null 2>&1 || fail "$tool is required."
done

version=$(sed -n 's/.*"version": *"\([^"]*\)".*/\1/p' "$root/share/jev-local/runtime.json")
case "$version" in
    ''|*[!0-9A-Za-z.+-]*) fail "could not read the version from share/jev-local/runtime.json (got '$version')." ;;
esac
tree="$root/dist/stage/jev-local"
[ -x "$tree/bin/jev-local" ] || fail "missing $tree; run scripts/build-dist.sh first."

out="$root/dist/npm"
stage=$(mktemp -d "$root/dist/.npm-stage.XXXXXX")
trap 'rm -rf "$stage"' EXIT
rm -rf "$out"
mkdir -p "$out"
cp -R "$tree" "$stage/package"
cp "$root/README.md" "$stage/package/README.md"
node -e '
const fs = require("node:fs");
const [source, target, version] = process.argv.slice(1);
const manifest = JSON.parse(fs.readFileSync(source, "utf8"));
manifest.version = version;
fs.writeFileSync(target, JSON.stringify(manifest, null, 2) + "\n");
' "$root/npm/jev-local/package.json" "$stage/package/package.json" "$version"
(cd "$stage/package" && npm pack --ignore-scripts --loglevel=warn --pack-destination "$out" >/dev/null)
for file in "$out"/*.tgz; do
    printf '%s\n' "$file"
done
