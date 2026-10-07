#!/bin/sh
set -eu

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
platform=darwin-arm64
share="share/jev-local"

fail() {
    printf 'build-dist: %s\n' "$1" >&2
    exit 1
}

version=$(sed -n 's/.*"version": *"\([^"]*\)".*/\1/p' "$root/$share/runtime.json")
case "$version" in
    ''|*[!0-9A-Za-z.+-]*) fail "could not read the version from $share/runtime.json (got '$version')." ;;
esac
for file in "$share/server.py" "$share/runtime.json" bin/jev-local LICENSE NOTICE; do
    [ -f "$root/$file" ] || fail "missing $file."
done

dist="$root/dist"
tree="$dist/stage/jev-local"
rm -rf "$dist/stage"
mkdir -p "$tree/bin" "$tree/$share"
cp "$root/bin/jev-local" "$tree/bin/jev-local"
cp "$root/$share/server.py" "$root/$share/runtime.json" "$tree/$share/"
cp "$root/LICENSE" "$root/NOTICE" "$tree/"
find "$tree" -type d -exec chmod 0755 {} +
find "$tree" -type f -exec chmod 0644 {} +
chmod 0755 "$tree/bin/jev-local"

name="jev-local-$version-$platform.tar.gz"
scratch=$(mktemp -d "$dist/.pack.XXXXXX")
trap 'rm -rf "$scratch"' EXIT
if [ "$(uname -s)" = Darwin ]; then
    COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs -czf "$scratch/$name" -C "$dist/stage" jev-local
else
    tar -czf "$scratch/$name" -C "$dist/stage" jev-local
fi
if command -v sha256sum >/dev/null 2>&1; then
    sum=$(cd "$scratch" && sha256sum "$name")
else
    sum=$(cd "$scratch" && shasum -a 256 "$name")
fi
printf '%s\n' "$sum" > "$scratch/$name.sha256"
mv -f "$scratch/$name" "$dist/$name"
mv -f "$scratch/$name.sha256" "$dist/$name.sha256"
printf '%s\n' "$dist/$name"
