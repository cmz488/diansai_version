#!/usr/bin/env bash
# Run one command in a private mount namespace with the working systemd resolver.
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "用法: $0 <command> [args...]" >&2
    exit 2
fi

if getent ahosts pypi.org >/dev/null 2>&1; then
    exec "$@"
fi

resolver=/run/systemd/resolve/stub-resolv.conf
if [[ ! -s "$resolver" ]]; then
    echo "DNS 修复失败: $resolver 不存在或为空" >&2
    exit 1
fi
if ! command -v unshare >/dev/null 2>&1; then
    echo "DNS 修复失败: unshare 不可用" >&2
    exit 1
fi

cache_root="${XDG_CACHE_HOME:-$HOME/.cache}/tspfile"
private_etc="$cache_root/etc-dns"
mkdir -p "$private_etc"
cp -a /etc/. "$private_etc/" 2>/dev/null || true
rm -f "$private_etc/resolv.conf"
ln -s "$resolver" "$private_etc/resolv.conf"

exec unshare -Urm bash -c '
    set -e
    private_etc=$1
    shift
    mount --bind "$private_etc" /etc
    exec "$@"
' bash "$private_etc" "$@"
