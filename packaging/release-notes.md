## Downloads

Standalone builds need no Python install. Download the one for your system, unpack it and run it.
They are not code-signed, so the operating system warns the first time you open one.

| System | File | How to run |
|---|---|---|
| Windows 10/11 (64-bit) | `vib-tutorial-*-windows-x64.zip` | Unzip, then run `vib-tutorial\vib-tutorial.exe`. If SmartScreen says "Windows protected your PC", click **More info → Run anyway**. |
| macOS (Apple Silicon) | `vib-tutorial-*-macos-arm64.zip` | Unzip to get **Vibration Tutorial.app**. If macOS says the app "is damaged" or "can't be opened", run `xattr -dr com.apple.quarantine "Vibration Tutorial.app"` in Terminal once, then open it. |
| Linux (x86-64) | `vib-tutorial-*-linux-x86_64.tar.gz` | `tar xzf vib-tutorial-*-linux-x86_64.tar.gz`, then run `vib-tutorial/vib-tutorial`. Needs an X11 or Wayland desktop. |

If a build doesn't run on your system (for example an Intel Mac), run from source with
[uv](https://docs.astral.sh/uv/) instead: `uvx --from git+https://github.com/WookieLumberjack/vib_tutorial@TAG vib-tutorial`,
with `TAG` replaced by this release's tag.

This software is for educational use only, with no warranty; see the README's disclaimer.
