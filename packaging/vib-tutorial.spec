# PyInstaller spec for the release builds: `uv run --group build pyinstaller packaging/vib-tutorial.spec`.
# Builds a one-folder app in dist/ (plus a .app bundle on macOS). One folder, not one file,
# because a one-file build unpacks Qt and SciPy to a temp directory on every launch.
import sys

# Qt modules the app never imports. PySide6 ships them all, and they would add ~100 MB+.
EXCLUDES = [
    "PySide6.Qt3DAnimation", "PySide6.Qt3DCore", "PySide6.Qt3DExtras", "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic", "PySide6.Qt3DRender", "PySide6.QtBluetooth", "PySide6.QtCharts",
    "PySide6.QtDataVisualization", "PySide6.QtGraphs", "PySide6.QtLocation",
    "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets", "PySide6.QtNetworkAuth",
    "PySide6.QtNfc", "PySide6.QtPdf", "PySide6.QtPdfWidgets", "PySide6.QtPositioning",
    "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuick3D", "PySide6.QtQuickControls2",
    "PySide6.QtQuickWidgets", "PySide6.QtRemoteObjects", "PySide6.QtScxml",
    "PySide6.QtSensors", "PySide6.QtSerialBus", "PySide6.QtSerialPort", "PySide6.QtSpatialAudio",
    "PySide6.QtSql", "PySide6.QtStateMachine", "PySide6.QtTextToSpeech", "PySide6.QtWebChannel",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineQuick", "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebSockets", "PySide6.QtWebView",
    # pyqtgraph's optional backends and exporters
    "PyQt5", "PyQt6", "PySide2", "matplotlib", "OpenGL", "pyqtgraph.opengl", "h5py",
    "tkinter", "pytest",
]

a = Analysis(
    ["launcher.py"],
    excludes=EXCLUDES,
)

# Qt plugins drag in libraries the app doesn't use: the virtual keyboard pulls in QML and
# Quick, the PDF image format pulls in QtPdf. Drop them, and Qt's translations, by path
# (the names differ per OS: libQt6Quick.so, Qt6Quick.dll, QtQuick.framework).
DROP = ("virtualkeyboard", "qtquick", "qt6quick", "qtqml", "qt6qml", "qtpdf", "qt6pdf", "qpdf",
        "translations")


def keep(entry):
    path = entry[0].lower().replace("\\", "/")
    return not any(word in path for word in DROP)


a.binaries = [entry for entry in a.binaries if keep(entry)]
a.datas = [entry for entry in a.datas if keep(entry)]

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="vib-tutorial",
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="vib-tutorial", upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Vibration Tutorial.app",
        bundle_identifier="io.github.wookielumberjack.vib-tutorial",
        info_plist={"NSHighResolutionCapable": True},
    )
