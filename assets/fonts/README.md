Place app font files here.

Recommended files for the current UI:

- KronaOne-Regular.ttf
- Actor-Regular.ttf

The app loads `.ttf`, `.otf`, and `.ttc` files from this folder at startup on
Windows, so packaged users do not need to install these fonts separately.

When packaging with PyInstaller, include this folder with the assets folder:

```powershell
pyinstaller --add-data "assets;assets" run.py
```
