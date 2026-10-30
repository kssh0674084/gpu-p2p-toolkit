@echo off
chcp 65001 >nul
cd /d %~dp0
echo ============================================
echo   GPU P2P 部署助手 - 打包为单文件 EXE
echo ============================================

echo [1/3] 安装依赖 (flask / paramiko / pyinstaller)...
pip install -r requirements.txt pyinstaller || goto :err

echo [2/3] 清理旧构建...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist GPUP2PToolkit.spec del /q GPUP2PToolkit.spec

echo [3/3] 开始打包...
rem 用 python -m 调用，避免 PATH 上残留其他环境的旧版 PyInstaller/Flask
rem --collect-all webview/clr_loader: 打包 pywebview 的 WebView2 加载器与 .NET 桥
python -m PyInstaller --onefile --noconsole --name GPUP2PToolkit --distpath dist_new --add-data "templates;templates" --icon gpu_p2p.ico --collect-all webview --collect-all clr_loader app.py || goto :err

echo.
echo ============================================
echo   打包完成: dist\GPUP2PToolkit.exe
echo ============================================
pause
exit /b 0

:err
echo.
echo [错误] 打包失败，请检查上方错误信息
pause
exit /b 1
