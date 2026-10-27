@echo off
title Push to GitHub: lostxmusafir/S-ocr
echo ============================================================
echo Pushing commits to https://github.com/lostxmusafir/S-ocr.git
echo ============================================================
echo.
git push -u origin main
echo.
echo ============================================================
if %errorlevel% equ 0 (
    echo [SUCCESS] Successfully pushed all commits to GitHub!
) else (
    echo [NOTE] If prompted for GitHub login, please sign in or use a GitHub Personal Access Token (PAT).
)
echo ============================================================
pause
