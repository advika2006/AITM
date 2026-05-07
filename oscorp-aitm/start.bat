@echo off
echo Starting Oscorp AITM...

:: Start the Streamlit backend in the background without opening the browser
start "Backend" cmd /c "python -m streamlit run app.py --server.headless true"

:: Start the Next.js frontend
cd frontend
start "Frontend" cmd /c "npm run dev"

:: Wait a few seconds for servers to initialize
timeout /t 5 /nobreak > NUL

:: Open the homepage in the default browser
start http://localhost:3000

echo Application started! Close this window to keep the servers running in the background.
