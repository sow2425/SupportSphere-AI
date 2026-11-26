@echo off
echo Starting VYRAQ AI...
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo Created .env from .env.example
    )
)
streamlit run app.py
