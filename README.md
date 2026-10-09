# JOLODHAR 💧

JOLODHAR is a smart IoT-ML based drain and water monitoring system. It uses
machine learning (Random Forest) to analyse sensor and climate data, including
rainfall, temperature, humidity and SPI drought index, across 38 districts of
Bangladesh. The dashboard shows live updates over WebSocket and maps, and can
export PDF reports with Bengali font support.

## Features
- Real-time monitoring dashboard (Flask + Socket.IO)
- ML prediction models using scikit-learn (Random Forest)
- JWT-based login and user management
- Interactive maps and charts (Leaflet, Chart.js)
- PDF report generation (ReportLab, Bengali supported)
- SQLite database

## Setup
pip install -r requirements.txt
python app.py
Open http://localhost:5000
