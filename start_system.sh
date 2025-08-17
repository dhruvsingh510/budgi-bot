# To start backend
cd backend
pip install -r requirements.txt
condasetup && conda activate budgibot-backend
python main.py

# To start frontend
cd ../frontend
pip install -r requirements.txt
condasetup && conda activate budgibot-frontend
streamlit run app.py