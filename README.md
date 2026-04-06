[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=22793464)

### 1. Edit .env file

```bash
cp .env.example .env
```

### 2. Install Dependencies

if you prefer running in virtual environment please run
```bash
python -m venv venv
```
in windows system run
```bash
.\venv\Scripts\Activate.ps1
```
then run
```bash
pip install -r requirements
```

This installs:
- Core system dependencies
- Streamlit web framework
- pandas, matplotlib, wordcloud for visualizations


### 3. Start the Web UI

```bash
streamlit run app.py
```

The UI will open in your browser at `http://localhost:8501`

### 4. Use the System

#### Upload a Course
1. Go to **Course Management**
2. Enter course name and ID
3. Upload your file (PDF, DOCX, MD, TXT)
4. Click **Index Course**
5. Wait for indexing to complete (~1-2 minutes)

#### Ask Questions
1. Select your course from the sidebar
2. Go to **Ask Questions**
3. Type your question
4. Click **Ask**
5. View the AI's response with citations

#### View Analytics
1. Go to **Analytics**
2. See question statistics
3. Generate word clouds
4. Export reports