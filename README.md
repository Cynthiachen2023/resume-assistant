## AI Job Assistant

An intelligent job assistant that helps users analyze and improve their resumes using AI.
Upload your resume or project summaries, chat with the assistant, and even paste a job description (JD) to get a match analysis and personalized suggestions.


### Setup
1. Clone the repository
git clone https://github.com/Cynthiachen2023/resume-assistant.git

    ```bash
    cd resume-assistant
    ```

2. Create a virtual environment
    ```bash
    python -m venv myenv
    source myenv/bin/activate      # macOS/Linux
    myenv\Scripts\activate         # Windows
    ```
3. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

4. Create an .env file
    ```bash
    GOOGLE_API_KEY=your_google_api_key
    GEMINI_MODEL_NAME= the model you want to use
    ```


5. Run the app
streamlit run app.py

### How to Use

1. Upload your resume or work experience files.

2. Click “Build Vector Index” to create your knowledge base.

3. Start chatting — ask:

    “How can I improve my project descriptions?”

    “Which experiences should I highlight?”

    “Here’s a JD — can you analyze my match and suggest improvements?”