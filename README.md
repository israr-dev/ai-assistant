How to Run Your Versatile AI Assistant:
Follow these steps to set up and run your AI assistant on your Windows machine:
	1. Install Python: If you don't have Python installed, download it from python.org and follow the installation instructions. Make sure to check the "Add Python to PATH" option during installation.
	2. Install Ollama:
		○ Download and install Ollama for Windows from the official website: ollama.com/download
		○ Once installed, open your command prompt or PowerShell and pull the models you want to use. For this application, you'll need at least two: 
		  § A general response model (e.g., llama3): 
      Bash
      ollama pull llama3			
      § An embedding model (e.g., nomic-embed-text): 
      Bash
      ollama pull nomic-embed-text
      Note: You can choose other models if you prefer, but ensure they are pulled. The app.py uses llama3 and nomic-embed-text by default.
	
 3. Inside your Project Directory:
		○ Git clone https://github.com/israr-dev/ai-assistant.git
		
	4. Install Python Dependencies:
		○ Open your command prompt or PowerShell.
		○ Navigate to your ai-assistant directory: 
      Bash
      cd path\to\your\ai-assistant
		
    ○ It's highly recommended to create a virtual environment to manage dependencies: 
      Bash
      python -m venv venv

    ○ Activate the virtual environment: 
      Bash
      .\venv\Scripts\activate

	  ○ Install the required Python libraries: 
      Bash
      pip install Flask requirements.txt
		
	5. Run the Flask Application:
		○ With your virtual environment still active, run the Flask app from the ai-assistant directory: 
      Bash
      python app.py
		
	6. Access the Web Interface:
		○ Open your web browser and go to http://127.0.0.1:5000.
		
Now you can upload your PDF document to build the knowledge base and then ask questions. The AI assistant will use the content from the uploaded PDF for specific queries and its general knowledge for other questions.![image](https://github.com/user-attachments/assets/2bb298f6-b440-4edb-94b9-d434c1b5192f)
