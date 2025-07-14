## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/alexandrosDimos/academic_research_graph.git

2. Download VSCode and install the Pythn extenstion

3. Download python distribution, as of now it works with `python3.10.9` or `python3.10.11`.

4. Press `Ctrl+Shift+P` to select the pyhon interpreter you downloaded.

4. Create a python virtual environment:
    ```bash
    python -m venv venv

5. Activate the virtual environment:
    ```bash
    .\venv\Scripts\activate

6. Install the requirements
    ```bash
    pip install -r .\requirements.txt

7. Use Ollama. Download ollama to your computer from [Ollama](https://ollama.com/)

8. Download any model you want I would recommend llama3.2 or gemma3:4b, since they are both light models
    ```bash
    ollama pull <model-name>

9. Uncomment `llm = ChatOllama(model="llama3.2", temperature=0.5)` and comment `llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)`

10. Switch to your branch. I have created 2 new branches on Github one for each, one named `vasilis` and one `konstantina` Run the following commands:
    ```bash
    git fetch origin
    git checkout feature/<your name>
    