import requests
import sys
import time
import threading
import numpy as np
import whisper
import sounddevice as sd
from queue import Queue
from rich.console import Console
import langchain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from tts import TextToSpeechService

import faiss
from datasets import load_dataset
from sklearn.preprocessing import normalize
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

console = Console()
stt = whisper.load_model("base.en")
tts = TextToSpeechService()

template = """
    You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less 
    than 20 words.
    
    The conversation transcript is as follows:
    {history}
    
    And here is the user's follow-up: {input}
    
    Your response:
"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
chain = ConversationChain(
    prompt=PROMPT,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="Assistant:"),
    llm=Ollama(model="llama3.2:1b"),
)

QUESTION_CONTEXT_SIM_TH = 0.5


def check_ollama_server(host="http://localhost:11434"):
    """
    Checks if the Ollama server is running by sending a GET request to the /api/tags endpoint.

    Args:
        host (str): The base URL of the Ollama server.

    Returns:
        bool: True if the server is running, False otherwise.
    """
    try:
        response = requests.get(f"{host}/api/tags", timeout=2)
        if response.status_code == 200:
            return True
        else:
            console.print(f"[red]Unexpected status from Ollama server: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Ollama server is not running or unreachable at {host}")
        console.print(f"[red]Error: {e}")
        return False


def record_audio(stop_event, data_queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.

    Args:
        stop_event (threading.Event): An event that, when set, signals the function to stop recording.
        data_queue (queue.Queue): A queue to which the recorded audio data will be added.

    Returns:
        None
    """
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)


def transcribe(audio_np: np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.

    Args:
        audio_np (numpy.ndarray): The audio data to be transcribed.

    Returns:
        str: The transcribed text.
    """
    result = stt.transcribe(audio_np, fp16=False)  # Set fp16=True if using a GPU
    text = result["text"].strip()
    return text


def get_llm_response(text: str) -> str:
    """
    Generates a response to the given text using the Llama-2 language model.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The generated response.
    """
    response = chain.predict(input=text)
    if response.startswith("Assistant:"):
        response = response[len("Assistant:") :].strip()
    return response


def play_audio(sample_rate, audio_array):
    """
    Plays the given audio data using the sounddevice library.

    Args:
        sample_rate (int): The sample rate of the audio data.
        audio_array (numpy.ndarray): The audio data to be played.

    Returns:
        None
    """
    sd.play(audio_array, sample_rate)
    sd.wait()


def get_context_embeddings(dataset) -> FAISS:
    """
    Compute embeddings for contexts, and return them in vectorstore
    :param dataset: train/test dataset
        :return: langchain_community.vectorstores.faiss.FAISS
    """
    texts = [sample["context"] for sample in dataset]
    documents = [Document(page_content=text) for text in texts]

    embedding_model = HuggingFaceEmbeddings()
    embeddings = [embedding_model.embed_query(doc.page_content) for doc in documents]

    # Normalise for cosine similarity
    normalized_embeddings = normalize(embeddings)

    # FAISS IndexFlatIP (inner product)
    dimension = len(normalized_embeddings[0])
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(normalized_embeddings))

    # FAISS vectorstore
    vectorstore = FAISS(embedding_function=embedding_model, index=index, docstore=None, index_to_docstore_id=None)

    # Connect documents
    docstore = InMemoryDocstore(dict(enumerate(documents)))
    vectorstore.docstore = docstore
    vectorstore.index_to_docstore_id = dict(enumerate(range(len(documents))))

    return vectorstore


if __name__ == "__main__":
    ds = load_dataset("neural-bridge/rag-dataset-12000")['test'] # computing embeddings for train too heavy

    console.print("[cyan]Assistant starting...")

    if not check_ollama_server():
        console.print("[red]Please start the Ollama server by running 'ollama serve'")
        sys.exit(1)

    console.print("[cyan]Assistant ready! Press Ctrl+C to exit.")

    vectorstore = get_context_embeddings(ds)

    try:
        while True:
            console.input(
                "Press Enter to start recording, then press Enter again to stop."
            )

            data_queue = Queue()  # type: ignore[var-annotated]
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue),
            )
            recording_thread.start()

            input()
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="earth"):
                    text = transcribe(audio_np)
                console.print(f"[yellow]You: {text}")

                # Find relevant context
                with console.status("Searching for relevant context...", spinner="earth"):
                    document, score = vectorstore.similarity_search_with_score(text, k=1)[0]
                    if score >= QUESTION_CONTEXT_SIM_TH:
                        console.print(f"[green]Found helpful context: {document.page_content}")
                        # Add context to the prompt template
                        text = f"{text}\n\nContext: {document.page_content}"

                with console.status("Generating response...", spinner="earth"):
                    response = get_llm_response(text)
                    sample_rate, audio_array = tts.long_form_synthesize(response)

                console.print(f"[cyan]Assistant: {response}")
                play_audio(sample_rate, audio_array)
            else:
                console.print(
                    "[red]No audio recorded. Please ensure your microphone is working."
                )

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")