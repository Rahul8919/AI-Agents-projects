import os
import io
from typing import Generator, List
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

# -------------------------------------------------------------
# Environment & OpenAI client setup
# -------------------------------------------------------------

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file. Server cannot start.")

client = OpenAI(api_key = openai_api_key)
MODEL_NAME = "gpt-4o-mini"


# below  are Available MCP tools
# explain_concept – Stream an explanation of any concept at a chosen complexity level (1 = kid-friendly … 5 = expert).
# summarize_text – Stream a concise summary of long text, controllable with a compression_ratio.
# generate_flashcards – Produce study flashcards (Q/A pairs) for rapid review of a topic.
# quiz_me – Stream an interactive quiz on a topic; reveals answers after questions.


# Let's define the mapping from integers (1-5) to explanation levels
EXPLANATION_LEVELS = {
    1: "like I'm 5 years old",
    2: "like I'm 10 years old",
    3: "like a high school student",
    4: "like a college student",
    5: "like an expert in the field",
}

# Define the explain concept function
def explain_concept(question: str, level: int) -> Generator[str, None, None]:
    """Stream an explanation of *question* at the requested *level* (1-5)."""

    if not question.strip():
        yield "Error: question cannot be blank."
        return

    level_desc = EXPLANATION_LEVELS.get(level, "clearly and concisely")
    system_prompt = "You are a helpful AI Tutor. Explain the following concept " f"{level_desc}."

    _stream = client.chat.completions.create(
        model = MODEL_NAME,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        stream = True,
        temperature = 0.7,
    )

    partial = ""

    for chunk in _stream:
        delta = getattr(chunk.choices[0].delta, "content", None)

        if delta:
            partial += delta
            yield partial
            
            
# Define the summarize text function
def summarize_text(text: str, compression_ratio: float = 0.3) -> Generator[str, None, None]:
    """Stream a summary of *text* compressed to roughly *compression_ratio* length.

    *compression_ratio* should be between 0.1 and 0.8.
    """

    if not text.strip():
        yield "Error: text cannot be blank."
        return

    ratio = max(0.1, min(compression_ratio, 0.8))

    system_prompt = (
        "You are a world-class summarizer. Reduce the following text to about "
        f"{int(ratio*100)}% of its original length while preserving key ideas."
    )

    _stream = client.chat.completions.create(
        model = MODEL_NAME,
        messages = [
            {"role": "system", "content": system_prompt},
             {"role": "user", "content": text},
    ],
    stream = True,
    temperature = 0.5,
    )

    partial = ""

    for chunk in _stream:
         delta = getattr(chunk.choices[0].delta, "content", None)
         if delta:
              partial += delta
              yield partial
              
              
# Define the generate flashcards function
def generate_flashcards(topic: str, num_cards: int = 5) -> Generator[str, None, None]:
    """Stream *num_cards* Q/A flashcards for *topic* in JSON lines format."""

    if num_cards < 1 or num_cards > 20:
        yield "Error: num_cards must be between 1 and 20."
        return

    if not topic.strip():
        yield "Error: topic cannot be blank."
        return

    system_prompt = (
        "You are an AI that generates study flashcards."
        'Return each flashcard on its own line as JSON: {"q": <question>, "a": <answer>}'
    )

    user_prompt = f"Create {num_cards} flashcards about {topic}."

    _stream = client.chat.completions.create(
        model = MODEL_NAME,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream = True,
        temperature = 0.8,
    )

    partial = ""

    for chunk in _stream:
        delta = getattr(chunk.choices[0].delta, "content", None)
        if delta:
            partial += delta
            yield partial
            
            
# Define the Quiz Me function
def quiz_me(topic: str, level: int = 3, num_questions: int = 5) -> Generator[str, None, None]:
    """Stream a quiz with numbered Qs then reveal answers after all questions."""

    if num_questions < 1 or num_questions > 15:
        yield "Error: num_questions must be between 1 and 15."
        return

    if not topic.strip():
        yield "Error: topic cannot be blank."
        return

    level_desc = EXPLANATION_LEVELS.get(level, "at an intermediate level")

    system_prompt = (
        "You are an AI quiz master. Generate a quiz of multiple-choice questions "
        f"about {topic} {level_desc}. Number the questions. After listing all Qs, "
        "add an \nANSWER KEY section with the correct options."
    )

    _stream = client.chat.completions.create(
        model = MODEL_NAME,
        messages = [{"role": "system", "content": system_prompt}],
        stream = True,
        temperature = 0.7,
    )
    partial = ""

    for chunk in _stream:
        delta = getattr(chunk.choices[0].delta, "content", None)
        if delta:
            partial += delta
            yield partial
            
#building gradio UI interface 
def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# AI Tutor MCP Toolkit - Demo Console")

        with gr.Tab("Explain Concept"):
            q = gr.Textbox(label="Concept / Question")
            lvl = gr.Slider(1, 5, value=3, step=1, label="Explanation Level")
            out1 = gr.Markdown()
            gr.Button("Explain").click(explain_concept, inputs=[q, lvl], outputs=out1)

        with gr.Tab("Summarize Text"):
            txt = gr.Textbox(lines=8, label="Long Text")
            ratio = gr.Slider(0.1, 0.8, value=0.3, step=0.05, label="Compression Ratio")
            out2 = gr.Markdown()
            gr.Button("Summarize").click(summarize_text, inputs=[txt, ratio], outputs=out2)

        with gr.Tab("Flashcards"):
            topic_fc = gr.Textbox(label="Topic")
            n_fc = gr.Slider(1, 20, value=5, step=1, label="# Cards")
            out3 = gr.Markdown()
            gr.Button("Generate").click(generate_flashcards, inputs=[topic_fc, n_fc], outputs=out3)

        with gr.Tab("Quiz Me"):
            topic_q = gr.Textbox(label="Topic")
            lvl_q = gr.Slider(1, 5, value=3, step=1, label="Difficulty Level")
            n_q = gr.Slider(1, 15, value=5, step=1, label="# Questions")
            out4 = gr.Markdown()
            gr.Button("Start Quiz").click(quiz_me, inputs=[topic_q, lvl_q, n_q], outputs=out4)

        return demo


if __name__ == "__main__":
    print("Starting AI Tutor MCP Toolkit on port 7860...")
    build_demo().launch(server_name="0.0.0.0", mcp_server=True) # for creating a public link for gradio place 'share=True' in launch

# --- END OF FILE ---