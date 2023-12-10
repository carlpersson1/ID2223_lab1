from transformers import pipeline
import gradio as gr
import random, string

# Download the model pipeline
pipe = pipeline(model="carlpersson/Whisper-Small-De")  # change to "your-username/the-name-you-picked"

# Create a dictionary of example sentences
dictionary = {
    'Guten Morgen': 'Good morning',
    'Wie geht es dir': 'How are you',
    'Ich heiße Thomas': 'My name is Thomas',
    'Können Sie das bitte wiederholen': 'Can you please repeat that?',
    'Ich verstehe nicht': 'I do not understand.',
    'Kann ich Ihnen helfen': 'Can I help you?',
    'Wo ist die Toilette': 'Where is the bathroom?',
    'Ich hätte gerne einen Kaffee': 'I would like a coffee.',
    'Ich lerne Deutsch': 'I am learning German.',
    'Entschuldigung, wo ist der Bahnhof': 'Excuse me, where is the train station?',
    'Das Wetter ist heute schön': 'The weather is nice today.',
    'Ich spreche ein bisschen Deutsch': 'I speak a little German.'
}

def generate_practice_sentence(difficulty):
    """Generate a random sentence and translation pair from the dictionary"""
    if type(difficulty) is not str:
        difficulty = difficulty.value
    hindi_sent, english_sent = random.choice(list(dictionary.items()))
    if difficulty == 'Hard':
        display_text = 'Hidden (hard difficulty)'
    else:
        display_text = hindi_sent
    return hindi_sent, english_sent, display_text


def transcribe(audio):
    """Function used to evaluate the model using the audio input"""
    text = pipe(audio)["text"]
    return text


mispronunced = []


async def compare_transcription(hindi_sentence, audio):
    """Returns the transcription and a stylized version where the correctly pronounced words are green and the
    incorrectly pronounced words are red."""
    if audio is None:
        return '', 'Your pronunciation will be evaluated here!'
    hindi_words = hindi_sentence.lower().split()
    # Remove punctuation and capital letters to make it easier to recognize well pronounced words
    transcription = transcribe(audio).translate(str.maketrans('', '', string.punctuation)).lower()
    transcribed_words = transcription.split()
    performance = []
    for i, word in enumerate(transcribed_words):
        if i < len(hindi_words) and word == hindi_words[i]:
            color = 'green'
        else:
            color = 'red'
            if i < len(hindi_words):
                mispronunced.append(f'Correct pronunciation: {hindi_words[i]}, Your pronunciation: {word}')
        performance.append(f"<span style='color:{color}; font-size: 20px;'>{word}</span>")
    colored_transcription = ' '.join(performance)
    # Wrapping in a styled div
    styled_transcription = f"Your pronunciation evaluation: {colored_transcription}"

    # Create history
    mispronounced = '\n'.join(mispronunced)

    return transcription, styled_transcription, mispronounced


with gr.Blocks() as demo:
    # Write title
    gr.Markdown("<h1 style='text-align: center;'>Hindi Practice using Whisper - Demo</h1>")
    # Initialize the hindi sentence and translation
    hindi_sentence, english_translation, display_text = generate_practice_sentence('Normal')
    hindi_sentence = gr.State(hindi_sentence)

    # Create UI
    with gr.Row():
        with gr.Column():
            # Audio input with submit and clear button
            audio_input = gr.Audio(sources=["microphone"], type="filepath")
            with gr.Row():
                clear_button = gr.Button("Clear")
                submit_button = gr.Button("Submit Audio")
        with gr.Column():
            # Displays the hindi and sentence translated to english
            sentence_display = gr.Label(value=display_text, label="Hindi Sentence")
            translation_display = gr.Label(value=english_translation, label="English Translation")

    # Shows the transcription of what you said and a comparison to what you were supposed to say
    transcription_output = gr.Label(value='', label="Transcribed Text")
    with gr.Row():
        with gr.Column():
            comparison_output = gr.Markdown('Your pronunciation will be evaluated here!')
        with gr.Column():
            history = gr.Textbox(value='', label='History of incorrect pronunciation', max_lines=4)

    new_sentence_button = gr.Button("New Random Sentence")

    # Create a difficulty selector
    with gr.Row():
        gr.Column()
        with gr.Column(scale=1):
            difficulty_selector = gr.Radio(choices=['Normal', 'Hard'], label='Select Difficulty', value='Normal',
                                           info='Normal: Shows both the sentence in Hindi and English. '
                                                'Hard: Shows only the sentence in English')
        gr.Column()

    # Generate a new sentence when a new difficulty is selected and hide the sentence to be spoken
    difficulty_selector.change(generate_practice_sentence, inputs=difficulty_selector,
                               outputs=[hindi_sentence, translation_display, sentence_display])
    # Create a new sentence
    new_sentence_button.click(
        generate_practice_sentence,
        inputs=difficulty_selector,
        outputs=[hindi_sentence, translation_display, sentence_display]
    )
    # Compare the transcription to the sentence when audio is submitted
    submit_button.click(
        compare_transcription,
        inputs=[hindi_sentence, audio_input],
        outputs=[transcription_output, comparison_output, history]
    )
    # Clear the audio
    clear_button.click(
        lambda: None,
        inputs=[],
        outputs=[audio_input]
    )

# I want to create a history of performance

# Launch the app
demo.launch()
