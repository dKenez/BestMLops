import gradio as gr

from bestmlops.model import classify_digit

# Create Gradio interface
iface = gr.Interface(
    fn=classify_digit,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="MNIST Digit Classification 🔢",
    description="Upload a handwritten digit image (0-9) to recognize it using MNIST-Digits-SigLIP2.",
)


def deploy_model():
    """
    Function to deploy the model.
    This function is called when the Gradio app is launched.
    """
    iface.launch()


# Launch the app
if __name__ == "__main__":
    deploy_model()
